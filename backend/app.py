"""
PRECIP.LAB — Backend
Flask server that runs the full dual-transform stacked ensemble pipeline
(monthly) or log1p-transform stacked ensemble (daily) on a user-uploaded
NASA/POWER CSV and returns metrics + base64-PNG plots.
"""
import os, io, base64, traceback, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

# ------------------------------------------------------------------
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

# ------------------------------------------------------------------
# Plot styling (dark editorial theme)
# ------------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': '#0f1419',
    'axes.facecolor':   '#1a1f2e',
    'savefig.facecolor':'#0f1419',
    'axes.edgecolor':   '#3a4556',
    'axes.labelcolor':  '#d1d5db',
    'xtick.color':      '#9ca3af',
    'ytick.color':      '#9ca3af',
    'grid.color':       '#2a3141',
    'text.color':       '#e5e7eb',
    'axes.titlecolor':  '#f3f4f6',
    'font.family':      'DejaVu Sans',
    'font.size':        10,
})


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-9)
    beta  = np.mean(sim) / (np.mean(obs) + 1e-9)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mse  = float(np.mean((y_pred - y_true) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'KGE': kge(y_true, y_pred)}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def detect_format_and_load(raw_bytes):
    text = raw_bytes.decode('utf-8', errors='ignore')
    lines = text.splitlines()
    monthly_idx, daily_idx = -1, -1
    for i, line in enumerate(lines):
        up = line.strip().upper()
        if monthly_idx < 0 and up.startswith('PARAMETER') and 'YEAR' in up and 'JAN' in up:
            monthly_idx = i
        if daily_idx < 0 and up.startswith('YEAR') and 'DOY' in up:
            daily_idx = i
        if monthly_idx >= 0 or daily_idx >= 0:
            break
    if monthly_idx >= 0:
        fmt, skip = 'monthly', monthly_idx
    elif daily_idx >= 0:
        fmt, skip = 'daily', daily_idx
    else:
        raise ValueError(
            "Could not detect NASA/POWER CSV format. Expected a monthly file with "
            "'PARAMETER,YEAR,JAN,...' header or a daily file with 'YEAR,DOY,...' header."
        )
    df = pd.read_csv(io.BytesIO(raw_bytes), skiprows=skip)
    df.columns = df.columns.str.strip()
    df.replace(-999, np.nan, inplace=True)
    return fmt, df


MONTH_MAP = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
             'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def reshape_monthly(raw_df):
    df = raw_df.copy()
    drop_cols = [c for c in ['ANN'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    melted = df.melt(id_vars=['PARAMETER', 'YEAR'], var_name='MON_STR', value_name='VALUE')
    melted['MONTH'] = melted['MON_STR'].map(MONTH_MAP)
    melted = melted.dropna(subset=['MONTH'])
    melted['MONTH'] = melted['MONTH'].astype(int)
    melted.drop(columns=['MON_STR'], inplace=True)
    out = melted.pivot_table(index=['YEAR', 'MONTH'], columns='PARAMETER',
                             values='VALUE').reset_index()
    out.columns.name = None
    out['DATE'] = pd.to_datetime(out['YEAR'].astype(str) + '-' +
                                 out['MONTH'].astype(str) + '-01')
    out = out.sort_values('DATE').reset_index(drop=True)
    out.replace(-999, np.nan, inplace=True)
    return out


MET = ['T2M', 'T2MDEW', 'T2M_MAX', 'T2M_MIN', 'QV2M', 'RH2M',
       'PS', 'WS2M', 'WS10M', 'WD10M', 'WD2M', 'GWETTOP']
TARGET = 'PRECTOTCORR'


def aggregate_daily_to_monthly(raw_df):
    """
    Roll a NASA/POWER daily frame up to monthly, mirroring how NASA/POWER
    produces its own monthly product from daily values.

    Aggregation rules:
      PRECTOTCORR (mm/day)   → mean (stays a daily-rate number)
      T2M, T2MDEW, QV2M,
      RH2M, PS, WS2M, WS10M,
      GWETTOP                → mean
      T2M_MAX                → max
      T2M_MIN                → min
      WD10M, WD2M            → circular mean via (u, v) components
    """
    df = raw_df.copy()
    # Need YEAR + DOY (or DATE) to compute MONTH
    if 'DATE' not in df.columns:
        if 'YEAR' not in df.columns or 'DOY' not in df.columns:
            raise ValueError("Daily frame needs YEAR and DOY (or a DATE) column "
                             "to aggregate to monthly.")
        df['DATE'] = pd.to_datetime(
            df['YEAR'].astype(int).astype(str) + '-' +
            df['DOY'].astype(int).astype(str), format='%Y-%j')

    df['YEAR']  = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month

    # Circular mean of wind direction via u/v components of a unit vector
    for dir_col in ['WD10M', 'WD2M']:
        if dir_col in df.columns:
            rad = np.radians(df[dir_col])
            df[f'_{dir_col}_u'] = np.cos(rad)
            df[f'_{dir_col}_v'] = np.sin(rad)

    agg_rules = {}
    for c in ['T2M', 'T2MDEW', 'QV2M', 'RH2M', 'PS',
              'WS2M', 'WS10M', 'GWETTOP', TARGET]:
        if c in df.columns:
            agg_rules[c] = 'mean'
    if 'T2M_MAX' in df.columns: agg_rules['T2M_MAX'] = 'max'
    if 'T2M_MIN' in df.columns: agg_rules['T2M_MIN'] = 'min'
    for dir_col in ['WD10M', 'WD2M']:
        if f'_{dir_col}_u' in df.columns:
            agg_rules[f'_{dir_col}_u'] = 'mean'
            agg_rules[f'_{dir_col}_v'] = 'mean'

    mon = df.groupby(['YEAR', 'MONTH'], as_index=False).agg(agg_rules)

    # Rebuild directions from u/v means
    for dir_col in ['WD10M', 'WD2M']:
        u_col, v_col = f'_{dir_col}_u', f'_{dir_col}_v'
        if u_col in mon.columns:
            ang = np.degrees(np.arctan2(mon[v_col], mon[u_col]))
            mon[dir_col] = (ang + 360) % 360
            mon.drop(columns=[u_col, v_col], inplace=True)

    mon['DATE'] = pd.to_datetime(
        mon['YEAR'].astype(str) + '-' + mon['MONTH'].astype(str) + '-01')
    mon = mon.sort_values('DATE').reset_index(drop=True)
    return mon


# ------------------------------------------------------------------
# Feature engineering — MONTHLY
# ------------------------------------------------------------------
def engineer_monthly(df):
    df = df.copy()
    missing = [c for c in MET + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in monthly data: {missing}")

    df['MON_SIN']  = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MON_COS']  = np.cos(2 * np.pi * df['MONTH'] / 12)
    df['MON_SIN2'] = np.sin(4 * np.pi * df['MONTH'] / 12)
    df['MON_COS2'] = np.cos(4 * np.pi * df['MONTH'] / 12)

    df['IS_MONSOON']      = ((df.MONTH >= 6) & (df.MONTH <= 9)).astype(int)
    df['IS_PRE_MONSOON']  = ((df.MONTH >= 3) & (df.MONTH <= 5)).astype(int)
    df['IS_POST_MONSOON'] = ((df.MONTH >= 10) & (df.MONTH <= 11)).astype(int)
    df['IS_WINTER']       = ((df.MONTH == 12) | (df.MONTH <= 2)).astype(int)

    df['TEMP_RANGE']    = df.T2M_MAX - df.T2M_MIN
    df['DEWPT_DEPRESS'] = df.T2M - df.T2MDEW
    df['MOISTURE_FLUX'] = df.QV2M * df.WS10M
    df['HUMID_TEMP']    = df.RH2M * df.T2M
    df['WIND_U10']      = df.WS10M * np.cos(np.radians(df.WD10M))
    df['WIND_V10']      = df.WS10M * np.sin(np.radians(df.WD10M))
    df['WIND_U2']       = df.WS2M  * np.cos(np.radians(df.WD2M))
    df['WIND_V2']       = df.WS2M  * np.sin(np.radians(df.WD2M))
    df['SAT_DEFICIT']   = 100 - df.RH2M
    df['MOIST_INSTAB']  = df.QV2M * (df.T2M - df.T2M_MIN)
    df['PRESS_TEMP']    = df.PS * df.T2M
    df['SOIL_HUMID']    = df.GWETTOP * df.RH2M
    df['WIND_SHEAR']    = df.WS10M - df.WS2M
    df['QV2M_SQ']       = df.QV2M ** 2
    df['RH2M_SQ']       = df.RH2M ** 2
    df['GWETTOP_SQ']    = df.GWETTOP ** 2

    for lag in [1, 2, 3, 6, 12]:
        df[f'RAIN_lag{lag}'] = df[TARGET].shift(lag)
    for var in MET:
        df[f'{var}_lag1'] = df[var].shift(1)
        df[f'{var}_lag2'] = df[var].shift(2)
        df[f'{var}_d1']   = df[var] - df[var].shift(1)

    for w in [3, 6, 12]:
        roll = df[TARGET].shift(1).rolling(w, min_periods=1)
        df[f'RAIN_rm{w}']   = roll.mean()
        df[f'RAIN_rs{w}']   = roll.std()
        df[f'RAIN_rx{w}']   = roll.max()
        df[f'RAIN_rmin{w}'] = roll.min()
        df[f'WET_MONTHS_{w}'] = (df[TARGET].shift(1) > 1.0).rolling(w, min_periods=1).sum()

    for var in ['T2M', 'QV2M', 'RH2M', 'PS', 'GWETTOP', 'WS10M']:
        for w in [3, 6, 12]:
            df[f'{var}_rm{w}'] = df[var].shift(1).rolling(w, min_periods=1).mean()
        df[f'{var}_rs12'] = df[var].shift(1).rolling(12, min_periods=1).std()

    df['PS_tend3'] = df.PS - df.PS.shift(3)
    df['PS_tend6'] = df.PS - df.PS.shift(6)

    clim = np.full(len(df), np.nan)
    for mval in range(1, 13):
        mask = df.MONTH == mval
        idx  = df.index[mask]
        yrs  = df.loc[mask, 'YEAR'].values
        vals = df.loc[mask, TARGET].values
        for j, ix in enumerate(idx):
            past = vals[yrs < yrs[j]]
            if len(past) > 0:
                clim[ix] = past.mean()
    df['RAIN_MON_CLIM'] = clim

    df['RAIN_LASTYR']  = df[TARGET].shift(12)
    df['RAIN_2YRSAGO'] = df[TARGET].shift(24)
    df['RAIN_CUM_YR'] = (
        df.groupby('YEAR')[TARGET]
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True))
    df['RAIN_ANOM'] = df[TARGET].shift(1) - df['RAIN_MON_CLIM']

    is_dry = (df[TARGET].shift(1) < 1.0).astype(int).to_numpy().copy()
    for i in range(1, len(is_dry)):
        is_dry[i] = is_dry[i - 1] + 1 if is_dry[i] == 1 else 0
    df['DRY_STREAK'] = is_dry

    is_wet = (df[TARGET].shift(1) >= 1.0).astype(int).to_numpy().copy()
    for i in range(1, len(is_wet)):
        is_wet[i] = is_wet[i - 1] + 1 if is_wet[i] == 1 else 0
    df['WET_STREAK'] = is_wet

    df['MONTH_x_QV2M']    = df.MONTH * df.QV2M
    df['MONTH_x_RH2M']    = df.MONTH * df.RH2M
    df['MONTH_x_GWETTOP'] = df.MONTH * df.GWETTOP

    df_clean = df.dropna().reset_index(drop=True)
    exclude = {'YEAR', 'DATE', 'MONTH', TARGET}
    feats = sorted([c for c in df_clean.columns if c not in exclude])
    return df_clean, feats


# ------------------------------------------------------------------
# Feature engineering — DAILY
# ------------------------------------------------------------------
def engineer_daily(df):
    df = df.copy()
    if 'DATE' not in df.columns:
        if 'DOY' not in df.columns or 'YEAR' not in df.columns:
            raise ValueError("Daily CSV must contain YEAR and DOY columns.")
        df['DATE'] = pd.to_datetime(df['YEAR'].astype(int).astype(str) + '-' +
                                    df['DOY'].astype(int).astype(str),
                                    format='%Y-%j')
    df['MONTH'] = df['DATE'].dt.month
    df['DAY']   = df['DATE'].dt.day

    missing = [c for c in MET + [TARGET, 'DOY'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in daily data: {missing}")

    doy = df['DOY']
    df['DOY_SIN']  = np.sin(2 * np.pi * doy / 365.25)
    df['DOY_COS']  = np.cos(2 * np.pi * doy / 365.25)
    df['DOY_SIN2'] = np.sin(4 * np.pi * doy / 365.25)
    df['DOY_COS2'] = np.cos(4 * np.pi * doy / 365.25)
    df['MON_SIN']  = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MON_COS']  = np.cos(2 * np.pi * df['MONTH'] / 12)
    df['IS_MONSOON']      = ((df.MONTH >= 6) & (df.MONTH <= 9)).astype(int)
    df['IS_PRE_MONSOON']  = ((df.MONTH >= 3) & (df.MONTH <= 5)).astype(int)
    df['IS_POST_MONSOON'] = ((df.MONTH >= 10) & (df.MONTH <= 11)).astype(int)
    df['IS_WINTER']       = ((df.MONTH == 12) | (df.MONTH <= 2)).astype(int)
    df['WEEK'] = df.DATE.dt.isocalendar().week.astype(int)

    df['TEMP_RANGE']    = df.T2M_MAX - df.T2M_MIN
    df['DEWPT_DEPRESS'] = df.T2M - df.T2MDEW
    df['MOISTURE_FLUX'] = df.QV2M * df.WS10M
    df['HUMID_TEMP']    = df.RH2M * df.T2M
    df['WIND_U10']      = df.WS10M * np.cos(np.radians(df.WD10M))
    df['WIND_V10']      = df.WS10M * np.sin(np.radians(df.WD10M))
    df['WIND_U2']       = df.WS2M  * np.cos(np.radians(df.WD2M))
    df['WIND_V2']       = df.WS2M  * np.sin(np.radians(df.WD2M))
    df['SAT_DEFICIT']   = 100 - df.RH2M
    df['MOIST_INSTAB']  = df.QV2M * (df.T2M - df.T2M_MIN)
    df['PRESS_TEMP']    = df.PS * df.T2M
    df['SOIL_HUMID']    = df.GWETTOP * df.RH2M
    df['WIND_SHEAR']    = df.WS10M - df.WS2M
    df['QV2M_SQ']       = df.QV2M ** 2
    df['RH2M_SQ']       = df.RH2M ** 2
    df['GWETTOP_SQ']    = df.GWETTOP ** 2

    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f'RAIN_lag{lag}'] = df[TARGET].shift(lag)
    for var in MET:
        df[f'{var}_lag1'] = df[var].shift(1)
        df[f'{var}_lag3'] = df[var].shift(3)
        df[f'{var}_d1']   = df[var] - df[var].shift(1)

    for w in [3, 7, 14, 30]:
        roll = df[TARGET].shift(1).rolling(w, min_periods=1)
        df[f'RAIN_rm{w}']   = roll.mean()
        df[f'RAIN_rs{w}']   = roll.std()
        df[f'RAIN_rx{w}']   = roll.max()
        df[f'WET_DAYS_{w}'] = (df[TARGET].shift(1) > 0.1).rolling(w, min_periods=1).sum()

    for var in ['T2M', 'QV2M', 'RH2M', 'PS', 'GWETTOP', 'WS10M']:
        for w in [7, 14, 30]:
            df[f'{var}_rm{w}'] = df[var].shift(1).rolling(w, min_periods=1).mean()
        df[f'{var}_rs30'] = df[var].shift(1).rolling(30, min_periods=1).std()

    df['PS_tend3'] = df.PS - df.PS.shift(3)
    df['PS_tend7'] = df.PS - df.PS.shift(7)

    clim = np.full(len(df), np.nan)
    for dval in df.DOY.unique():
        mask = df.DOY == dval
        idx  = df.index[mask]
        yrs  = df.loc[mask, 'YEAR'].values
        vals = df.loc[mask, TARGET].values
        for j, ix in enumerate(idx):
            past = vals[yrs < yrs[j]]
            if len(past) > 0:
                clim[ix] = past.mean()
    df['RAIN_DOY_CLIM'] = clim

    df['RAIN_LASTYR'] = df[TARGET].shift(365)
    df['RAIN_CUM_YR'] = (
        df.groupby('YEAR')[TARGET]
        .apply(lambda x: x.shift(1).fillna(0).cumsum())
        .reset_index(level=0, drop=True))

    is_dry = (df[TARGET].shift(1) < 0.1).astype(int).to_numpy().copy()
    for i in range(1, len(is_dry)):
        is_dry[i] = is_dry[i - 1] + 1 if is_dry[i] == 1 else 0
    df['DRY_STREAK'] = is_dry

    is_wet = (df[TARGET].shift(1) >= 0.1).astype(int).to_numpy().copy()
    for i in range(1, len(is_wet)):
        is_wet[i] = is_wet[i - 1] + 1 if is_wet[i] == 1 else 0
    df['WET_STREAK'] = is_wet

    df_clean = df.dropna().reset_index(drop=True)
    exclude = {'YEAR', 'DOY', 'DATE', 'MONTH', 'DAY', TARGET}
    feats = sorted([c for c in df_clean.columns if c not in exclude])
    return df_clean, feats


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_eda(df, freq='monthly'):
    df = df.copy()
    # Be tolerant — if caller passed a raw daily frame without DATE/MONTH,
    # derive them on the fly so this function is self-contained.
    if 'DATE' not in df.columns:
        if 'DOY' in df.columns and 'YEAR' in df.columns:
            df['DATE'] = pd.to_datetime(
                df['YEAR'].astype(int).astype(str) + '-' +
                df['DOY'].astype(int).astype(str),
                format='%Y-%j')
        elif 'YEAR' in df.columns and 'MONTH' in df.columns:
            df['DATE'] = pd.to_datetime(
                df['YEAR'].astype(int).astype(str) + '-' +
                df['MONTH'].astype(int).astype(str) + '-01')
    if 'MONTH' not in df.columns and 'DATE' in df.columns:
        df['MONTH'] = df['DATE'].dt.month

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ma_win = 12 if freq == 'monthly' else 30

    ax = axes[0, 0]
    ax.plot(df.DATE, df[TARGET], lw=0.6, alpha=0.55, color='#60a5fa')
    ax.plot(df.DATE, df[TARGET].rolling(ma_win).mean(), lw=1.8, color='#f87171',
            label=f'{ma_win}-pt MA')
    ax.set_title('Rainfall Time Series'); ax.set_ylabel('mm/day')
    ax.legend(); ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.hist(df[TARGET].dropna(), bins=50, color='#34d399',
            edgecolor='#064e3b', alpha=0.8)
    ax.set_yscale('log')
    ax.set_title('Rainfall Distribution (log y)'); ax.set_xlabel('mm/day')
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    mean_by_mon = df.groupby('MONTH')[TARGET].mean()
    ax.bar(mean_by_mon.index, mean_by_mon.values,
           color='#a78bfa', edgecolor='#3b0764')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_title('Monthly Climatology'); ax.set_ylabel('mean mm/day')
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.scatter(df['T2M'], df[TARGET], s=6, alpha=0.4, color='#fbbf24')
    ax.set_title('T2M vs Rainfall'); ax.set_xlabel('T2M (°C)')
    ax.set_ylabel('mm/day'); ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.scatter(df['QV2M'], df[TARGET], s=6, alpha=0.4, color='#f472b6')
    ax.set_title('QV2M vs Rainfall'); ax.set_xlabel('QV2M (g/kg)')
    ax.set_ylabel('mm/day'); ax.grid(alpha=0.25)

    ax = axes[1, 2]
    corrs = df[MET].corrwith(df[TARGET]).sort_values()
    ax.barh(corrs.index, corrs.values,
            color=['#f87171' if v < 0 else '#60a5fa' for v in corrs.values])
    ax.set_title('Correlation with Rainfall'); ax.axvline(0, color='white', lw=0.7)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig_to_b64(fig)


def plot_ts(dates, y_true, y_pred, title, color_pred='#ef4444'):
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(dates, y_true, lw=1.4, alpha=0.85, label='Actual', color='#60a5fa')
    ax.plot(dates, y_pred, lw=1.4, alpha=0.85, label='Predicted',
            color=color_pred, linestyle='--')
    ax.fill_between(dates, y_true, y_pred, alpha=0.12, color='#9ca3af')
    ax.set_title(title); ax.set_ylabel('Rainfall (mm/day)')
    ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_scatter(y_true, y_pred, title, color='#a78bfa'):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=22, alpha=0.55, edgecolors='white',
               linewidths=0.25, c=color)
    mx = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, mx], [0, mx], color='#f87171', lw=1.3, ls='--',
            label='Perfect prediction')
    ax.set_xlabel('Actual (mm/day)'); ax.set_ylabel('Predicted (mm/day)')
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ax.set_title(f'{title}\nR²={r2:.4f}  RMSE={rmse:.3f}')
    ax.legend(); ax.grid(alpha=0.25)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_residuals(dates, residuals, title='Residual Analysis'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(dates, residuals, lw=0.7, alpha=0.75, color='#22d3ee')
    ax.axhline(0, color='#f87171', ls='-', lw=1)
    ax.set_title('Residuals Over Time'); ax.set_ylabel('Residual (mm/day)')
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(residuals, bins=50, color='#c084fc', edgecolor='#3b0764', alpha=0.85)
    ax.axvline(0, color='#f87171', ls='--', lw=1)
    ax.set_title(f'Residual Distribution  μ={residuals.mean():.3f}  σ={residuals.std():.3f}')
    ax.grid(alpha=0.25)

    ax = axes[2]
    sorted_res = np.sort(residuals)
    n = len(sorted_res)
    theoretical = np.sort(np.random.default_rng(42).normal(0, residuals.std(), n))
    ax.scatter(theoretical, sorted_res, s=8, alpha=0.5, color='#34d399')
    lo, hi = theoretical.min(), theoretical.max()
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_title('Q-Q Plot (vs Normal)'); ax.set_xlabel('Theoretical')
    ax.set_ylabel('Observed'); ax.grid(alpha=0.25)

    plt.tight_layout()
    return fig_to_b64(fig)


def plot_model_comparison(model_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(model_metrics.keys())
    rmses = [model_metrics[n]['RMSE'] for n in names]
    r2s   = [model_metrics[n]['R2']   for n in names]
    colors = ['#60a5fa'] * (len(names) - 1) + ['#f59e0b']

    ax = axes[0]
    bars = ax.bar(range(len(names)), rmses, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Model RMSE Comparison (lower = better)')
    ax.set_ylabel('RMSE (mm/day)'); ax.grid(alpha=0.25, axis='y')
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(range(len(names)), r2s, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title('Model R² Comparison (higher = better)')
    ax.set_ylabel('R²'); ax.grid(alpha=0.25, axis='y')
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig_to_b64(fig)


def plot_rmse_by_month(test_months, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    vals = []
    for m in range(1, 13):
        mask = test_months == m
        if mask.sum() > 0:
            vals.append(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
        else:
            vals.append(0)
    ax.bar(labels, vals, color='#06b6d4', edgecolor='white', linewidth=0.5)
    ax.set_title('RMSE by Month (Test Set)'); ax.set_ylabel('RMSE (mm/day)')
    ax.grid(alpha=0.25, axis='y')
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_full_timeseries(train_dates, y_train, pred_train,
                         test_dates, y_test, pred_test):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(train_dates, y_train, lw=0.5, alpha=0.45, color='#6b7280',
            label='Train (actual)')
    ax.plot(train_dates, pred_train, lw=0.5, alpha=0.45, color='#34d399',
            label='Train (pred)')
    ax.plot(test_dates, y_test, lw=1.2, alpha=0.9, color='#60a5fa',
            label='Test (actual)')
    ax.plot(test_dates, pred_test, lw=1.2, alpha=0.9, color='#ef4444',
            ls='--', label='Test (pred)')
    ax.set_title('Full Timeline — Train & Test Actual vs Predicted')
    ax.set_ylabel('Rainfall (mm/day)')
    ax.legend(loc='upper left'); ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig_to_b64(fig)


def plot_shap(model, X, feature_names, title, max_display=15):
    """Returns (bar_b64, beeswarm_b64)."""
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    except Exception:
        return None, None

    def style_ax():
        ax = plt.gca()
        ax.set_facecolor('#1a1f2e')
        for lbl in ax.get_yticklabels() + ax.get_xticklabels():
            lbl.set_color('#d1d5db')
        try: ax.xaxis.label.set_color('#d1d5db')
        except Exception: pass

    fig1 = plt.figure(figsize=(10, 6), facecolor='#0f1419')
    shap.summary_plot(sv, X, plot_type='bar', max_display=max_display,
                      feature_names=list(feature_names), show=False)
    fig1.suptitle(title + ' — SHAP Feature Importance',
                  color='#f3f4f6', fontsize=12)
    style_ax()
    b1 = fig_to_b64(fig1)

    fig2 = plt.figure(figsize=(10, 6), facecolor='#0f1419')
    shap.summary_plot(sv, X, max_display=max_display,
                      feature_names=list(feature_names), show=False)
    fig2.suptitle(title + ' — SHAP Beeswarm',
                  color='#f3f4f6', fontsize=12)
    style_ax()
    b2 = fig_to_b64(fig2)
    return b1, b2


def plot_feature_engineering_bar(feature_cols):
    groups = {
        'Raw meteorological':  sum(1 for f in feature_cols if f in MET),
        'Cyclical / seasonal': sum(1 for f in feature_cols if any(k in f for k in
                                    ['SIN', 'COS', 'IS_', 'WEEK'])),
        'Interactions':        sum(1 for f in feature_cols if any(k in f for k in
                                    ['TEMP_RANGE', 'DEWPT_DEPRESS', 'MOISTURE_FLUX',
                                     'HUMID_TEMP', 'WIND_U', 'WIND_V', 'SAT_DEFICIT',
                                     'MOIST_INSTAB', 'PRESS_TEMP', 'SOIL_HUMID',
                                     'WIND_SHEAR', '_SQ', 'MONTH_x_'])),
        'Rain lags':           sum(1 for f in feature_cols if 'RAIN_lag' in f),
        'Met lags / deltas':   sum(1 for f in feature_cols if '_lag' in f and
                                    'RAIN' not in f) +
                                sum(1 for f in feature_cols if '_d1' in f),
        'Rolling stats':       sum(1 for f in feature_cols if any(k in f for k in
                                    ['_rm', '_rs', '_rx', '_rmin', 'WET_'])),
        'Climatology / tend':  sum(1 for f in feature_cols if any(k in f for k in
                                    ['CLIM', 'LASTYR', '2YRSAGO', 'CUM', 'ANOM',
                                     'STREAK', 'PS_tend'])),
    }
    groups = {k: v for k, v in groups.items() if v > 0}

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#60a5fa', '#a78bfa', '#f472b6', '#34d399',
              '#fbbf24', '#22d3ee', '#f87171']
    bars = ax.barh(list(groups.keys()), list(groups.values()),
                   color=colors[:len(groups)], edgecolor='white', linewidth=0.5)
    for b, v in zip(bars, groups.values()):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, str(v),
                va='center', color='#e5e7eb', fontsize=9)
    ax.set_title(f'Feature Families — Total: {len(feature_cols)} features')
    ax.set_xlabel('Count'); ax.grid(alpha=0.25, axis='x')
    plt.tight_layout()
    return fig_to_b64(fig)


# ------------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------------
def run_monthly_pipeline(df_full, progress=None):
    def p(msg):
        if progress is not None: progress.append(msg)

    p('Feature engineering...')
    df_clean, FEATURE_COLS = engineer_monthly(df_full)
    n = len(df_clean)
    if n < 50:
        raise ValueError(f"Not enough clean samples ({n}). Monthly pipeline needs ≥ 50.")

    i1, i2 = int(n * 0.70), int(n * 0.85)
    train = df_clean.iloc[:i1].copy()
    val   = df_clean.iloc[i1:i2].copy()
    test  = df_clean.iloc[i2:].copy()

    y_tr_raw, y_va_raw, y_te_raw = train[TARGET].values, val[TARGET].values, test[TARGET].values
    y_tr_sqrt, y_va_sqrt = np.sqrt(y_tr_raw), np.sqrt(y_va_raw)
    y_tr_log,  y_va_log  = np.log1p(y_tr_raw), np.log1p(y_va_raw)

    X_tr, X_va, X_te = train[FEATURE_COLS], val[FEATURE_COLS], test[FEATURE_COLS]

    LAG_FEATS = [c for c in FEATURE_COLS if any(x in c for x in
                 ['lag', 'rm', 'rs', 'rx', 'rmin', 'STREAK', 'WET_MONTHS',
                  'CUM', 'LASTYR', '2YRSAGO', 'CLIM', 'ANOM'])]
    if not LAG_FEATS:
        LAG_FEATS = FEATURE_COLS

    def to_raw_sqrt(p_): return np.clip(p_ ** 2, 0, None)
    def to_raw_log(p_):  return np.clip(np.expm1(p_), 0, None)

    p('Training LightGBM (sqrt, met only)...')
    lgb_sqrt = LGBMRegressor(objective='regression', num_leaves=20, learning_rate=0.08,
        n_estimators=3000, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    lgb_sqrt.fit(X_tr[MET], y_tr_sqrt,
                 eval_set=[(X_va[MET], y_va_sqrt)],
                 callbacks=[early_stopping(100, verbose=False)])

    p('Training XGBoost (sqrt, lag)...')
    xgb_sqrt = XGBRegressor(n_estimators=3000, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, early_stopping_rounds=100,
        tree_method='hist', reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=10, random_state=42, verbosity=0)
    xgb_sqrt.fit(X_tr[LAG_FEATS], y_tr_sqrt,
                 eval_set=[(X_va[LAG_FEATS], y_va_sqrt)], verbose=False)

    p('Training CatBoost (sqrt, all features)...')
    cat_sqrt = CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=4,
        l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=0)
    cat_sqrt.fit(X_tr, y_tr_sqrt, eval_set=(X_va, y_va_sqrt),
                 early_stopping_rounds=150, verbose=False)

    p('Training LightGBM (log1p)...')
    lgb_log = LGBMRegressor(objective='regression', num_leaves=15, learning_rate=0.05,
        n_estimators=2000, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=5, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)
    lgb_log.fit(X_tr[MET], y_tr_log,
                eval_set=[(X_va[MET], y_va_log)],
                callbacks=[early_stopping(100, verbose=False)])

    p('Training XGBoost (log1p)...')
    xgb_log = XGBRegressor(n_estimators=2000, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=100,
        tree_method='hist', reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0)
    xgb_log.fit(X_tr[LAG_FEATS], y_tr_log,
                eval_set=[(X_va[LAG_FEATS], y_va_log)], verbose=False)

    p('Training CatBoost (log1p)...')
    cat_log = CatBoostRegressor(iterations=5000, learning_rate=0.02, depth=5,
        l2_leaf_reg=5, loss_function='RMSE', random_seed=42, verbose=0)
    cat_log.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log),
                early_stopping_rounds=200, verbose=False)

    p('Fitting Ridge meta-learner (6-model stack)...')
    def stack_matrix(X_):
        return np.column_stack([
            to_raw_sqrt(lgb_sqrt.predict(X_[MET])),
            to_raw_sqrt(xgb_sqrt.predict(X_[LAG_FEATS])),
            to_raw_sqrt(cat_sqrt.predict(X_)),
            to_raw_log(lgb_log.predict(X_[MET])),
            to_raw_log(xgb_log.predict(X_[LAG_FEATS])),
            to_raw_log(cat_log.predict(X_)),
        ])
    S_tr = stack_matrix(X_tr); S_va = stack_matrix(X_va); S_te = stack_matrix(X_te)

    gs = GridSearchCV(Ridge(), {'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100]},
                      cv=TimeSeriesSplit(n_splits=min(5, max(2, len(val) // 6))),
                      scoring='neg_root_mean_squared_error')
    gs.fit(S_va, y_va_raw)
    meta = Ridge(alpha=gs.best_params_['alpha'])
    meta.fit(S_va, y_va_raw)

    pred_train = np.clip(meta.predict(S_tr), 0, None)
    pred_test  = np.clip(meta.predict(S_te), 0, None)

    p('Computing metrics...')
    base_preds = {
        'LGB (sqrt)':  to_raw_sqrt(lgb_sqrt.predict(X_te[MET])),
        'XGB (sqrt)':  to_raw_sqrt(xgb_sqrt.predict(X_te[LAG_FEATS])),
        'CAT (sqrt)':  to_raw_sqrt(cat_sqrt.predict(X_te)),
        'LGB (log1p)': to_raw_log(lgb_log.predict(X_te[MET])),
        'XGB (log1p)': to_raw_log(xgb_log.predict(X_te[LAG_FEATS])),
        'CAT (log1p)': to_raw_log(cat_log.predict(X_te)),
    }
    model_metrics = {nm: all_metrics(y_te_raw, pr) for nm, pr in base_preds.items()}
    model_metrics['STACKED'] = all_metrics(y_te_raw, pred_test)

    meta_weights = {lbl: float(w) for lbl, w in zip(
        ['LGB_sqrt', 'XGB_sqrt', 'CAT_sqrt', 'LGB_log', 'XGB_log', 'CAT_log'],
        meta.coef_)}
    meta_weights['_intercept']  = float(meta.intercept_)
    meta_weights['_best_alpha'] = float(gs.best_params_['alpha'])

    p('Generating plots...')
    plots = {
        'eda': plot_eda(df_full.dropna(subset=[TARGET]), freq='monthly'),
        'feature_engineering': plot_feature_engineering_bar(FEATURE_COLS),
        'model_comparison':    plot_model_comparison(model_metrics),
        'residuals':  plot_residuals(test['DATE'].values, y_te_raw - pred_test,
                                      'Stacked Ensemble Residuals'),
        'rmse_by_month':   plot_rmse_by_month(test['MONTH'].values, y_te_raw, pred_test),
        'full_timeseries': plot_full_timeseries(
            train['DATE'].values, y_tr_raw, pred_train,
            test['DATE'].values,  y_te_raw, pred_test),
    }

    model_plots = {}
    for name, pr in base_preds.items():
        model_plots[name] = {
            'timeseries': plot_ts(test['DATE'].values, y_te_raw, pr, f'{name} — Test Set'),
            'scatter':    plot_scatter(y_te_raw, pr, name),
        }
    model_plots['STACKED'] = {
        'timeseries': plot_ts(test['DATE'].values, y_te_raw, pred_test,
                               'Stacked Ensemble — Test Set',
                               color_pred='#f59e0b'),
        'scatter':    plot_scatter(y_te_raw, pred_test,
                                    'Stacked Ensemble', color='#f59e0b'),
    }
    plots['models'] = model_plots

    p('Computing SHAP (CatBoost main)...')
    b1, b2 = plot_shap(cat_sqrt, X_te, FEATURE_COLS,
                        'CatBoost (sqrt) — Main Model')
    plots['shap_bar'] = b1; plots['shap_beeswarm'] = b2

    p('Computing SHAP (LightGBM)...')
    b1l, b2l = plot_shap(lgb_sqrt, X_te[MET], MET, 'LightGBM (met only)')
    plots['shap_lgb_bar'] = b1l; plots['shap_lgb_beeswarm'] = b2l

    p('Computing SHAP (XGBoost)...')
    b1x, b2x = plot_shap(xgb_sqrt, X_te[LAG_FEATS], LAG_FEATS,
                          'XGBoost (lag/rolling)')
    plots['shap_xgb_bar'] = b1x; plots['shap_xgb_beeswarm'] = b2x

    return {
        'ok': True,
        'frequency':    'monthly',
        'n_samples':    int(n),
        'n_features':   int(len(FEATURE_COLS)),
        'train_period': [str(train['DATE'].min().date()), str(train['DATE'].max().date())],
        'val_period':   [str(val['DATE'].min().date()),   str(val['DATE'].max().date())],
        'test_period':  [str(test['DATE'].min().date()),  str(test['DATE'].max().date())],
        'split_sizes':  {'train': len(train), 'val': len(val), 'test': len(test)},
        'metrics':      model_metrics,
        'meta_weights': meta_weights,
        'feature_cols': FEATURE_COLS,
        'plots':        plots,
    }


def run_daily_pipeline(df_full, progress=None):
    def p(msg):
        if progress is not None: progress.append(msg)

    p('Feature engineering...')
    df_clean, FEATURE_COLS = engineer_daily(df_full)
    n = len(df_clean)
    if n < 500:
        raise ValueError(f"Not enough clean samples ({n}). Daily pipeline needs ≥ 500 "
                         f"(the 365-day lag drops the first year).")

    i1, i2 = int(n * 0.70), int(n * 0.85)
    train = df_clean.iloc[:i1].copy()
    val   = df_clean.iloc[i1:i2].copy()
    test  = df_clean.iloc[i2:].copy()

    y_tr_raw = train[TARGET].values
    y_va_raw = val[TARGET].values
    y_te_raw = test[TARGET].values
    y_tr = np.log1p(y_tr_raw); y_va = np.log1p(y_va_raw)
    X_tr, X_va, X_te = train[FEATURE_COLS], val[FEATURE_COLS], test[FEATURE_COLS]

    LAG_FEATS = [c for c in FEATURE_COLS if any(x in c for x in
                 ['lag', 'rm', 'rs', 'rx', 'STREAK', 'WET_DAYS',
                  'CUM', 'LASTYR', 'CLIM'])]
    if not LAG_FEATS:
        LAG_FEATS = FEATURE_COLS

    p('Training LightGBM (met only)...')
    lgb_model = LGBMRegressor(objective='regression', num_leaves=20,
        learning_rate=0.09, n_estimators=1000, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1)
    lgb_model.fit(X_tr[MET], y_tr,
                  eval_set=[(X_va[MET], y_va)],
                  callbacks=[early_stopping(50, verbose=False)])

    p('Training XGBoost (lag/rolling)...')
    xgb_model = XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
        tree_method='hist', random_state=42, verbosity=0)
    xgb_model.fit(X_tr[LAG_FEATS], y_tr,
                  eval_set=[(X_va[LAG_FEATS], y_va)], verbose=False)

    p('Training CatBoost (all features)...')
    cat_model = CatBoostRegressor(iterations=5000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, loss_function='RMSE', random_seed=42, verbose=0)
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                  early_stopping_rounds=100, verbose=False)

    p('Fitting Ridge meta-learner (3-model stack)...')
    lgb_ptr = lgb_model.predict(X_tr[MET]);       lgb_pv = lgb_model.predict(X_va[MET]);     lgb_pt = lgb_model.predict(X_te[MET])
    xgb_ptr = xgb_model.predict(X_tr[LAG_FEATS]); xgb_pv = xgb_model.predict(X_va[LAG_FEATS]); xgb_pt = xgb_model.predict(X_te[LAG_FEATS])
    cat_ptr = cat_model.predict(X_tr);            cat_pv = cat_model.predict(X_va);           cat_pt = cat_model.predict(X_te)

    S_tr = np.column_stack([lgb_ptr, xgb_ptr, cat_ptr])
    S_va = np.column_stack([lgb_pv,  xgb_pv,  cat_pv])
    S_te = np.column_stack([lgb_pt,  xgb_pt,  cat_pt])

    gs = GridSearchCV(Ridge(), {'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100]},
                      cv=TimeSeriesSplit(n_splits=min(5, max(2, len(val) // 30))),
                      scoring='neg_root_mean_squared_error')
    gs.fit(S_va, y_va)
    meta = Ridge(alpha=gs.best_params_['alpha'])
    meta.fit(S_va, y_va)

    pred_train = np.clip(np.expm1(meta.predict(S_tr)), 0, None)
    pred_test  = np.clip(np.expm1(meta.predict(S_te)), 0, None)

    p('Computing metrics...')
    base_preds = {
        'LightGBM': np.clip(np.expm1(lgb_pt), 0, None),
        'XGBoost':  np.clip(np.expm1(xgb_pt), 0, None),
        'CatBoost': np.clip(np.expm1(cat_pt), 0, None),
    }
    model_metrics = {nm: all_metrics(y_te_raw, pr) for nm, pr in base_preds.items()}
    model_metrics['STACKED'] = all_metrics(y_te_raw, pred_test)

    meta_weights = {
        'LightGBM': float(meta.coef_[0]),
        'XGBoost':  float(meta.coef_[1]),
        'CatBoost': float(meta.coef_[2]),
        '_intercept':  float(meta.intercept_),
        '_best_alpha': float(gs.best_params_['alpha']),
    }

    p('Generating plots...')
    plots = {
        'eda': plot_eda(df_full.dropna(subset=[TARGET]), freq='daily'),
        'feature_engineering': plot_feature_engineering_bar(FEATURE_COLS),
        'model_comparison':    plot_model_comparison(model_metrics),
        'residuals':  plot_residuals(test['DATE'].values, y_te_raw - pred_test,
                                      'Stacked Ensemble Residuals'),
        'rmse_by_month':   plot_rmse_by_month(test['MONTH'].values, y_te_raw, pred_test),
        'full_timeseries': plot_full_timeseries(
            train['DATE'].values, y_tr_raw, pred_train,
            test['DATE'].values,  y_te_raw, pred_test),
    }

    model_plots = {}
    for name, pr in base_preds.items():
        model_plots[name] = {
            'timeseries': plot_ts(test['DATE'].values, y_te_raw, pr, f'{name} — Test Set'),
            'scatter':    plot_scatter(y_te_raw, pr, name),
        }
    model_plots['STACKED'] = {
        'timeseries': plot_ts(test['DATE'].values, y_te_raw, pred_test,
                               'Stacked Ensemble — Test Set',
                               color_pred='#f59e0b'),
        'scatter':    plot_scatter(y_te_raw, pred_test,
                                    'Stacked Ensemble', color='#f59e0b'),
    }
    plots['models'] = model_plots

    # SHAP on a sample for speed
    sample_n  = min(1000, len(X_te))
    X_te_s    = X_te.sample(n=sample_n, random_state=42)
    X_te_met  = X_te_s[MET]
    X_te_lag  = X_te_s[LAG_FEATS]

    p('Computing SHAP (CatBoost main)...')
    b1, b2 = plot_shap(cat_model, X_te_s, FEATURE_COLS, 'CatBoost — Main Model')
    plots['shap_bar'] = b1; plots['shap_beeswarm'] = b2

    p('Computing SHAP (LightGBM)...')
    b1l, b2l = plot_shap(lgb_model, X_te_met, MET, 'LightGBM (met only)')
    plots['shap_lgb_bar'] = b1l; plots['shap_lgb_beeswarm'] = b2l

    p('Computing SHAP (XGBoost)...')
    b1x, b2x = plot_shap(xgb_model, X_te_lag, LAG_FEATS, 'XGBoost (lag/rolling)')
    plots['shap_xgb_bar'] = b1x; plots['shap_xgb_beeswarm'] = b2x

    return {
        'ok': True,
        'frequency':    'daily',
        'n_samples':    int(n),
        'n_features':   int(len(FEATURE_COLS)),
        'train_period': [str(train['DATE'].min().date()), str(train['DATE'].max().date())],
        'val_period':   [str(val['DATE'].min().date()),   str(val['DATE'].max().date())],
        'test_period':  [str(test['DATE'].min().date()),  str(test['DATE'].max().date())],
        'split_sizes':  {'train': len(train), 'val': len(val), 'test': len(test)},
        'metrics':      model_metrics,
        'meta_weights': meta_weights,
        'feature_cols': FEATURE_COLS,
        'plots':        plots,
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'No file uploaded.'}), 400
    f = request.files['file']
    raw = f.read()
    if not raw:
        return jsonify({'ok': False, 'error': 'Empty file.'}), 400

    forced = request.form.get('frequency', 'auto').strip().lower()

    try:
        detected, raw_df = detect_format_and_load(raw)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400

    freq = detected if forced in ('auto', '') else forced

    try:
        if freq == 'monthly':
            df_full = reshape_monthly(raw_df) if 'PARAMETER' in raw_df.columns else raw_df.copy()
            if 'DATE' not in df_full.columns and {'YEAR', 'MONTH'}.issubset(df_full.columns):
                df_full['DATE'] = pd.to_datetime(
                    df_full['YEAR'].astype(str) + '-' +
                    df_full['MONTH'].astype(str) + '-01')
            result = run_monthly_pipeline(df_full)
            # Wrap in the new multi-result shape for consistency,
            # but also leave the top-level fields populated for backward compat
            result['results'] = [result.copy()]
            return jsonify(result)

        elif freq == 'daily':
            # Run DAILY pipeline on the uploaded data
            daily_result = run_daily_pipeline(raw_df.copy())
            daily_result['label'] = 'Daily predictions'

            # Also aggregate → MONTHLY and run the monthly pipeline
            try:
                monthly_df = aggregate_daily_to_monthly(raw_df)
                monthly_result = run_monthly_pipeline(monthly_df)
                monthly_result['label'] = 'Monthly predictions (aggregated from daily)'
                monthly_result['aggregated_from_daily'] = True
                results = [daily_result, monthly_result]
            except Exception as ee:
                # If aggregation/monthly run fails, still return the daily result
                # rather than blowing up the whole request.
                results = [daily_result]
                daily_result['monthly_aggregation_error'] = str(ee)

            # Return a multi-result envelope. Put the daily result at the top
            # level so old client code still sees the daily metrics/plots.
            envelope = dict(daily_result)
            envelope['results'] = results
            return jsonify(envelope)

        else:
            return jsonify({'ok': False, 'error': f'Unsupported frequency: {freq}'}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'ok': False, 'error': str(e), 'traceback': tb}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'ok': True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
