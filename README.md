# PRECIP.LAB — Stacked Ensemble Rainfall Predictor

An interactive web app that runs a **dual-transform stacked ensemble rainfall-prediction pipeline** end-to-end on any user-uploaded NASA/POWER MERRA-2 CSV. Replicates every step from the original Shillong-monthly and Guwahati-daily notebooks: feature engineering, chronological 70/15/15 split, base-model training, Ridge meta-learner, full SHAP interpretability, and every paper-ready plot.

---

## What it does

Drop a CSV → the backend auto-detects monthly vs daily format → runs the full pipeline → the frontend shows everything below.

### 🔁 Daily uploads get **both** daily *and* monthly predictions

When you upload a **daily** CSV, PRECIP.LAB automatically:

1. Runs the **daily pipeline** (3-model stack, log1p-transform) on your original data.
2. **Aggregates the daily data to monthly** — mean for rates/temperatures/humidity, max for `T2M_MAX`, min for `T2M_MIN`, circular mean for wind directions — then runs the full **monthly pipeline** (6-model dual-transform stack) on top.

The result is **two complete result sets from one upload**, selectable via a top-of-page tab switcher. Every section (EDA, feature engineering, metrics, per-model plots, SHAP, Ridge weights) updates instantly when you switch tabs.

Uploading a **monthly** CSV runs only the monthly pipeline (there's no way to derive daily predictions from monthly data).

### What's on the page

| Section | Content |
|---|---|
| **§ Dataset spec** | Static reference of every required column (target, time columns, temperature, humidity, pressure, wind, soil) with units and descriptions |
| **Result tabs** | *(Daily uploads only)* Switch between daily predictions and monthly-aggregated predictions |
| **§02 Run Summary** | Frequency, sample/feature counts, train/val/test split sizes, test period, headline Stacked R² & KGE |
| **§03 EDA** | 6-panel overview: time series + MA, log-scale distribution, monthly climatology, T2M vs rain, QV2M vs rain, correlation ranking |
| **§04 Feature Engineering** | Family-count breakdown + filterable chip browser showing every engineered feature |
| **§05 Metrics** | MAE · MSE · RMSE · R² · KGE for every base model + STACKED, with mini-bars and the stacked row highlighted |
| **§06 Actual vs Predicted** | Tabbed view — switch between each base learner and the STACKED ensemble to see time-series overlay + scatter plot |
| **§07 Diagnostics** | Residual time-series / histogram / Q-Q plot, RMSE-by-month, full train+test timeline |
| **§08 SHAP** | Bar + beeswarm for CatBoost (main), LightGBM (met-only), XGBoost (lag/rolling) |
| **§09 Ridge Meta-Learner** | Per-base-model Ridge weights, role tags, best α, intercept |

### 💫 Modern, interactive UI
Animated atmospheric background, glassmorphism elements, smooth scroll navigation, hover effects on data cards, and `IntersectionObserver`-powered scroll animations that gently reveal plots and tables. Single self-contained HTML file — no build step, no bundler, no npm.

---

## Requirements

- Python 3.10+
- ~1 GB RAM (daily datasets can use more)
- No GPU needed — CPU training is fast enough

**Typical runtime:**
- Monthly CSV → ~25 s
- Daily CSV → ~45–60 s (runs daily *and* monthly pipelines back-to-back)

---

## Install

```bash
pip install -r requirements.txt
```

Or install the packages directly:

```bash
pip install flask flask-cors xgboost lightgbm catboost shap scikit-learn pandas numpy matplotlib
```

Contents of `requirements.txt`:

```
flask>=2.3
flask-cors>=4.0
xgboost>=2.0
lightgbm>=4.0
catboost>=1.2
shap>=0.44
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
```

---

## Run (development)

```bash
# from the project root
cd precip-lab/backend
python app.py
```

Then open **http://localhost:5000** in your browser.

### Run (production / longer jobs)

The Flask dev server is single-threaded — use **gunicorn** so long-running daily-pipeline requests don't block health checks:

```bash
pip install gunicorn
cd precip-lab/backend
gunicorn -w 2 -t 600 -b 0.0.0.0:5000 app:app
```

`-t 600` gives each worker 10 minutes per request — enough for a full daily + monthly run on 40+ years of data. On Windows, use **waitress** instead:

```bash
pip install waitress
waitress-serve --port=5000 app:app
```

---

## Input CSV format

The app accepts the **exact raw CSV format** that NASA/POWER returns.

### Monthly (wide format)
Header block starting with `-BEGIN HEADER-` is optional — the app auto-skips it. The actual data header must look like:

```
PARAMETER,YEAR,JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC,ANN
GWETTOP,1981,0.80,0.79,0.76,...
PRECTOTCORR,1981,...
T2M,1981,...
```

Required `PARAMETER` values (case-sensitive):
`PRECTOTCORR`, `T2M`, `T2MDEW`, `T2M_MAX`, `T2M_MIN`, `QV2M`, `RH2M`, `PS`, `WS2M`, `WS10M`, `WD10M`, `WD2M`, `GWETTOP`

### Daily (flat format)

Header like:

```
YEAR,DOY,T2M,T2MDEW,T2M_MAX,T2M_MIN,QV2M,RH2M,PS,WS2M,WS10M,WD10M,WD2M,GWETTOP,PRECTOTCORR
1981,1,14.2,8.1,...
```

Minimum record lengths:
- Monthly: ≥ 50 usable rows (the 24-month lag drops the first 2 years)
- Daily: ≥ 500 usable rows (the 365-day lag drops the first year)

The app's landing page has a visual spec card listing every required column — scroll down on `/` to see it.

---

## Project layout

```
precip-lab/
├── backend/
│   └── app.py            # Flask server + full ML pipeline (~1,070 lines)
├── frontend/
│   └── index.html        # Single-page dashboard (~1,770 lines, self-contained)
├── requirements.txt
├── .gitignore
└── README.md
```

Flask serves the frontend at `/` and exposes the JSON API at `/api/analyze`.

---

## API

### `POST /api/analyze`
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file` — the CSV (required)
  - `frequency` — `auto` (default), `monthly`, or `daily`
- **Returns:**
  ```
  {
    "ok": true,
    "frequency": "daily" | "monthly",
    "results": [
      { "frequency": "daily",   "label": "Daily predictions",   "metrics": {...}, "plots": {...}, ... },
      { "frequency": "monthly", "label": "Monthly predictions (aggregated from daily)",
        "aggregated_from_daily": true, "metrics": {...}, "plots": {...}, ... }
    ],
    "metrics": {...},      // mirrors results[0] for backward compatibility
    "plots":   {...},
    ...
  }
  ```
  - For a **daily** upload: `results` has **two** entries (daily + monthly-aggregated).
  - For a **monthly** upload: `results` has **one** entry.

### `GET /api/health`
Returns `{"ok": true}` — useful for container health-checks.

---

## Verified run (Shillong monthly, 1981–2025)

```
n_samples  = 516 (after feature engineering + dropna)
n_features = 128

Test metrics:
  LGB (sqrt)      RMSE=7.7383  R²=0.5895  KGE=0.4264
  XGB (sqrt)      RMSE=9.4344  R²=0.3899  KGE=0.2817
  CAT (sqrt)      RMSE=7.4784  R²=0.6166  KGE=0.4246
  LGB (log1p)     RMSE=7.9883  R²=0.5626  KGE=0.3972
  XGB (log1p)     RMSE=10.6598 R²=0.2211  KGE=0.1536
  CAT (log1p)     RMSE=8.3185  R²=0.5257  KGE=0.3481
  STACKED         RMSE=5.2069  R²=0.8142  KGE=0.6703    ← the big win
```

Stacking lifts R² from 0.62 (best individual) to 0.81 — a 31% relative improvement. Total end-to-end time ≈ 25 s on a modest CPU.

---

## How daily → monthly aggregation works

When you upload a daily CSV, the rollup to monthly uses the same rules NASA/POWER applies to produce its own monthly product:

| Variable | Aggregation |
|---|---|
| `PRECTOTCORR`, `T2M`, `T2MDEW`, `QV2M`, `RH2M`, `PS`, `WS2M`, `WS10M`, `GWETTOP` | Mean |
| `T2M_MAX` | Max |
| `T2M_MIN` | Min |
| `WD10M`, `WD2M` (wind direction) | Circular mean via (u, v) components of a unit vector — avoids the wrap-around bug a naive average produces |

The aggregated frame is then fed directly into the monthly pipeline, which engineers its own monthly-specific features (24-month lags, yearly climatology, wet/dry streaks at the monthly scale, etc.) and trains the full 6-model dual-transform stack.

---

## Customizing

### Change the model hyperparameters
Edit the `LGBMRegressor(...)`, `XGBRegressor(...)`, `CatBoostRegressor(...)` calls inside `run_monthly_pipeline` / `run_daily_pipeline` in `backend/app.py`. The defaults mirror the tuned values from the original notebooks.

### Change the split ratio
Look for `i1 = int(n * 0.70); i2 = int(n * 0.85)` in both pipeline functions.

### Disable the daily-to-monthly aggregation
In `app.py`, inside the `/api/analyze` route, delete the block that runs `aggregate_daily_to_monthly` and `run_monthly_pipeline` after the daily pipeline. The daily upload will then return a single result set.

### Add another plot
1. Write a plotting function that takes numpy arrays / DataFrames and returns `fig_to_b64(fig)`.
2. Add the output to the `plots = {...}` dict inside the pipeline.
3. Add an `<img>` tag and a `setImg('plot-XXX', d.plots.XXX)` call in `frontend/index.html`.

### Make a different base model dominant for SHAP
Change which model is passed to `plot_shap(...)` for the "main model" SHAP card in each pipeline.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not detect NASA/POWER CSV format` | Your file doesn't have either a `PARAMETER,YEAR,JAN,...` or `YEAR,DOY,...` header. Download it directly from the NASA/POWER data portal, not a pre-processed version. |
| `Missing required columns: [...]` | The CSV is missing one of the 13 required parameters. Check the `PARAMETER` column (monthly) or column headers (daily) — the landing page has the full spec. |
| `Not enough clean samples` | Your file is too short — monthly needs ≥ 50 rows after dropping the first 24 months of lags, daily needs ≥ 500 rows. |
| Request times out after ~30 s for daily files | You're using the Flask dev server (`python app.py`). Switch to gunicorn with `-t 600` (or `waitress-serve` on Windows). |
| Only one tab shows up for a daily upload | The monthly aggregation silently failed — open the browser console or check the server log for the `monthly_aggregation_error` field in the response. |
| SHAP plot looks cut off / labels overlap | Max 15 features shown by default; change `max_display=15` in `plot_shap(...)`. |

---

## Credits

Built around the **dual-transform stacked ensemble** methodology from the Shillong-monthly and Guwahati-daily rainfall notebooks. Uses LightGBM, XGBoost, CatBoost, SHAP, scikit-learn, Flask, and matplotlib. Climate data from NASA/POWER (MERRA-2 reanalysis).
