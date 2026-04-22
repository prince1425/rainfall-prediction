# PRECIP.LAB — Stacked Ensemble Rainfall Predictor

An interactive web app that runs your **dual-transform stacked ensemble rainfall-prediction pipeline** end-to-end on any user-uploaded NASA/POWER MERRA-2 CSV. Replicates every step from the original Shillong-monthly and Guwahati-daily notebooks: feature engineering, chronological 70/15/15 split, base-model training, Ridge meta-learner, full SHAP interpretability, and every paper-ready plot.

---

## What it does

Drop a CSV → the backend auto-detects monthly vs daily format → runs the full pipeline → the frontend shows:

| Section | Content |
|---|---|
| **§02 Run Summary** | Frequency, sample/feature counts, train/val/test split sizes, test period, headline Stacked R² & KGE |
| **§03 EDA** | 6-panel overview: time series + 12-mo/30-day MA, log-scale distribution, monthly climatology, T2M vs rain, QV2M vs rain, correlation ranking |
| **§04 Feature Engineering** | Family-count breakdown + filterable chip browser showing every engineered feature |
| **§05 Metrics** | MAE · MSE · RMSE · R² · KGE for every base model + STACKED, with mini-bars and the stacked row highlighted |
| **§06 Actual vs Predicted** | Unrolled vertical view showcasing the time-series overlay and scatter plot for every single base learner and the STACKED ensemble |
| **§07 Diagnostics** | Residual time-series / histogram / Q-Q plot, RMSE-by-month, full train+test timeline |
| **§08 SHAP** | Bar + beeswarm for CatBoost (main), LightGBM (met-only), XGBoost (lag/rolling) |
| **§09 Ridge Meta-Learner** | Per-base-model Ridge weights, role tags, best α, intercept |

### 💫 Modern, Interactive UI
The frontend is highly dynamic, featuring an animated atmospheric background, glassmorphism elements, smooth scroll navigation with a floating "back to top" button, hover effects on data cards, and `IntersectionObserver`-powered scroll animations that gently reveal plots and tables as you scroll.

---

## Requirements

- Python 3.10+
- ~1 GB RAM (daily datasets can use more)
- No GPU needed — CPU training is fast enough (monthly: ~25 s, daily: ~2–4 min)

---

## Install

```bash
pip install flask flask-cors xgboost lightgbm catboost shap scikit-learn pandas numpy matplotlib
```

Or save the following as `requirements.txt` and run `pip install -r requirements.txt`:

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

The Flask dev server is single-threaded — use **gunicorn** for real use so long-running daily-pipeline requests don't block health checks:

```bash
pip install gunicorn
cd precip-lab/backend
gunicorn -w 2 -t 600 -b 0.0.0.0:5000 app:app
```

`-t 600` gives each worker 10 minutes per request — enough for a full daily run on 40+ years of data.

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

---

## Project layout

```
precip-lab/
├── backend/
│   └── app.py            # Flask server + full ML pipeline (~1,070 lines)
├── frontend/
│   └── index.html        # Single-page dashboard (~1,860 lines, self-contained)
└── README.md
```

The frontend is a single static HTML file with inline CSS and JS — no build step, no bundler, no npm. Flask serves it at `/` and the JSON API at `/api/analyze`.

---

## API

### `POST /api/analyze`
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file` — the CSV (required)
  - `frequency` — `auto` (default), `monthly`, or `daily`
- **Returns:** JSON with metrics, meta-weights, feature list, and base64-PNG plots.

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

Stacking lifts R² from 0.62 (best individual) to 0.81 — a 31 % relative improvement. Total end-to-end time ≈ 25 s on a modest CPU.

---

## Customizing

### Change the model hyperparameters
Edit the `LGBMRegressor(...)`, `XGBRegressor(...)`, `CatBoostRegressor(...)` calls inside `run_monthly_pipeline` / `run_daily_pipeline` in `backend/app.py`. The defaults mirror the tuned values from your original notebooks.

### Change the split ratio
Look for `i1 = int(n * 0.70); i2 = int(n * 0.85)` in both pipeline functions.

### Add another plot
1. Write a plotting function that takes numpy arrays / DataFrames and returns `fig_to_b64(fig)`.
2. Add the output to the `plots = {...}` dict inside the pipeline.
3. Add an `<img>` tag and a `setImg('plot-XXX', d.plots.XXX)` call in `frontend/index.html`.

### Make a different base model dominant
Change which model is passed to `plot_shap(...)` for the "main model" SHAP card in each pipeline.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not detect NASA/POWER CSV format` | Your file doesn't have either a `PARAMETER,YEAR,JAN,...` or `YEAR,DOY,...` header. Save the file directly from the NASA/POWER data portal, not a pre-processed version. |
| `Missing required columns: [...]` | The CSV is missing one of the 13 required parameters. Check the `PARAMETER` column (monthly) or column headers (daily). |
| `Not enough clean samples` | Your file is too short — monthly needs ≥ 50 rows after dropping the first 24 months of lags, daily needs ≥ 500 rows. |
| Request times out after ~30 s for daily files | You're using the Flask dev server (`python app.py`). Switch to gunicorn with `-t 600`. |
| SHAP plot looks cut off / labels overlap | Max 15 features shown by default; change `max_display=15` in `plot_shap(...)`. |

---

## Credits

Built around the **dual-transform stacked ensemble** methodology from the Shillong-monthly and Guwahati-daily rainfall notebooks. Uses LightGBM, XGBoost, CatBoost, SHAP, scikit-learn, Flask, and matplotlib. Climate data from NASA/POWER (MERRA-2 reanalysis).
