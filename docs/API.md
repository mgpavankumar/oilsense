# OilSense API v1

Interactive specification: `/docs`. Machine-readable OpenAPI: `/openapi.json`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | /health | Process health and serving mode |
| GET | /api/v1/metadata | Dataset, metrics, available test dates, mode |
| GET | /api/v1/history?limit=500 | Last 20–5000 historical observations |
| GET | /api/v1/backtest | All measured held-out predictions |
| POST | /api/v1/forecast | Historical forecast by observation date, or cutoff inference after training |

Request: `{"as_of":"2025-12-30"}`. Extra fields are rejected. Unknown dates, malformed dates, dates before the test period, and missing observation dates return 422. Bodies exceeding 16 KiB return 413, including streamed bodies. Request validation is described in OpenAPI.

Response fields: as_of, forecast_for (null for an unknown future observation date), horizon, current_price, predicted_price, predicted_change_pct, direction (up or down_or_flat), probability_up (uncalibrated), source (held_out_backtest or saved_spark_pipeline), model_data_cutoff, baseline_price. Price units: USD/barrel.

Metadata reports `recorded_backtest` or `spark`. Recorded mode is transparent replay of genuinely computed ML forecasts, not online inference. All historical selectable dates are held out. The final observation can be scored by saved pipelines only. Model/data checksums are verified before live feature computation.

Responses include generated request IDs, no-store caching, and basic security headers. Unexpected errors omit internal exception text. Public authentication and rate controls are deployment responsibilities; keep the default localhost binding for a classroom demonstration.
