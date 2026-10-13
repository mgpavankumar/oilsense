# Verification - 13 October 2026

- Actual Spark 3.5.7 training completed on the committed Brent CSV with 2,793 observations.
- Six parameter candidates were evaluated chronologically. Final test set: 416 rows.
- Persisted regression pipeline was reloaded and reproduced its prediction within 1e-8.
- All 12 pytest checks passed, covering temporal leakage, observation windows, duplicate conflicts, invalid prices, OpenAPI, valid forecasts, invalid dates, extra fields, body limits, and history bounds.
- Real HTTP saved-pipeline inference at 2025-12-31 returned a valid response with no invented future target date.
- Desktop 1440 px and mobile 390 px dashboard interactions passed without JavaScript errors or horizontal overflow.
- Local runtime: Python 3.12 and Java 17; GitHub CI independently verifies the recommended Python 3.11 runtime.

Measured regression RMSE 1.3597549100278 USD/barrel vs baseline 1.3501579323574315. Classification ROC-AUC 0.48265013417229585. These are weak forecasting results, transparently reported; successful software tests do not demonstrate predictive value or production capacity.
