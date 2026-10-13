# Five-minute classroom demonstration

1. State the procurement-planning problem, the Brent benchmark, and the historical cutoff.
2. Start the server and explain its mode. Show the dashboard's dataset card and held-out metrics.
3. Select two dates in Forecast Lab. Explain predicted versus actual prices, direction score, and the difference between regression and classification.
4. Show the baseline comparison. Explain honestly that the fitted models did not outperform persistence.
5. Open `/docs`, run `/api/v1/metadata`, then send a valid forecast request and an invalid date. Show a 422 validation response.
6. Show `features.py`: Spark DataFrame, Window, and lag/rolling features. Explain no future features.
7. Show `train.py`: Pipeline stages, ParamGridBuilder, chronological selection, evaluators, save/load verification.
8. After training, show `/health` with spark mode and request the dataset cutoff forecast. Explain why the future observation date is null.

Viva: An Estimator learns from data with fit; a Transformer applies transform. StandardScaler is an Estimator and StandardScalerModel is a Transformer. Pipeline holds stages; PipelineModel holds fitted stages. RMSE penalizes large numerical errors; ROC-AUC evaluates ranking across thresholds; precision and recall here refer specifically to the rise class. PySpark ML is the DataFrame-based API in the MLlib family.
