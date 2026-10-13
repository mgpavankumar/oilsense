# Case study: OilSense

## Problem statement

Crude oil price volatility makes procurement and budget planning uncertain. OilSense investigates whether historical daily Brent spot prices can support next-observation price and direction prediction, using a reproducible distributed-machine-learning workflow. The product makes predictions and their measured reliability understandable through an interactive dashboard and an API.

## Objectives

1. Clean and summarize daily Brent observations with Spark DataFrames.
2. Derive temporally valid features from past prices and dates.
3. Train regression and classification models through Spark ML pipelines.
4. Select hyperparameters using earlier training and later validation periods.
5. Evaluate untouched test dates against a persistence baseline.
6. Save and reload fitted pipelines and expose predictions through an API.

## Data and labels

Brent spot price, USD/barrel; 2,793 valid observations from 2015-01-02 to 2025-12-31. Regression target: the next available observed price. Classification label: 1 if the next price is strictly greater, otherwise 0 (down or flat). Non-trading days are not fabricated.

## Features and processing

Current price, price lags 1 and 5, rolling means 5 and 20, rolling 20-price standard deviation, one-observation return, weekday. Rolling statistics include the current known observation. Twenty valid observations are required. Invalid dates/prices are removed; identical duplicates collapse and conflicting duplicate prices are rejected.

VectorAssembler creates a vector; StandardScaler learns scaling statistics; a regression/classification estimator fits the model. A fitted pipeline is a Transformer. Labels and future timestamps are never input features.

## Experimental design

Approximately 70/15/15 chronological train/validation/test. Boundary rows whose labels cross into a later partition are excluded from fitting. ParamGridBuilder supplies two candidates each for Linear Regression regularization, Random Forest depth, and Logistic Regression regularization. Candidates are fit only on training data and scored on validation data; selection uses regression RMSE or classification ROC-AUC. Selected pipelines are refit on the pre-test data, and evaluated once on test dates.

Random CrossValidator/TrainValidationSplit is intentionally not used: random folds do not model the direction of time. The explicit chronological parameter loop demonstrates model tuning while preserving time order. Future improvements include multiple walk-forward folds, feature ablation, additional economic signals, and calibration.

## Findings

The regression model is slightly worse than persistence on test RMSE. Direction ROC-AUC is below 0.5, indicating no useful ranking performance in the tested period. These are valid negative findings; the product must not present them as reliable trading intelligence. Exact metrics and trial scores are in web/report.json.

## Product flow

Historical CSV → Spark cleaning/feature windows → chronological tuning → held-out evaluation → saved pipelines → FastAPI → dashboard.

The static recorded mode displays actual model output from the held-out set. Saved-model mode also supports inference at the dataset cutoff. No new model is trained by a web request.
