import argparse, json, hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from .features import FEATURES, spark_session, clean_prices, feature_frame

ROOT = Path(__file__).resolve().parents[1]

def regression_metrics(frame, prediction='prediction'):
    return {m: RegressionEvaluator(labelCol='next_price', predictionCol=prediction, metricName=m).evaluate(frame) for m in ['mae', 'rmse', 'r2']}

def classification_metrics(frame):
    result = {'roc_auc': BinaryClassificationEvaluator(labelCol='label').evaluate(frame)}
    for name, metric in [('precision', 'precisionByLabel'), ('recall', 'recallByLabel'), ('f1', 'fMeasureByLabel')]:
        result[name] = MulticlassClassificationEvaluator(labelCol='label', metricName=metric, metricLabel=1.0).evaluate(frame)
    result['confusion_matrix'] = [r.asDict() for r in frame.groupBy('label', 'prediction').count().orderBy('label', 'prediction').collect()]
    return result

def pipeline(estimator):
    return Pipeline(stages=[VectorAssembler(inputCols=FEATURES, outputCol='raw_features'),
        StandardScaler(inputCol='raw_features', outputCol='features', withMean=True, withStd=True), estimator])

def train(data_path=ROOT/'data/brent_daily.csv', output=ROOT/'artifacts'):
    spark = spark_session()
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    try:
        clean = clean_prices(spark, data_path).cache()
        frame = feature_frame(clean).cache()
        dates = [r.date for r in frame.select('date').orderBy('date').collect()]
        if len(dates) < 200: raise ValueError('At least 200 usable rows are required')
        validation_start, test_start = dates[int(len(dates)*.70)], dates[int(len(dates)*.85)]
        # Purge boundary targets: no training label can reach validation or test dates.
        training = frame.filter(F.col('target_date') < F.lit(validation_start)).cache()
        validation = frame.filter((F.col('date') >= F.lit(validation_start)) & (F.col('target_date') < F.lit(test_start))).cache()
        test = frame.filter(F.col('date') >= F.lit(test_start)).cache()
        fitting = frame.filter(F.col('target_date') < F.lit(test_start)).cache()
        trials=[]; best_reg=None; best_cls=None
        estimators = [LinearRegression(labelCol='next_price', maxIter=50), RandomForestRegressor(labelCol='next_price', seed=42, numTrees=30), LogisticRegression(labelCol='label', maxIter=60)]
        for estimator in estimators:
            is_classifier=isinstance(estimator, LogisticRegression)
            grid = (ParamGridBuilder().addGrid(estimator.maxDepth,[3,5]).build() if isinstance(estimator,RandomForestRegressor)
                    else ParamGridBuilder().addGrid(estimator.regParam,[.01,.1]).build())
            for params in grid:
                model=pipeline(estimator).fit(training,params)
                predictions=model.transform(validation)
                score=(BinaryClassificationEvaluator().evaluate(predictions) if is_classifier else RegressionEvaluator(labelCol='next_price',metricName='rmse').evaluate(predictions))
                trial={'algorithm':estimator.__class__.__name__,'params':{p.name:v for p,v in params.items()},'validation_score':score,'selection_metric':'roc_auc' if is_classifier else 'rmse'}
                trials.append(trial)
                best=best_cls if is_classifier else best_reg
                if best is None or (score>best[0] if is_classifier else score<best[0]):
                    selected=(score,estimator.copy(params),trial)
                    if is_classifier:best_cls=selected
                    else:best_reg=selected
        reg=pipeline(best_reg[1]).fit(fitting);cls=pipeline(best_cls[1]).fit(fitting)
        for name,model in [('regression',reg),('classification',cls)]:
            model.write().overwrite().save(str(output/name))
        reg_result=reg.transform(test).select('date','target_date','price','next_price','prediction')
        cls_result=cls.transform(test)
        baseline=regression_metrics(test,'price');metrics=regression_metrics(reg_result)
        report={'created_at':datetime.now(timezone.utc).isoformat(),'series':'Brent Europe spot','units':'USD/barrel','data_start':str(clean.agg(F.min('date')).first()[0]),'data_end':str(clean.agg(F.max('date')).first()[0]),'data_rows':clean.count(),'statistics':clean.agg(F.min('price').alias('min'),F.max('price').alias('max'),F.avg('price').alias('mean'),F.stddev_samp('price').alias('stddev')).first().asDict(),'trained_through':str(fitting.agg(F.max('target_date')).first()[0]),'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),'features':FEATURES,'splits':{'validation_start':str(validation_start),'test_start':str(test_start),'train_rows':training.count(),'validation_rows':validation.count(),'test_rows':test.count()},'regression':{'algorithm':best_reg[2]['algorithm'],'params':best_reg[2]['params'],'metrics':metrics,'baseline':baseline,'beats_baseline_rmse':metrics['rmse']<baseline['rmse']},'classification':{'algorithm':best_cls[2]['algorithm'],'params':best_cls[2]['params'],'metrics':classification_metrics(cls_result)},'trials':trials,'limitations':['Historical revised data, not a point-in-time release archive.','Small daily dataset demonstrates Spark workflows, not big-data capacity.','No live quote feed; no guarantee of investment performance.','Classification probabilities are uncalibrated model scores.']}
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        predictions=[{**r.asDict(),'date':str(r.date),'target_date':str(r.target_date)} for r in reg_result.orderBy('date').collect()]
        probabilities=cls_result.select('date','prediction',vector_to_array('probability')[1].alias('probability_up')).collect()
        direction={str(r.date):{'direction':'up' if r.prediction else 'down_or_flat','probability_up':r.probability_up} for r in probabilities}
        predictions=[{**r,**direction[r['date']]} for r in predictions]
        (output/'backtest.json').write_text(json.dumps(predictions,allow_nan=False))
        shutil.copyfile(data_path,output/'training_data.csv')
        # Only test observations become selectable historical predictions. They were unseen by final models.
        Path(ROOT/'web/report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        Path(ROOT/'web/backtest.json').write_text(json.dumps(predictions,allow_nan=False))
        history=[{'date':str(r.date),'price':r.price} for r in clean.collect()]
        Path(ROOT/'web/history.json').write_text(json.dumps(history,allow_nan=False))
        # Verify persisted models reload and reproduce one prediction.
        check=PipelineModel.load(str(output/'regression')).transform(test.orderBy('date').limit(1)).first().prediction
        expected=reg_result.orderBy('date').first().prediction
        if abs(check-expected)>1e-8:raise RuntimeError('Persistence check failed')
        print(json.dumps({'regression':report['regression'],'classification':report['classification'],'splits':report['splits']},indent=2))
        return report
    finally:spark.stop()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'data/brent_daily.csv');parser.add_argument('--output',type=Path,default=ROOT/'artifacts');args=parser.parse_args();train(args.data,args.output)
