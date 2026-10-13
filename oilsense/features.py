import os
from pyspark.sql import SparkSession, Window, functions as F

FEATURES = ['price', 'lag_1', 'lag_5', 'ma_5', 'ma_20', 'volatility_20', 'return_1', 'weekday']

def spark_session():
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    spark = (SparkSession.builder.master(os.getenv('SPARK_MASTER', 'local[2]'))
             .appName('OilSense').config('spark.sql.shuffle.partitions', '2')
             .config('spark.ui.enabled', 'false').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    return spark

def clean_prices(spark, path):
    raw = spark.read.option('header', True).csv(str(path))
    data = raw.select(F.to_date('Date').alias('date'), F.col('Price').cast('double').alias('price'))
    # Conflicting duplicates are rejected rather than choosing an arbitrary observation.
    data = data.filter(F.col('date').isNotNull() & F.col('price').isNotNull() & ~F.isnan('price') & (F.col('price') > 0) & (F.col('price') < 10000))
    if data.groupBy('date').agg(F.countDistinct('price').alias('n')).filter('n > 1').count():
        raise ValueError('Conflicting prices for the same date')
    return data.dropDuplicates(['date']).orderBy('date')

def feature_frame(data, labels=True):
    w = Window.orderBy('date')
    result = (data.withColumn('lag_1', F.lag('price', 1).over(w))
              .withColumn('lag_5', F.lag('price', 5).over(w))
              .withColumn('ma_5', F.avg('price').over(w.rowsBetween(-4, 0)))
              .withColumn('ma_20', F.avg('price').over(w.rowsBetween(-19, 0)))
              .withColumn('volatility_20', F.stddev_samp('price').over(w.rowsBetween(-19, 0)))
              .withColumn('return_1', F.col('price') / F.col('lag_1') - 1)
              .withColumn('weekday', F.dayofweek('date').cast('double'))
              .withColumn('observations', F.count('price').over(w.rowsBetween(-19, 0))))
    if labels:
        result = (result.withColumn('target_date', F.lead('date', 1).over(w))
                  .withColumn('next_price', F.lead('price', 1).over(w))
                  .withColumn('label', (F.col('next_price') > F.col('price')).cast('double')))
    return result.filter('observations = 20').dropna(subset=FEATURES + (['next_price'] if labels else []))
