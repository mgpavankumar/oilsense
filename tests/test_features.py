import pytest
from datetime import date,timedelta
from oilsense.features import spark_session, feature_frame, clean_prices

@pytest.fixture(scope='module')
def spark():
    session=spark_session();yield session;session.stop()

def records():return [(date(2020,1,1)+timedelta(days=i),float(40+i)) for i in range(50)]

def test_future_prices_cannot_change_past_features(spark):
    data=records();changed=[(d,p if i<35 else 999.) for i,(d,p) in enumerate(data)]
    a=feature_frame(spark.createDataFrame(data,['date','price'])).filter("date = '2020-01-30'").first()
    b=feature_frame(spark.createDataFrame(changed,['date','price'])).filter("date = '2020-01-30'").first()
    assert a.ma_20==b.ma_20==59.5
    assert a.lag_1==68 and a.next_price==70
    assert a.target_date>a.date

def test_requires_twenty_observations(spark):
    result=feature_frame(spark.createDataFrame(records(),['date','price']))
    assert result.count()==30
    assert result.agg({'date':'min'}).first()[0]==date(2020,1,20)

def test_cleaning_rejects_conflicting_duplicates(spark,tmp_path):
    f=tmp_path/'data.csv';f.write_text('Date,Price\n2020-01-01,50\n2020-01-01,51\n')
    with pytest.raises(ValueError,match='Conflicting'):clean_prices(spark,f)

def test_invalid_prices_are_removed(spark,tmp_path):
    f=tmp_path/'data.csv';f.write_text('Date,Price\ninvalid,50\n2020-01-01,-2\n2020-01-02,NaN\n2020-01-03,60\n2020-01-03,60\n')
    assert clean_prices(spark,f).count()==1
