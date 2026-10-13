import json, os, threading, uuid, logging, hashlib
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from .features import spark_session, clean_prices, feature_frame

ROOT=Path(__file__).resolve().parents[1]
class ForecastRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    as_of: date
class ForecastResponse(BaseModel):
    as_of: date
    forecast_for: date | None
    horizon: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    direction: str
    probability_up: float
    source: str
    model_data_cutoff: str
    baseline_price: float

class Engine:
    def __init__(self):
        self.lock=threading.Lock();self.spark=None;self.reg=None;self.cls=None
        self.report=json.loads((ROOT/'web/report.json').read_text())
        self.rows=json.loads((ROOT/'web/backtest.json').read_text())
        self.history=json.loads((ROOT/'web/history.json').read_text())
        artifact=Path(os.getenv('OILSENSE_ARTIFACT_DIR',ROOT/'artifacts'))
        if (artifact/'regression').exists() and (artifact/'classification').exists():
            # Use the artifact's report: cached demo and loaded models may have different cutoffs.
            self.report=json.loads((artifact/'report.json').read_text())
            self.rows=json.loads((artifact/'backtest.json').read_text())
            self.spark=spark_session();self.reg=PipelineModel.load(str(artifact/'regression'));self.cls=PipelineModel.load(str(artifact/'classification'))
            data_path=artifact/'training_data.csv'
            if hashlib.sha256(data_path.read_bytes()).hexdigest()!=self.report['data_sha256']:raise RuntimeError('Model dataset checksum mismatch')
            self.data=clean_prices(self.spark,data_path).cache()
            self.history=[{'date':str(r.date),'price':r.price} for r in self.data.collect()]
    def close(self):
        if self.spark:self.spark.stop()
    def forecast(self,as_of):
        key=as_of.isoformat();row=next((r for r in self.rows if r['date']==key),None)
        if row:
            price=row['price'];pred=row['prediction'];prob=row['probability_up'];direction=row['direction'];target=row['target_date'];source='held_out_backtest'
        elif key==self.report['data_end'] and self.reg:
            with self.lock:
                frame=feature_frame(self.data.filter(self.data.date<=key),False).orderBy('date',ascending=False).limit(1)
                if not frame.count():raise HTTPException(422,'Insufficient observation history')
                price=frame.first().price;pred=self.reg.transform(frame).first().prediction
                result=self.cls.transform(frame).select('prediction',vector_to_array('probability')[1].alias('p')).first()
                prob=result.p;direction='up' if result.prediction else 'down_or_flat';target=None;source='saved_spark_pipeline'
        else:raise HTTPException(422,'Choose a held-out test date, or the dataset cutoff after training models. Weekends and missing observations are unavailable.')
        return {'as_of':key,'forecast_for':target,'horizon':'next available price observation','current_price':price,'predicted_price':pred,'predicted_change_pct':(pred/price-1)*100,'direction':direction,'probability_up':prob,'source':source,'model_data_cutoff':self.report['trained_through'],'baseline_price':price}

def create_app(engine_factory=Engine):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine=engine_factory();yield;app.state.engine.close()
    app=FastAPI(title='OilSense Forecast API',version='1.0.0',description='Brent crude oil historical forecasting. USD per barrel. Probabilities are uncalibrated. Historical observations are not live quotes.',lifespan=lifespan)
    @app.middleware('http')
    async def safeguards(request:Request,call_next):
        request_id=str(uuid.uuid4())
        if request.method=='POST':
            length=request.headers.get('content-length')
            if length and (not length.isdigit() or int(length)>16384):return JSONResponse({'detail':'Request body limit is 16 KiB'},status_code=413)
            chunks=[];total=0
            async for chunk in request.stream():
                total+=len(chunk)
                if total>16384:return JSONResponse({'detail':'Request body limit is 16 KiB'},status_code=413)
                chunks.append(chunk)
            request._body=b''.join(chunks)
        response=await call_next(request)
        response.headers.update({'X-Request-ID':request_id,'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY','Referrer-Policy':'no-referrer','Cache-Control':'no-store'})
        return response
    @app.exception_handler(Exception)
    async def unexpected(request,exc):
        logging.getLogger('oilsense').exception('Request failed')
        return JSONResponse({'detail':'Internal server error'},status_code=500)
    @app.get('/health')
    def health():return {'status':'ok','mode':'spark' if app.state.engine.reg else 'recorded_backtest'}
    @app.get('/api/v1/metadata')
    def metadata():
        e=app.state.engine
        return {'report':e.report,'available_dates':[r['date'] for r in e.rows],'latest_forecast_available':bool(e.reg),'mode':'spark' if e.reg else 'recorded_backtest'}
    @app.get('/api/v1/history')
    def history(limit:int=Query(500,ge=20,le=5000)):return app.state.engine.history[-limit:]
    @app.get('/api/v1/backtest')
    def backtest():return app.state.engine.rows
    @app.post('/api/v1/forecast',response_model=ForecastResponse)
    def forecast(payload:ForecastRequest):return app.state.engine.forecast(payload.as_of)
    @app.get('/',include_in_schema=False)
    def home():return FileResponse(ROOT/'web/index.html')
    app.mount('/',StaticFiles(directory=ROOT/'web'),name='static')
    return app
app=create_app()
