import pytest,json
from fastapi.testclient import TestClient
from oilsense.api import create_app, ROOT

@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setenv('OILSENSE_ARTIFACT_DIR',str(tmp_path))
    with TestClient(create_app()) as c:yield c

def test_health_and_documentation(client):
    assert client.get('/health').json()['mode']=='recorded_backtest'
    schema=client.get('/openapi.json').json()
    assert '/api/v1/forecast' in schema['paths']
    assert client.get('/').status_code==200
    assert client.get('/styles.css').status_code==200

def test_forecast_matches_measured_backtest(client):
    row=json.loads((ROOT/'web/backtest.json').read_text())[0]
    r=client.post('/api/v1/forecast',json={'as_of':row['date']})
    assert r.status_code==200
    assert r.json()['predicted_price']==row['prediction']
    assert r.json()['source']=='held_out_backtest'
    assert 0<=r.json()['probability_up']<=1
    assert r.headers['x-request-id'] and r.headers['x-content-type-options']=='nosniff'

@pytest.mark.parametrize('body',[{'as_of':'bad'},{'as_of':'1990-01-01'},{'as_of':'2025-01-01','future_price':100},{}])
def test_rejects_invalid_or_unavailable_inputs(client,body):assert client.post('/api/v1/forecast',json=body).status_code==422

def test_limits_request_body(client):
    assert client.post('/api/v1/forecast',content='x'*17000).status_code==413
    assert client.post('/api/v1/forecast',content=iter([b'x'*8000,b'x'*9000])).status_code==413

def test_history_pagination_bounds(client):
    assert len(client.get('/api/v1/history?limit=20').json())==20
    assert client.get('/api/v1/history?limit=1000000').status_code==422
