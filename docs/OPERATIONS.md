# Local runtime and deployment

Recommended: Python 3.11, Java 17, 2 CPU cores, roughly 2 GB available RAM for the small local Spark demo. Run one API worker. Startup loads saved model artifacts if present; missing models intentionally select recorded-backtest mode.

Train in a separate offline release step. Keep artifacts/report.json, backtest.json, training_data.csv and both pipeline directories from the same run. The CSV hash is verified against the report. The training script checks reloaded predictions. Stop the server before replacing artifacts; use a versioned immutable artifact directory in real deployment. For a larger Spark dataset, replace global date ordering with an appropriate partitioned time-series design.

API binds localhost by default. Public hosting needs HTTPS, authentication/rate limiting at a gateway, explicit resource/time limits, logging without sensitive payloads, health monitoring and an approved data refresh strategy. The application has bounded JSON input but no multi-tenant admission control. Do not expose expensive Spark inference to unrestricted public traffic. No production host or paid service is configured.

The web/ directory can be served independently as an offline static backtest explorer, e.g. `python -m http.server 8080 --directory web`. It is visibly marked static recorded mode and cannot compute new predictions. This is useful for the classroom when Java is unavailable.
