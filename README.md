# SSK Vehicle Occupancy Forecasting — PySpark + Kafka

Same pipeline as before (same features, same XGBoost models, same API), with the data engineering
moved to **PySpark / Spark SQL functions** and a **Kafka → Spark Structured Streaming** ingest added.

```text
feature_pipeline.py   Spark feature engineering (window functions) + exact-time inference features
spark_session.py      builds the SparkSession (UTC, local[*])
train.py              Spark builds the dataset -> XGBoost trains -> versioned models/
retrain.py            same as train.py
predict.py            offline inference (CLI)
app.py                FastAPI (same endpoints + /predict/stream)
kafka_producer.py     replays the raw workbook OR runs the 30s synthetic demo stream
kafka_stream.py       Spark Structured Streaming: Kafka -> clean -> parquet (+ optional prediction)
docker-compose.yml    single-node Kafka for local use
data/raw/Vehicle_Data_ DwellTime _Tra.xlsx    <- put your workbook here

---

## 0. Prerequisites

* **Python 3.10 – 3.12**
* **Java 17 or 21** (`java -version`) — Spark needs it. Set `JAVA_HOME` if Spark can't find it.
* **Docker** — only for Kafka (section 4).
* Windows: use **WSL2** (Spark on native Windows needs extra Hadoop setup).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows/WSL: same; plain Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/raw                  # then copy your workbook into data/raw/
```

## 1. Train

```bash
python train.py
```

* Spark builds the supervised dataset → `data/processed/supervised_multihorizon/` (parquet).
* XGBoost (point, P50, P90) is trained and saved to `models/v001/`, with the same promotion rule
  as before (a new version is promoted only if validation MAE **and** MSE both improve).
* The first run takes a bit longer because Spark starts up.

Retrain any time: `python retrain.py`.

## 2. Predict from the command line

```bash
python predict.py                                            # latest timestamp in the workbook
python predict.py --prediction-time "2026-09-17 10:17:00"    # exact time
python predict.py --raw other_file.xlsx                      # other workbook / .csv
```

## 3. Run the API

```bash
uvicorn app:app --port 8000
```

Wait for `Application startup complete` (Spark is started once at boot). Do **not** use
`--workers N`: every worker would start its own Spark.

Swagger UI: http://127.0.0.1:8000/docs

```bash
# health
curl http://127.0.0.1:8000/health

# predict from the default workbook in data/raw/
curl -X POST "http://127.0.0.1:8000/predict?prediction_time=2026-09-17%2010:17:00"

# peak forecast
curl -X POST "http://127.0.0.1:8000/predict/peak?prediction_time=2026-09-17%2010:17:00"

# upload your own workbook
curl -X POST -F "file=@data/raw/Vehicle_Data_ DwellTime _Tra.xlsx" \
     "http://127.0.0.1:8000/predict?prediction_time=2026-09-17%2010:17:00"

# predict from the data the Kafka stream has landed (section 4)
curl -X POST "http://127.0.0.1:8000/predict/stream?prediction_time=2026-09-17%2010:17:00"
curl -X POST "http://127.0.0.1:8000/predict/stream/peak?prediction_time=2026-09-17%2010:17:00"
```

Postman: `POST /predict`, Body → form-data, `file` (File) and optional `prediction_time` (Text).
Omit `prediction_time` to use the latest observed timestamp.

## 4. Kafka streaming — step by step

You need **three terminals**, all in the project folder with the venv active.

**Terminal 1 — start Kafka**

```bash
docker compose up -d
docker compose ps            # state should be "running"
```

**Terminal 2 — start the Spark streaming job** (leave it running)

```bash
python kafka_stream.py
```

The first start downloads the Spark–Kafka connector from Maven (needs internet, one time; cached
afterwards). It then waits for messages and prints one block per micro-batch (every 10 s):

```text
[batch 0] stored 1180 visits -> data/stream/visits
+-------------------+--------+-----+
|bucket             |arrivals|exits|
...
```

**Terminal 3 — publish data** (simulates the gate system)

```bash
python kafka_producer.py --rate 200
```

Useful options: `--rate 0` (as fast as possible), `--limit 500`, `--from-time "2026-09-16 00:00:00"`,
`--topic`, `--bootstrap`. Messages are JSON, one per completed visit, sent in Exit-Time order:

```json
{"entry_time": "2026-09-17 09:12:40", "exit_time": "2026-09-17 11:03:05", "risk_category": "LOW", "duration_min": 110.4}
```

**What the job does:** reads the topic → parses JSON → casts types → drops malformed/incomplete
messages → appends to the parquet folder `data/stream/visits` → prints the 15-minute arrivals/exits
for the batch. Offsets are stored in `data/stream/_checkpoint`, so restarting the job continues
where it stopped.

### 4.1 Demo: 50 synthetic points every 30 seconds + prediction every 30 seconds

For a presentation/demo, use the workbook itself as the starting history and then move the simulated
clock into the future. The producer finds the **latest timestamp in the workbook** and generates its
first synthetic point at exactly `latest_timestamp + 30 seconds`, followed by one point every 30 seconds.
The default is 50 synthetic points (about 25 minutes of simulated future time).

The demo seeds the Kafka topic with roughly the last 24 hours of real workbook visits first. This is
intentional because the model has a 24-hour lag feature; using only five raw rows would not provide
reliable history for that feature.

Run the stream job with automatic prediction enabled:

```bash
python kafka_stream.py --interval 30 --predict
```

Then, in another terminal:

```bash
python kafka_producer.py --demo --points 50 --interval 30
```

The producer will print the exact clock, for example:

```text
Latest workbook timestamp: 2026-09-18 10:24:02.478000
Streaming 50 synthetic points every 30s starting at 2026-09-18 10:24:32.478000
  demo 01/50: event_time=2026-09-18 10:24:32.478000
  demo 02/50: event_time=2026-09-18 10:25:02.478000
  ...
```

Only synthetic demo messages contain `event_time`. Spark uses that field as the exact prediction
clock, so a synthetic visit can have a future `exit_time` without moving the forecast timestamp ahead.
After each synthetic point is received, Spark runs the existing XGBoost point/P50/P90 models and
prints the six configured horizons (15m, 30m, 1h, 2h, 3h, 6h). It also writes one JSON result per
prediction to `data/stream/predictions.jsonl`.

Useful demo overrides:

```bash
python kafka_producer.py --demo --points 30 --interval 30
python kafka_producer.py --demo --points 50 --interval 30 --seed-hours 24
```

If you change the interval, keep `kafka_stream.py --interval` close to the same value for the
presentation so each micro-batch normally contains one synthetic point.

**Use the streamed data**

```bash
python predict.py --raw data/stream/visits --prediction-time "2026-09-17 10:17:00"
# or the API:  POST /predict/stream?prediction_time=...
# or retrain:  python retrain.py --raw data/stream/visits
```

**Demo note.** The synthetic demo messages represent future gate activity and include an
`event_time` clock for presentation purposes. This is separate from the existing real-data replay
mode. The underlying model and feature definitions are unchanged.

**Other commands**

```bash
python kafka_stream.py --once            # process what is already in the topic, then exit
python kafka_stream.py --starting-offsets latest   # ignore history on the very first start
docker compose down                      # stop Kafka
```

**Reset everything** (needed before replaying the same file, otherwise visits are stored twice):

```bash
rm -rf data/stream
docker compose down && docker compose up -d     # also clears the topic
```

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `JAVA_HOME is not set` / `Unable to locate a Java Runtime` | Install Java 17/21 and set `JAVA_HOME`. |
| `unresolved dependency ... spark-sql-kafka` | Machine can't reach Maven Central (proxy/offline). Allow `repo1.maven.org`, or download the jar once and run with `spark.jars` instead. |
| `NoBrokersAvailable` / producer hangs | Kafka not up: `docker compose ps`, port 9092 free? |
| `Input not found: data/stream/visits` | Stream job hasn't stored a batch yet — start the producer and wait for `[batch N] stored ...`. |
| `Form data requires "python-multipart"` | `pip install python-multipart` (already in requirements.txt). |
| Spark prints `No Partition Defined for Window operation` | Expected: a time series needs global order and the 15-min grid is small. |

## Notes

* `active_vehicles` is an in-system occupancy proxy, not a physical queue length.
* The Aug 22–23 unreliable period and the Aug 24 washout are never fabricated or zero-filled.
* Excel files are read with pandas (Spark cannot read `.xlsx` natively) and then handed to Spark; CSV
  and parquet are read directly by Spark.
* The LightGBM point benchmark from the old `train.py` was dropped to keep things simple. P50/P90 are
  XGBoost quantile models as before.
* Spark runs locally (`local[*]`). To use a cluster later, set `SPARK_MASTER=spark://host:7077`.
