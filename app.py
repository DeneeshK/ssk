from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from predict import predict, predict_peak
from spark_session import get_spark

RAW_DEFAULT = Path("data/raw/Vehicle_Data_ DwellTime _Tra.xlsx")
STREAM_DIR = Path("data/stream/visits")  # parquet written by kafka_stream.py
MODELS_DIR = Path("models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_spark("ssk-api")  # start Spark once at boot so the first request is not slow
    yield


app = FastAPI(title="SSK Vehicle Occupancy Forecast API (Spark)", version="2.0.0", lifespan=lifespan)


async def _run(fn, path: Path, prediction_time: str | None) -> dict:
    """Run a (blocking, Spark-backed) prediction function off the event loop; map errors to HTTP 400."""
    try:
        return await run_in_threadpool(fn, path, MODELS_DIR, prediction_time)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _forecast(fn, file: UploadFile | None, prediction_time: str | None) -> dict:
    if file is None:
        return await _run(fn, RAW_DEFAULT, prediction_time)

    suffix = Path(file.filename or "input.xlsx").suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Please upload an Excel .xlsx or .xls file.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(await file.read())
        return await _run(fn, temp_path, prediction_time)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def forecast(
    file: UploadFile | None = File(default=None),
    prediction_time: str | None = None,
) -> dict:
    """Predict occupancy from an optional uploaded raw InOut Excel workbook.

    If no file is uploaded, the default workbook under data/raw/ is used.
    """
    return await _forecast(predict, file, prediction_time)


@app.post("/predict/peak")
async def forecast_peak(
    file: UploadFile | None = File(default=None),
    prediction_time: str | None = None,
) -> dict:
    """Return the highest predicted occupancy across all configured horizons."""
    return await _forecast(predict_peak, file, prediction_time)


@app.post("/predict/stream")
async def forecast_from_stream(prediction_time: str | None = None) -> dict:
    """Predict from the visits that the Kafka streaming job has landed in data/stream/visits."""
    return await _run(predict, STREAM_DIR, prediction_time)


@app.post("/predict/stream/peak")
async def forecast_peak_from_stream(prediction_time: str | None = None) -> dict:
    """Peak forecast from the Kafka-fed data."""
    return await _run(predict_peak, STREAM_DIR, prediction_time)
