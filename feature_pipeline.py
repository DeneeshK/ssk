"""
feature_pipeline.py  (PySpark version)
======================================
Single source of truth for turning raw vehicle visits (Entry Time / Exit Time) into model input.
Training, retraining, inference and the Kafka stream all use this file.

The feature definitions are the same as the pandas version; only the engine changed:
pandas shift / rolling / cumsum  ->  Spark window functions (lag, lead, sum/avg over rowsBetween).

Pipeline
--------
raw input -> load_raw -> validate_and_clean -> build_occupancy_grid -> add_reliability
          -> add_features (+ targets) -> `wide` frame
`wide` frame -> build_training_set()         (long format: one row per timestamp x horizon)
cleaned visits -> inference_frame_at()       (exact-timestamp features for one prediction time)

Accepted raw inputs (load_raw):
  * .xlsx / .xls  (sheet `InOut`, read with pandas because Spark cannot read Excel natively)
  * .csv          (header row, same column names)
  * a parquet folder (this is what the Kafka stream job writes: data/stream/visits)

Timestamp semantics (unchanged)
-------------------------------
Each grid row is labelled with the START of its 15-minute bucket. `active_vehicles` at row t is the
cumulative net flow through the END of bucket t. Flow / rolling features at t use only buckets BEFORE t.
`active_vehicles` is an in-system occupancy proxy, NOT a physical queue length.

Reliability (unchanged)
-----------------------
Aug 22, Aug 23 and Aug 24 00:00-11:45 are unreliable. Values are never modified or zero-filled; any
feature whose source interval is unreliable is NULL, any target in an unreliable interval is NULL, and
rows with any NULL are dropped from the supervised data.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd
from pyspark.errors import AnalysisException
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
RAW_SHEET = "InOut"
RAW_COLUMNS = [
    "Country", "Plate Prefix", "Plate Number", "Entry Time", "Exit Time",
    "Risk Category", "Duration in Min", "Time Taken",
]
# Excel column name -> snake_case name used inside Spark / parquet (no spaces).
RAW_TO_SNAKE = {
    "Entry Time": "entry_time",
    "Exit Time": "exit_time",
    "Risk Category": "risk_category",
    "Duration in Min": "duration_min",
}
KEEP_COLUMNS = list(RAW_TO_SNAKE.values())

INTERVAL_MINUTES = 15
INTERVAL_SECONDS = INTERVAL_MINUTES * 60

UNRELIABLE_DATES = ["2026-08-22", "2026-08-23"]
WASHOUT_START = pd.Timestamp("2026-08-24 00:00")
WASHOUT_END = pd.Timestamp("2026-08-24 12:00")  # exclusive -> last bad bucket is 11:45

LAGS = {
    "lag_15m": 1, "lag_30m": 2, "lag_1h": 4, "lag_2h": 8,
    "lag_3h": 12, "lag_4h": 16, "lag_6h": 24, "lag_24h": 96,
}
FLOW_WINDOWS = {"15m": 1, "30m": 2, "1h": 4}
FLOW_SOURCES = ["arrivals", "exits", "net_flow"]
ROLLING_STEPS = 4  # 1 hour

HORIZON_STEPS = {15: 1, 30: 2, 60: 4, 120: 8, 180: 12, 360: 24}
HORIZONS = list(HORIZON_STEPS)

BASE_FEATURES = [
    "active_vehicles",
    "lag_15m", "lag_30m", "lag_1h", "lag_2h", "lag_3h", "lag_4h", "lag_6h", "lag_24h",
    "arrivals_15m", "arrivals_30m", "arrivals_1h",
    "exits_15m", "exits_30m", "exits_1h",
    "net_flow_15m", "net_flow_30m", "net_flow_1h",
    "occupancy_change_15m", "occupancy_change_1h", "occupancy_change_2h",
    "rolling_mean_occupancy_1h", "rolling_mean_arrivals_1h",
    "rolling_mean_exits_1h", "rolling_mean_net_flow_1h",
    "hour", "day_of_week", "is_weekend",
]
MODEL_FEATURES = BASE_FEATURES + ["horizon_minutes"]  # 29 columns, fixed order
LONG_COLUMNS = ["timestamp"] + MODEL_FEATURES + ["target"]

TRAIN_END = pd.Timestamp("2026-09-05 04:45:00")        # train:      timestamp <  TRAIN_END
VALIDATION_END = pd.Timestamp("2026-09-11 18:00:00")   # validation: TRAIN_END <= timestamp < VALIDATION_END
                                                       # test:       timestamp >= VALIDATION_END


class DataValidationError(ValueError):
    """The raw input does not meet the expected schema or content rules."""


class InsufficientHistoryError(ValueError):
    """Not enough reliable history to build the features."""


def target_column(minutes: int) -> str:
    return f"target_{minutes}m"


def ts_lit(value) -> Column:
    """Python timestamp -> Spark timestamp literal."""
    return F.lit(pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")).cast("timestamp")


def bucket_ts(col: Column) -> Column:
    """Floor a timestamp column to the start of its 15-minute bucket."""
    return F.timestamp_seconds(F.floor(F.unix_timestamp(col) / INTERVAL_SECONDS) * INTERVAL_SECONDS)


# --------------------------------------------------------------------------------------
# 1. Raw input
# --------------------------------------------------------------------------------------
def load_raw(spark: SparkSession, path) -> DataFrame:
    """Read an Excel workbook, CSV file or parquet folder into a Spark DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    suffix = p.suffix.lower()

    try:
        if p.is_dir() or suffix == ".parquet":
            return spark.read.parquet(str(p))
        if suffix == ".csv":
            return spark.read.option("header", True).csv(str(p))
    except AnalysisException as e:
        raise DataValidationError(f"Could not read {p}: {e}") from e

    if suffix in {".xlsx", ".xls"}:
        try:
            pdf = pd.read_excel(p, sheet_name=RAW_SHEET)
        except ValueError as e:  # e.g. sheet not found
            raise DataValidationError(f"Could not read sheet '{RAW_SHEET}' from {p}: {e}") from e
        pdf = pdf.rename(columns=RAW_TO_SNAKE)
        cols = [c for c in KEEP_COLUMNS if c in pdf.columns]
        pdf = pdf[cols].copy()
        for c in ("entry_time", "exit_time"):
            if c in pdf.columns:
                pdf[c] = pd.to_datetime(pdf[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        rows = [
            tuple(None if pd.isna(v) else str(v) for v in row)
            for row in pdf.itertuples(index=False, name=None)
        ]
        schema = StructType([StructField(c, StringType(), True) for c in cols])
        return spark.createDataFrame(rows, schema)

    raise DataValidationError(f"Unsupported input '{p.name}'. Use .xlsx, .xls, .csv or a parquet folder.")


def validate_and_clean(raw: DataFrame) -> DataFrame:
    """Validate the schema and return typed columns: entry_time, exit_time, risk_category, duration_min."""
    for old, new in RAW_TO_SNAKE.items():
        if old in raw.columns:
            raw = raw.withColumnRenamed(old, new)

    missing = [c for c in KEEP_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s) {missing}. Expected raw columns: {RAW_COLUMNS}"
        )

    df = raw.select(
        F.col("entry_time").cast("timestamp").alias("entry_time"),
        F.col("exit_time").cast("timestamp").alias("exit_time"),
        F.coalesce(F.col("risk_category").cast("string"), F.lit("UNKNOWN")).alias("risk_category"),
        F.col("duration_min").cast("double").alias("duration_min"),
    )

    # One Spark job collects every validation statistic.
    s = df.agg(
        F.count(F.lit(1)).alias("n"),
        F.sum(F.col("entry_time").isNull().cast("int")).alias("bad_entry"),
        F.sum(F.col("exit_time").isNull().cast("int")).alias("bad_exit"),
        F.sum(F.col("duration_min").isNull().cast("int")).alias("bad_duration"),
        F.sum((F.col("exit_time") < F.col("entry_time")).cast("int")).alias("inverted"),
    ).first()

    if not s["n"]:
        raise DataValidationError("Raw input contains no rows.")
    for col, key in (("entry_time", "bad_entry"), ("exit_time", "bad_exit")):
        if s[key]:
            raise DataValidationError(
                f"Column '{col}' has {s[key]} missing or unparseable value(s). Every row must have "
                f"both an Entry Time and an Exit Time."
            )
    if s["inverted"]:
        raise DataValidationError(f"{s['inverted']} row(s) have Exit Time earlier than Entry Time.")
    if s["bad_duration"]:
        raise DataValidationError("Column 'duration_min' contains missing or non-numeric values.")
    return df


# --------------------------------------------------------------------------------------
# 2. Canonical 15-minute grid + occupancy
# --------------------------------------------------------------------------------------
def build_occupancy_grid(df: DataFrame) -> DataFrame:
    """Complete 15-min grid with arrivals, exits, net_flow, active_vehicles."""
    arrivals = df.groupBy(bucket_ts(F.col("entry_time")).alias("timestamp")).agg(
        F.count(F.lit(1)).alias("arrivals")
    )
    exits = df.groupBy(bucket_ts(F.col("exit_time")).alias("timestamp")).agg(
        F.count(F.lit(1)).alias("exits")
    )

    bounds = df.agg(
        F.min(bucket_ts(F.col("entry_time"))).alias("start"),
        F.max(bucket_ts(F.col("exit_time"))).alias("end"),
    )
    grid = bounds.select(
        F.explode(F.sequence("start", "end", F.expr(f"INTERVAL {INTERVAL_MINUTES} MINUTES"))).alias("timestamp")
    )

    grid = (
        grid.join(arrivals, "timestamp", "left")
        .join(exits, "timestamp", "left")
        .fillna(0, subset=["arrivals", "exits"])
        .withColumn("net_flow", F.col("arrivals") - F.col("exits"))  # Entry = +1, Exit = -1
    )
    # NOTE: windows below have no partition key on purpose (a time series needs global order).
    # The grid has only one row per 15 minutes, so this is tiny even for years of data.
    w_all = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return grid.withColumn("active_vehicles", F.sum("net_flow").over(w_all))


# --------------------------------------------------------------------------------------
# 3. Reliability
# --------------------------------------------------------------------------------------
def add_reliability(grid: DataFrame) -> DataFrame:
    """Add boolean `is_reliable`. Values are NOT altered, only flagged."""
    day = F.date_format("timestamp", "yyyy-MM-dd")
    unreliable = day.isin(UNRELIABLE_DATES) | (
        (F.col("timestamp") >= ts_lit(WASHOUT_START)) & (F.col("timestamp") < ts_lit(WASHOUT_END))
    )
    return grid.withColumn("is_reliable", ~unreliable)


# --------------------------------------------------------------------------------------
# 4. Features and targets (window functions)
# --------------------------------------------------------------------------------------
def add_features(grid: DataFrame) -> DataFrame:
    """Add the 28 base features and the 6 targets. A feature is NULL unless ALL its source intervals are reliable."""
    w = Window.orderBy("timestamp")
    g = grid.withColumn("_rel", F.col("is_reliable").cast("int"))

    # Occupancy lags: valid only if the interval the lag is read from is reliable.
    for name, steps in LAGS.items():
        source_ok = F.coalesce(F.lag("is_reliable", steps).over(w), F.lit(False))
        g = g.withColumn(name, F.when(source_ok, F.lag("active_vehicles", steps).over(w)))

    # Trailing flow sums over the completed intervals BEFORE t (rows t-window .. t-1).
    for source in FLOW_SOURCES:
        for suffix, window in FLOW_WINDOWS.items():
            ww = w.rowsBetween(-window, -1)
            ok = (F.count(F.lit(1)).over(ww) == window) & (F.sum("_rel").over(ww) == window)
            g = g.withColumn(f"{source}_{suffix}", F.when(ok, F.sum(source).over(ww)))

    # Occupancy changes (inherit NULL from the masked lags).
    g = (
        g.withColumn("occupancy_change_15m", F.col("active_vehicles") - F.col("lag_15m"))
        .withColumn("occupancy_change_1h", F.col("active_vehicles") - F.col("lag_1h"))
        .withColumn("occupancy_change_2h", F.col("active_vehicles") - F.col("lag_2h"))
    )

    # Backward-looking 1h rolling means (t-4 .. t-1); all four intervals must be reliable.
    wr = w.rowsBetween(-ROLLING_STEPS, -1)
    rolling_ok = (F.count(F.lit(1)).over(wr) == ROLLING_STEPS) & (F.sum("_rel").over(wr) == ROLLING_STEPS)
    rolling_sources = {
        "rolling_mean_occupancy_1h": "active_vehicles",
        "rolling_mean_arrivals_1h": "arrivals",
        "rolling_mean_exits_1h": "exits",
        "rolling_mean_net_flow_1h": "net_flow",
    }
    for name, source in rolling_sources.items():
        g = g.withColumn(name, F.when(rolling_ok, F.avg(source).over(wr)))

    # Calendar features. Spark dayofweek: Sun=1..Sat=7  ->  pandas style Mon=0..Sun=6.
    g = (
        g.withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", (F.dayofweek("timestamp") + 5) % 7)
        .withColumn("is_weekend", F.col("day_of_week").isin(5, 6).cast("int"))
    )

    # Direct multi-horizon targets: active_vehicles(t + H), NULL if t + H is unreliable or off-grid.
    for minutes, steps in HORIZON_STEPS.items():
        future_ok = F.coalesce(F.lead("is_reliable", steps).over(w), F.lit(False))
        g = g.withColumn(target_column(minutes), F.when(future_ok, F.lead("active_vehicles", steps).over(w)))

    return g.drop("_rel")


def build_wide_frame(raw: DataFrame) -> DataFrame:
    """THE shared entry point: raw DataFrame -> wide feature/target frame (one row per 15-min bucket)."""
    df = validate_and_clean(raw)
    grid = build_occupancy_grid(df)
    grid = add_reliability(grid)
    return add_features(grid)


# --------------------------------------------------------------------------------------
# 5. Training view: long format, one row per (timestamp, horizon)
# --------------------------------------------------------------------------------------
def build_training_set(wide: DataFrame) -> DataFrame:
    """Long-format supervised dataset. Rows with any NULL feature/target are dropped (no fillna)."""
    parts = []
    for minutes in HORIZONS:
        part = (
            wide.filter(F.col("is_reliable"))
            .select(
                "timestamp",
                *[F.col(c).cast("double").alias(c) for c in BASE_FEATURES],
                F.col(target_column(minutes)).cast("double").alias("target"),
            )
            .dropna()
            .withColumn("horizon_minutes", F.lit(minutes))
        )
        parts.append(part)
    return reduce(DataFrame.unionByName, parts).select(LONG_COLUMNS)


def split_chronological(long_df: DataFrame):
    """Chronological train / validation / test split on `timestamp` (never shuffled)."""
    ts = F.col("timestamp")
    train = long_df.filter(ts < ts_lit(TRAIN_END))
    validation = long_df.filter((ts >= ts_lit(TRAIN_END)) & (ts < ts_lit(VALIDATION_END)))
    test = long_df.filter(ts >= ts_lit(VALIDATION_END))
    return train, validation, test


# --------------------------------------------------------------------------------------
# 6. Inference (exact prediction timestamp, e.g. 10:17)
# --------------------------------------------------------------------------------------
def _is_unreliable_timestamp(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    if ts.normalize() in pd.to_datetime(UNRELIABLE_DATES):
        return True
    return WASHOUT_START <= ts < WASHOUT_END


def _interval_is_reliable(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True when the interval (start, end] does not overlap a bad period."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        return True
    for date_text in UNRELIABLE_DATES:
        day_start = pd.Timestamp(date_text)
        day_end = day_start + pd.Timedelta(days=1)
        if start < day_end and end > day_start:
            return False
    if start < WASHOUT_END and end > WASHOUT_START:
        return False
    return True


def _count_when(cond: Column) -> Column:
    return F.sum(F.when(cond, 1).otherwise(0))


def _active_at(ts: pd.Timestamp) -> Column:
    """Vehicles inside the system at an exact timestamp."""
    t = ts_lit(ts)
    return _count_when((F.col("entry_time") <= t) & (F.col("exit_time") > t))


def _arrivals_in(start: pd.Timestamp, end: pd.Timestamp) -> Column:
    return _count_when((F.col("entry_time") > ts_lit(start)) & (F.col("entry_time") <= ts_lit(end)))


def _exits_in(start: pd.Timestamp, end: pd.Timestamp) -> Column:
    return _count_when((F.col("exit_time") > ts_lit(start)) & (F.col("exit_time") <= ts_lit(end)))


def latest_event_time(clean: DataFrame) -> pd.Timestamp:
    row = clean.agg(F.max("entry_time").alias("e"), F.max("exit_time").alias("x")).first()
    return max(pd.Timestamp(row["e"]), pd.Timestamp(row["x"]))


def inference_frame_at(clean: DataFrame, prediction_time) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Build model features anchored to an exact prediction timestamp.

    `clean` is the output of validate_and_clean(). All counts are computed in ONE Spark aggregation
    over the visits; the result is a tiny pandas frame (one row per horizon, MODEL_FEATURES order).
    """
    ts = pd.Timestamp(prediction_time)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    if _is_unreliable_timestamp(ts):
        raise InsufficientHistoryError(
            f"Prediction time {ts} falls in a known unreliable period. Cannot predict from it."
        )

    lag_minutes = {
        "lag_15m": 15, "lag_30m": 30, "lag_1h": 60, "lag_2h": 120,
        "lag_3h": 180, "lag_4h": 240, "lag_6h": 360, "lag_24h": 1440,
    }
    for name, minutes in lag_minutes.items():
        lag_ts = ts - pd.Timedelta(minutes=minutes)
        if _is_unreliable_timestamp(lag_ts):
            raise InsufficientHistoryError(f"Cannot build {name}: source timestamp {lag_ts} is unreliable.")

    flow_minutes = {15: "15m", 30: "30m", 60: "1h"}
    for minutes in flow_minutes:
        if not _interval_is_reliable(ts - pd.Timedelta(minutes=minutes), ts):
            raise InsufficientHistoryError(
                f"Requested {minutes}-minute history ending at {ts} overlaps an unreliable period."
            )

    past_times = [ts - pd.Timedelta(minutes=15 * k) for k in (1, 2, 3, 4)]
    if any(_is_unreliable_timestamp(x) for x in past_times):
        raise InsufficientHistoryError(
            f"Cannot build 1-hour rolling features for {ts}: reliable history is unavailable."
        )
    for x in past_times:
        if not _interval_is_reliable(x - pd.Timedelta(minutes=15), x):
            raise InsufficientHistoryError(
                f"Requested 15-minute history ending at {x} overlaps an unreliable period."
            )

    # ---- one Spark aggregation computes every count -----------------------------------
    exprs = [F.min("entry_time").alias("first_entry"), _active_at(ts).alias("active_now")]
    for name, minutes in lag_minutes.items():
        exprs.append(_active_at(ts - pd.Timedelta(minutes=minutes)).alias(name))
    for minutes, suffix in flow_minutes.items():
        start = ts - pd.Timedelta(minutes=minutes)
        exprs.append(_arrivals_in(start, ts).alias(f"arrivals_{suffix}"))
        exprs.append(_exits_in(start, ts).alias(f"exits_{suffix}"))
    for k, x in enumerate(past_times, start=1):
        start = x - pd.Timedelta(minutes=15)
        exprs.append(_active_at(x).alias(f"past_active_{k}"))
        exprs.append(_arrivals_in(start, x).alias(f"past_arr_{k}"))
        exprs.append(_exits_in(start, x).alias(f"past_exit_{k}"))
    row = clean.agg(*exprs).first().asDict()

    first_entry = pd.Timestamp(row["first_entry"])
    if ts < first_entry:
        raise InsufficientHistoryError(
            f"Prediction time {ts} is earlier than the first observed entry {first_entry}."
        )
    val = {k: int(v or 0) for k, v in row.items() if k != "first_entry"}

    current_active = val["active_now"]
    lag_values = {name: val[name] for name in lag_minutes}
    flow_values = {}
    for suffix in flow_minutes.values():
        flow_values[f"arrivals_{suffix}"] = val[f"arrivals_{suffix}"]
        flow_values[f"exits_{suffix}"] = val[f"exits_{suffix}"]
        flow_values[f"net_flow_{suffix}"] = val[f"arrivals_{suffix}"] - val[f"exits_{suffix}"]

    ks = (1, 2, 3, 4)
    base = {
        "active_vehicles": current_active,
        **lag_values,
        **flow_values,
        "occupancy_change_15m": current_active - lag_values["lag_15m"],
        "occupancy_change_1h": current_active - lag_values["lag_1h"],
        "occupancy_change_2h": current_active - lag_values["lag_2h"],
        "rolling_mean_occupancy_1h": sum(val[f"past_active_{k}"] for k in ks) / 4.0,
        "rolling_mean_arrivals_1h": sum(val[f"past_arr_{k}"] for k in ks) / 4.0,
        "rolling_mean_exits_1h": sum(val[f"past_exit_{k}"] for k in ks) / 4.0,
        "rolling_mean_net_flow_1h": sum(val[f"past_arr_{k}"] - val[f"past_exit_{k}"] for k in ks) / 4.0,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek in (5, 6)),
    }
    rows = [{**base, "horizon_minutes": h} for h in HORIZONS]
    return ts, pd.DataFrame(rows, columns=MODEL_FEATURES)


def build_inference_frame(spark: SparkSession, path, prediction_time=None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Load raw visits (xlsx / csv / parquet folder) and build exact-time inference features.

    If `prediction_time` is omitted, the latest observed Entry/Exit time is used.
    """
    clean = validate_and_clean(load_raw(spark, path))
    if prediction_time is None:
        prediction_time = latest_event_time(clean)
    return inference_frame_at(clean, prediction_time)
