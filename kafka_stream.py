"""Spark Structured Streaming job: Kafka -> clean -> parquet, with optional 30s demo prediction.

Normal mode:
    python kafka_stream.py

Demo mode (automatic prediction once per synthetic event):
    python kafka_stream.py --interval 30 --predict

The producer's synthetic messages carry ``event_time``. That field is the exact prediction clock,
so a synthetic future Exit Time never pushes the prediction timestamp into the future.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

import feature_pipeline as fp
from spark_session import get_spark

EVENT_SCHEMA = StructType([
    StructField("event_time", StringType()),
    StructField("entry_time", StringType()),
    StructField("exit_time", StringType()),
    StructField("risk_category", StringType()),
    StructField("duration_min", DoubleType()),
])


def parse_events(kafka_df: DataFrame) -> DataFrame:
    """Kafka JSON -> typed visit rows. event_time is optional and used only by the demo predictor."""
    parsed = kafka_df.select(
        F.from_json(F.col("value").cast("string"), EVENT_SCHEMA).alias("e")
    ).select("e.*")
    return (
        parsed.select(
            F.col("event_time").cast("timestamp").alias("event_time"),
            F.col("entry_time").cast("timestamp").alias("entry_time"),
            F.col("exit_time").cast("timestamp").alias("exit_time"),
            F.coalesce(F.col("risk_category"), F.lit("UNKNOWN")).alias("risk_category"),
            F.col("duration_min"),
        )
        .where(
            F.col("entry_time").isNotNull()
            & F.col("exit_time").isNotNull()
            & F.col("duration_min").isNotNull()
            & (F.col("exit_time") >= F.col("entry_time"))
        )
    )


def flow_summary(batch_df: DataFrame) -> DataFrame:
    """Arrivals / exits per 15-minute bucket for one micro-batch."""
    arrivals = batch_df.groupBy(fp.bucket_ts(F.col("entry_time")).alias("bucket")).agg(
        F.count(F.lit(1)).alias("arrivals")
    )
    exits = batch_df.groupBy(fp.bucket_ts(F.col("exit_time")).alias("bucket")).agg(
        F.count(F.lit(1)).alias("exits")
    )
    return arrivals.join(exits, "bucket", "full_outer").fillna(0).orderBy(F.desc("bucket"))


def run_prediction(
    stream_dir: Path,
    models_dir: Path,
    prediction_time,
    predictions_out: Path,
) -> None:
    """Run one exact-time forecast and append the JSON result to a local JSONL file."""
    from predict import predict

    result = predict(stream_dir, models_dir, prediction_time=prediction_time)
    peak = max(result["forecast"], key=lambda item: item["point_prediction"])
    result["peak_forecast"] = {
        "horizon_minutes": peak["horizon_minutes"],
        "point_prediction": peak["point_prediction"],
        "p50_prediction": peak["p50_prediction"],
        "p90_prediction": peak["p90_prediction"],
        "forecast_timestamp": (
            prediction_time + pd.Timedelta(minutes=int(peak["horizon_minutes"]))
        ).isoformat(),
    }

    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with predictions_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    print("[prediction]")
    print(json.dumps(result, indent=2, default=str))


def make_batch_handler(
    out_dir: Path,
    predict_enabled: bool = False,
    models_dir: Path = Path("models"),
    predictions_out: Path = Path("data/stream/predictions.jsonl"),
):
    def process_batch(batch_df: DataFrame, batch_id: int) -> None:
        batch_df = batch_df.cache()
        n = batch_df.count()
        if n == 0:
            batch_df.unpersist()
            return

        batch_df.write.mode("append").parquet(str(out_dir))
        print(f"\n[batch {batch_id}] stored {n} visits -> {out_dir}")
        flow_summary(batch_df).show(6, truncate=False)

        # Only synthetic demo messages have event_time. Therefore the seed/history batch does not
        # trigger a prediction, while each later 30s demo point does.
        if predict_enabled:
            prediction_times = (
                batch_df.select("event_time")
                .where(F.col("event_time").isNotNull())
                .distinct()
                .orderBy("event_time")
                .collect()
            )
            for item in prediction_times:
                run_prediction(
                    out_dir,
                    models_dir,
                    item["event_time"],
                    predictions_out,
                )

        batch_df.unpersist()

    return process_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka -> Spark Structured Streaming -> parquet (+ optional demo predictions).")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="vehicle-visits")
    parser.add_argument("--out", type=Path, default=Path("data/stream/visits"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/stream/_checkpoint"))
    parser.add_argument("--starting-offsets", default="earliest", choices=["earliest", "latest"],
                        help="only used the first time (afterwards the checkpoint remembers the position)")
    parser.add_argument("--interval", type=int, default=10, help="micro-batch interval in seconds")
    parser.add_argument("--once", action="store_true", help="process what is in the topic now, then stop")
    parser.add_argument("--predict", action="store_true", help="run an exact-time prediction for each synthetic event_time")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--predictions-out", type=Path, default=Path("data/stream/predictions.jsonl"))
    args = parser.parse_args()

    spark = get_spark("ssk-kafka-stream", with_kafka=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribe", args.topic)
        .option("startingOffsets", args.starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    writer = (
        parse_events(raw)
        .writeStream.foreachBatch(
            make_batch_handler(
                args.out,
                predict_enabled=args.predict,
                models_dir=args.models_dir,
                predictions_out=args.predictions_out,
            )
        )
        .option("checkpointLocation", str(args.checkpoint))
    )
    writer = writer.trigger(availableNow=True) if args.once else writer.trigger(processingTime=f"{args.interval} seconds")

    print(f"Streaming from topic '{args.topic}' ({args.bootstrap}) into {args.out} ...")
    if args.predict:
        print(f"Automatic prediction enabled; predictions will be triggered by Kafka event_time and written to {args.predictions_out}.")
    query = writer.start()
    query.awaitTermination()


if __name__ == "__main__":
    main()
