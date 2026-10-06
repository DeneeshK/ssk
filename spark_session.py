"""Single place that builds the SparkSession used by training, inference, the API and streaming."""

from __future__ import annotations

import os
import time

import pyspark


def kafka_package() -> str:
    """Kafka connector coordinates that match the installed PySpark (Scala 2.13 for Spark 4, 2.12 for 3.x)."""
    major = int(pyspark.__version__.split(".")[0])
    scala = "2.13" if major >= 4 else "2.12"
    return f"org.apache.spark:spark-sql-kafka-0-10_{scala}:{pyspark.__version__}"


def get_spark(app_name: str = "ssk-occupancy", with_kafka: bool = False):
    """Return (or reuse) a local SparkSession.

    Everything runs in UTC so naive timestamps from Excel / API requests mean the same thing in
    Python and in Spark. Set SPARK_MASTER to point at a real cluster (default: local[*]).
    """
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraJavaOptions", "-Duser.timezone=UTC")
        .config("spark.sql.ansi.enabled", "false")  # same cast-to-null behaviour on Spark 3.5 and 4.x
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.ui.showConsoleProgress", "false")
    )
    if with_kafka:
        builder = builder.config("spark.jars.packages", kafka_package())

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
