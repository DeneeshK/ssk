"""Publish vehicle visits to Kafka.

Two modes are supported:

1. Normal replay (existing behaviour): replay workbook visits in Exit-Time order.
2. Demo mode: seed Kafka with recent real workbook history, then generate a small future
   synthetic stream starting exactly 30 seconds after the latest workbook timestamp.

Demo example:
    python kafka_producer.py --demo --points 50 --interval 30

The demo messages include ``event_time`` only for synthetic points. The Spark stream uses that
field as the exact prediction clock, so future Exit Times do not move the prediction timestamp.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

DEFAULT_RAW = Path("data/raw/Vehicle_Data_ DwellTime _Tra.xlsx")
RAW_TO_SNAKE = {
    "Entry Time": "entry_time",
    "Exit Time": "exit_time",
    "Risk Category": "risk_category",
    "Duration in Min": "duration_min",
}


def load_events(path: Path, from_time: str | None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="InOut")
    df = df.rename(columns=RAW_TO_SNAKE)[list(RAW_TO_SNAKE.values())]
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    df["risk_category"] = df["risk_category"].fillna("UNKNOWN")
    df = df.dropna(subset=["entry_time", "exit_time", "duration_min"]).sort_values("exit_time")
    if from_time:
        df = df[df["exit_time"] >= pd.Timestamp(from_time)]
    return df.reset_index(drop=True)


def latest_workbook_time(events: pd.DataFrame) -> pd.Timestamp:
    if events.empty:
        raise ValueError("Workbook contains no valid visits.")
    return max(events["entry_time"].max(), events["exit_time"].max())


def make_producer(bootstrap: str):
    errors: list[str] = []

    def on_delivery(err, msg):
        if err is not None:
            errors.append(str(err))

    return Producer({"bootstrap.servers": bootstrap}), errors, on_delivery


def publish_payloads(
    producer: Producer,
    topic: str,
    payloads,
    rate: float,
    errors: list[str],
    on_delivery,
    label: str,
) -> int:
    delay = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    for payload in payloads:
        producer.produce(
            topic,
            value=json.dumps(payload).encode("utf-8"),
            on_delivery=on_delivery,
        )
        producer.poll(0)
        sent += 1
        if delay:
            time.sleep(delay)
        if sent % 500 == 0:
            print(f"  {label}: sent {sent}")
    producer.flush()
    if errors:
        print(f"{label}: {len(errors)} delivery error(s); first: {errors[0]}")
    return sent


def row_payload(row) -> dict:
    return {
        "entry_time": pd.Timestamp(row.entry_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
        "exit_time": pd.Timestamp(row.exit_time).strftime("%Y-%m-%d %H:%M:%S.%f"),
        "risk_category": str(row.risk_category),
        "duration_min": float(row.duration_min),
    }


def demo_payloads(
    events: pd.DataFrame,
    points: int,
    interval_seconds: int,
    seed_hours: int,
    seed_rate: float,
    bootstrap: str,
    topic: str,
    start_delay: float,
    random_seed: int,
) -> tuple[pd.Timestamp, int]:
    """Publish recent real history, then synthetic future visits at fixed 30s-style ticks."""
    if points <= 0:
        raise ValueError("--points must be > 0.")
    if interval_seconds <= 0:
        raise ValueError("--interval must be > 0 seconds.")
    if seed_hours <= 0:
        raise ValueError("--seed-hours must be > 0.")

    base_time = latest_workbook_time(events)
    seed_start = base_time - pd.Timedelta(hours=seed_hours)

    # Keep all completed visits that could influence the recent/24h feature history.
    seed = events[events["exit_time"] >= seed_start].copy().sort_values("exit_time")
    if seed.empty:
        raise ValueError("No recent workbook rows are available for seeding.")

    producer, errors, on_delivery = make_producer(bootstrap)
    print(f"Latest workbook timestamp: {base_time}")
    print(f"Seeding {len(seed)} real visits covering approximately the last {seed_hours} hours...")

    seed_payloads = (row_payload(row) for row in seed.itertuples(index=False))
    publish_payloads(
        producer,
        topic,
        seed_payloads,
        seed_rate,
        errors,
        on_delivery,
        "seed",
    )

    print(f"Seed complete. Waiting {start_delay:g}s before synthetic stream...")
    if start_delay > 0:
        time.sleep(start_delay)

    rng = random.Random(random_seed)
    duration_pool = seed["duration_min"].astype(float).clip(lower=0.1).tolist()
    risk_pool = seed["risk_category"].astype(str).tolist()

    # The FIRST synthetic event is exactly one interval after the latest Excel timestamp.
    synthetic = []
    for i in range(1, points + 1):
        event_time = base_time + pd.Timedelta(seconds=interval_seconds * i)
        duration = float(rng.choice(duration_pool))
        # The synthetic exit is deliberately in the future. event_time is the prediction clock,
        # so the future exit does not pull the prediction timestamp ahead.
        exit_time = event_time + pd.Timedelta(minutes=duration)
        synthetic.append({
            "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "entry_time": event_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "risk_category": rng.choice(risk_pool) if risk_pool else "UNKNOWN",
            "duration_min": duration,
        })

    print(
        f"Streaming {len(synthetic)} synthetic points every {interval_seconds}s "
        f"starting at {synthetic[0]['event_time']}"
    )
    sent = 0
    delay = float(interval_seconds)
    for payload in synthetic:
        producer.produce(
            topic,
            value=json.dumps(payload).encode("utf-8"),
            on_delivery=on_delivery,
        )
        producer.poll(0)
        sent += 1
        print(f"  demo {sent:02d}/{points}: event_time={payload['event_time']}")
        if sent < points:
            time.sleep(delay)

    producer.flush()
    if errors:
        print(f"Synthetic stream finished with {len(errors)} delivery error(s); first: {errors[0]}")
    else:
        print("Synthetic stream finished successfully.")
    return base_time, sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish vehicle visits to Kafka.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="vehicle-visits")
    parser.add_argument("--rate", type=float, default=20.0, help="normal replay messages per second (0 = as fast as possible)")
    parser.add_argument("--from-time", default=None, help="normal replay: only visits that exit at/after this time")
    parser.add_argument("--limit", type=int, default=0, help="normal replay: stop after N messages (0 = all)")

    parser.add_argument("--demo", action="store_true", help="seed recent real history, then generate future synthetic points")
    parser.add_argument("--points", type=int, default=50, help="demo synthetic points (default: 50)")
    parser.add_argument("--interval", type=int, default=30, help="demo interval between synthetic points in seconds (default: 30)")
    parser.add_argument("--seed-hours", type=int, default=24, help="demo real-history window before the latest workbook time (default: 24h)")
    parser.add_argument("--seed-rate", type=float, default=500.0, help="demo seed messages per second (default: 500)")
    parser.add_argument("--start-delay", type=float, default=2.0, help="demo delay after seeding before first synthetic point")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    events = load_events(args.raw, None if args.demo else args.from_time)
    if not args.demo:
        if args.limit:
            events = events.head(args.limit)
        print(f"Sending {len(events)} visits to topic '{args.topic}' on {args.bootstrap} ...")
        producer, errors, on_delivery = make_producer(args.bootstrap)
        payloads = (row_payload(row) for row in events.itertuples(index=False))
        sent = publish_payloads(
            producer, args.topic, payloads, args.rate, errors, on_delivery, "replay"
        )
        print(f"Done. Sent {sent} messages.")
        return

    demo_payloads(
        events=events,
        points=args.points,
        interval_seconds=args.interval,
        seed_hours=args.seed_hours,
        seed_rate=args.seed_rate,
        bootstrap=args.bootstrap,
        topic=args.topic,
        start_delay=args.start_delay,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
