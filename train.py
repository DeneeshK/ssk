"""Training: Spark builds the dataset (feature engineering), XGBoost trains the models.

Spark does the heavy data work (grid, windows, features, targets, chronological split).
The supervised table is small (rows = 15-min buckets x 6 horizons), so it is collected with
toPandas() and fed to normal XGBoost - the same models, parameters and promotion rule as before.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import feature_pipeline as fp
from spark_session import get_spark

DEFAULT_RAW = Path("data/raw/Vehicle_Data_ DwellTime _Tra.xlsx")
DEFAULT_DATASET = Path("data/processed/supervised_multihorizon")  # parquet folder written by Spark
DEFAULT_MODELS_DIR = Path("models")

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 1500,
    "learning_rate": 0.051612,
    "max_depth": 6,
    "min_child_weight": 8,
    "subsample": 0.988352,
    "colsample_bytree": 0.952685,
    "gamma": 0.089951,
    "reg_alpha": 0.285222,
    "reg_lambda": 2.843754,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_version(models_dir: Path) -> str:
    versions = []
    for path in models_dir.glob("v[0-9][0-9][0-9]"):
        try:
            versions.append(int(path.name[1:]))
        except ValueError:
            pass
    return f"v{(max(versions, default=0) + 1):03d}"


def evaluate_by_horizon(model, df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for horizon in fp.HORIZONS:
        part = df[df["horizon_minutes"] == horizon]
        if part.empty:
            continue
        pred = model.predict(part[features])
        y_true = part["target"]
        mae = float(mean_absolute_error(y_true, pred))
        mse = float(mean_squared_error(y_true, pred))
        rows.append({"horizon_minutes": horizon, "mae": mae, "mse": mse, "rmse": float(np.sqrt(mse)), "n": int(len(part))})
    metrics_df = pd.DataFrame(rows)
    summary = {
        "mean_mae": float(metrics_df["mae"].mean()),
        "mean_mse": float(metrics_df["mse"].mean()),
        "mean_rmse": float(metrics_df["rmse"].mean()),
    }
    return metrics_df, summary


def quantile_metrics(model, df: pd.DataFrame, features: list[str], alpha: float) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for horizon in fp.HORIZONS:
        part = df[df["horizon_minutes"] == horizon]
        if part.empty:
            continue
        pred = np.asarray(model.predict(part[features]))
        y = part["target"].to_numpy()
        errors = y - pred
        pinball = np.maximum(alpha * errors, (alpha - 1.0) * errors)
        rows.append({
            "horizon_minutes": horizon,
            "pinball_loss": float(pinball.mean()),
            "coverage": float(np.mean(y <= pred)),
            "mae": float(mean_absolute_error(y, pred)),
            "n": int(len(part)),
        })
    metrics_df = pd.DataFrame(rows)
    summary = {
        "mean_pinball_loss": float(metrics_df["pinball_loss"].mean()),
        "mean_coverage": float(metrics_df["coverage"].mean()),
        "mean_mae": float(metrics_df["mae"].mean()),
    }
    return metrics_df, summary


def build_dataset(spark, raw_path: Path, dataset_path: Path) -> DataFrame:
    """Spark: raw visits -> wide features -> long supervised table (saved as parquet)."""
    raw = fp.load_raw(spark, raw_path)
    wide = fp.build_wide_frame(raw)
    long_df = fp.build_training_set(wide).cache()
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.write.mode("overwrite").parquet(str(dataset_path))
    return long_df


def to_pandas(df: DataFrame) -> pd.DataFrame:
    return df.orderBy("timestamp", "horizon_minutes").toPandas()


def train_point_xgb(train: pd.DataFrame, val: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        train[fp.MODEL_FEATURES], train["target"],
        eval_set=[(val[fp.MODEL_FEATURES], val["target"])], verbose=False,
    )
    return model


def train_quantile_xgb(train: pd.DataFrame, val: pd.DataFrame, alpha: float) -> XGBRegressor:
    params = dict(XGB_PARAMS)
    params.update({"objective": "reg:quantileerror", "quantile_alpha": alpha})
    model = XGBRegressor(**params)
    model.fit(
        train[fp.MODEL_FEATURES], train["target"],
        eval_set=[(val[fp.MODEL_FEATURES], val["target"])], verbose=False,
    )
    return model


def current_active_version(models_dir: Path) -> str | None:
    path = models_dir / "active_version.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("active_version")


def load_active_metrics(models_dir: Path) -> dict | None:
    version = current_active_version(models_dir)
    if not version:
        return None
    metrics_path = models_dir / version / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text()).get("point_model", {}).get("validation_summary")


def should_promote(candidate: dict, active: dict | None) -> tuple[bool, str]:
    if active is None:
        return True, "No previous active model metrics found; promoting first candidate."
    if candidate["mean_mae"] < active["mean_mae"] and candidate["mean_mse"] < active["mean_mse"]:
        return True, "Candidate validation MAE and MSE are both lower than the active model."
    return False, "Candidate did not improve both validation MAE and MSE; active model retained."


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SSK multi-horizon occupancy models (Spark + XGBoost).")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="Excel/CSV file or a parquet folder (e.g. data/stream/visits from Kafka).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args()

    spark = get_spark("ssk-train")
    args.models_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(spark, args.raw, args.dataset)
    train_s, val_s, test_s = fp.split_chronological(dataset)
    train, val, test = to_pandas(train_s), to_pandas(val_s), to_pandas(test_s)

    if train.empty or val.empty or test.empty:
        raise ValueError("Chronological split produced an empty train, validation, or test set.")

    version = next_version(args.models_dir)
    version_dir = args.models_dir / version
    version_dir.mkdir(parents=True, exist_ok=False)

    print(f"Dataset rows: {len(train) + len(val) + len(test)}")
    print(f"Train/validation/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"Training version: {version}")

    point = train_point_xgb(train, val)
    point_val_df, point_val_summary = evaluate_by_horizon(point, val, fp.MODEL_FEATURES)
    point_test_df, point_test_summary = evaluate_by_horizon(point, test, fp.MODEL_FEATURES)

    p50 = train_quantile_xgb(train, val, 0.50)
    p50_val_df, p50_val_summary = quantile_metrics(p50, val, fp.MODEL_FEATURES, 0.50)
    p50_test_df, p50_test_summary = quantile_metrics(p50, test, fp.MODEL_FEATURES, 0.50)

    p90 = train_quantile_xgb(train, val, 0.90)
    p90_val_df, p90_val_summary = quantile_metrics(p90, val, fp.MODEL_FEATURES, 0.90)
    p90_test_df, p90_test_summary = quantile_metrics(p90, test, fp.MODEL_FEATURES, 0.90)

    joblib.dump(point, version_dir / "xgb_multihorizon_point.joblib")
    joblib.dump(p50, version_dir / "xgb_multihorizon_p50.joblib")
    joblib.dump(p90, version_dir / "xgb_multihorizon_p90.joblib")

    schema = {
        "feature_names": fp.MODEL_FEATURES,
        "feature_count": len(fp.MODEL_FEATURES),
        "horizons_minutes": fp.HORIZONS,
        "target": "target",
        "timestamp": "timestamp",
    }
    (version_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    metrics = {
        "version": version,
        "created_at_utc": utc_now(),
        "data": {
            "raw_file": str(args.raw),
            "processed_file": str(args.dataset),
            "train_end": fp.TRAIN_END.isoformat(),
            "validation_end": fp.VALIDATION_END.isoformat(),
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
        },
        "point_model": {
            "model": "xgboost_multihorizon_point",
            "parameters": XGB_PARAMS,
            "validation_by_horizon": point_val_df.to_dict(orient="records"),
            "validation_summary": point_val_summary,
            "test_by_horizon": point_test_df.to_dict(orient="records"),
            "test_summary": point_test_summary,
        },
        "p50_model": {
            "model": "xgboost_multihorizon_p50",
            "alpha": 0.50,
            "validation_by_horizon": p50_val_df.to_dict(orient="records"),
            "validation_summary": p50_val_summary,
            "test_by_horizon": p50_test_df.to_dict(orient="records"),
            "test_summary": p50_test_summary,
        },
        "p90_model": {
            "model": "xgboost_multihorizon_p90",
            "alpha": 0.90,
            "validation_by_horizon": p90_val_df.to_dict(orient="records"),
            "validation_summary": p90_val_summary,
            "test_by_horizon": p90_test_df.to_dict(orient="records"),
            "test_summary": p90_test_summary,
        },
    }
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    import pyspark
    import sys
    import xgboost
    metadata = {
        "version": version,
        "created_at_utc": utc_now(),
        "feature_count": len(fp.MODEL_FEATURES),
        "feature_names": fp.MODEL_FEATURES,
        "horizons_minutes": fp.HORIZONS,
        "software": {
            "python": sys.version,
            "pyspark": pyspark.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    (version_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    candidate = point_val_summary
    active_version = current_active_version(args.models_dir)
    active_metrics = load_active_metrics(args.models_dir)
    promote, reason = should_promote(candidate, active_metrics)

    registry = {
        "active_version": version if promote else active_version,
        "updated_at_utc": utc_now(),
        "decision": "promoted" if promote else "candidate_rejected",
        "reason": reason,
        "candidate_version": version,
        "candidate_validation": candidate,
        "previous_active_version": active_version,
        "previous_active_validation": active_metrics,
    }
    (args.models_dir / "active_version.json").write_text(json.dumps(registry, indent=2))

    print("\nXGBoost point test metrics:")
    print(point_test_df.to_string(index=False))
    print(f"\nP90 mean test coverage: {p90_test_summary['mean_coverage']:.4f}")
    print(f"\nPromotion decision: {registry['decision']} - {reason}")


if __name__ == "__main__":
    main()
