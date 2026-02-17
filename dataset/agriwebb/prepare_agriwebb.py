#!/usr/bin/env python3
"""Prepare AgriWebb data into TimeCAP-compatible processed artifacts.

This is a data-only pipeline. No model code is changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


WEATHER_RENAME = {
    "DAILY_RAIN": "daily_rain_mm",
    "MAX_TEMP": "max_temp_c",
    "MIN_TEMP": "min_temp_c",
    "DTR": "dtr_c",
    "AMB_TEMP": "amb_temp_c",
    "RH_TMAX": "rh_tmax_pct",
    "RH_TMIN": "rh_tmin_pct",
    "THI_DAILY": "thi_daily",
}

NUMERIC_FEATURES = [
    "weight_last_obs_kg",
    "days_since_last_weight",
    "age_months",
    "daily_rain_mm",
    "max_temp_c",
    "min_temp_c",
    "dtr_c",
    "amb_temp_c",
    "rh_tmax_pct",
    "rh_tmin_pct",
    "thi_daily",
    "load_mj_per_day",
    "paddock_land_size_ha",
    "paddock_arable_land_size_ha",
    "active_move_flag",
]

CAT_FEATURES = [
    "record_input_type",
    "sex",
    "neutered",
    "major_category",
    "age_class",
    "canon_breed",
    "pasture_state",
]


@dataclass
class PreparedSamples:
    x_time_raw: np.ndarray
    x_missing: np.ndarray
    x_text: np.ndarray
    y_class: np.ndarray
    y_adg: np.ndarray
    sample_meta: pd.DataFrame
    feature_names_base: List[str]
    numeric_feature_names: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AgriWebb data for TimeCAP")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing raw AgriWebb CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory for processed outputs",
    )
    parser.add_argument("--history-days", type=int, default=56)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--neutral-threshold", type=float, default=0.10)
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    return parser.parse_args()


def normalize_text_value(value: object, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    value = str(value).strip()
    return value if value else default


def load_tables(input_dir: Path) -> Dict[str, pd.DataFrame]:
    required = [
        "farm-info.csv",
        "paddock-info.csv",
        "paddock-moves.csv",
        "weather.csv",
        "weights.csv",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSV files: {missing}")

    tables = {
        "farm": pd.read_csv(input_dir / "farm-info.csv"),
        "paddock": pd.read_csv(input_dir / "paddock-info.csv"),
        "moves": pd.read_csv(input_dir / "paddock-moves.csv"),
        "weather": pd.read_csv(input_dir / "weather.csv"),
        "weights": pd.read_csv(input_dir / "weights.csv"),
    }
    return tables


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce").dt.floor("D")
    weather = weather.rename(columns=WEATHER_RENAME)
    keep_cols = ["FARM_ID", "DATE", *WEATHER_RENAME.values()]
    weather = weather[keep_cols].sort_values(["FARM_ID", "DATE"]).drop_duplicates(
        ["FARM_ID", "DATE"], keep="last"
    )
    return weather


def deduplicate_weights(weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.copy()
    weights["OBSERVATION_DATE"] = pd.to_datetime(weights["OBSERVATION_DATE"], errors="coerce")
    weights["CREATION_DATE"] = pd.to_datetime(weights["CREATION_DATE"], errors="coerce")
    weights["DOB"] = pd.to_datetime(weights["DOB"], errors="coerce")
    weights["day"] = weights["OBSERVATION_DATE"].dt.floor("D")

    weights["RECORD_INPUT_TYPE"] = (
        weights["RECORD_INPUT_TYPE"].astype(str).str.strip().str.title()
    )
    weights["SEX"] = weights["SEX"].astype(str).str.strip().str.title()
    weights["NEUTERED"] = weights["NEUTERED"].map(lambda x: "True" if bool(x) else "False")

    weights["_creation_sort"] = weights["CREATION_DATE"].fillna(pd.Timestamp("1900-01-01"))
    weights["_record_sort"] = weights["RECORD_ID"].fillna("").astype(str)

    # Approved rule:
    # 1) latest OBSERVATION_DATE
    # 2) latest CREATION_DATE
    # 3) highest RECORD_ID
    weights = weights.sort_values(
        [
            "LIVESTOCK_ID",
            "day",
            "OBSERVATION_DATE",
            "_creation_sort",
            "_record_sort",
        ],
        kind="mergesort",
    ).drop_duplicates(["LIVESTOCK_ID", "day"], keep="last")

    weights = weights.drop(columns=["_creation_sort", "_record_sort"])
    return weights


def prepare_moves(moves: pd.DataFrame, paddock: pd.DataFrame, max_day: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    moves = moves.copy()
    paddock = paddock.copy()

    moves["START_DATE"] = pd.to_datetime(moves["START_DATE"], errors="coerce")
    moves["END_DATE"] = pd.to_datetime(moves["END_DATE"], errors="coerce")
    moves["start_day"] = moves["START_DATE"].dt.floor("D")
    moves["end_day"] = moves["END_DATE"].dt.floor("D")
    moves["end_day"] = moves["end_day"].fillna(max_day + pd.Timedelta(days=1))

    invalid = moves["end_day"] <= moves["start_day"]
    moves.loc[invalid, "end_day"] = moves.loc[invalid, "start_day"] + pd.Timedelta(days=1)

    paddock = paddock.rename(
        columns={
            "LAND_SIZE_HA": "paddock_land_size_ha",
            "ARABLE_LAND_SIZE_HA": "paddock_arable_land_size_ha",
            "PASTURE_STATE": "pasture_state",
        }
    )

    moves = moves.merge(
        paddock[["PADDOCK_ID", "pasture_state", "paddock_land_size_ha", "paddock_arable_land_size_ha"]],
        on="PADDOCK_ID",
        how="left",
    )

    moves["pasture_state"] = moves["pasture_state"].map(normalize_text_value)

    move_cols = [
        "LIVESTOCK_ID",
        "start_day",
        "end_day",
        "LOAD_MJ_PER_DAY",
        "PADDOCK_ID",
        "pasture_state",
        "paddock_land_size_ha",
        "paddock_arable_land_size_ha",
    ]
    moves = moves[move_cols].dropna(subset=["LIVESTOCK_ID", "start_day", "end_day"]) \
        .sort_values(["LIVESTOCK_ID", "start_day", "end_day", "PADDOCK_ID"])

    move_map: Dict[str, pd.DataFrame] = {
        livestock_id: grp.reset_index(drop=True)
        for livestock_id, grp in moves.groupby("LIVESTOCK_ID", sort=False)
    }
    return move_map


def build_category_vocab(weights: pd.DataFrame, paddock: pd.DataFrame) -> Dict[str, List[str]]:
    def uniq(series: pd.Series) -> List[str]:
        vals = sorted({normalize_text_value(v) for v in series})
        if "Unknown" not in vals:
            vals.append("Unknown")
        return vals

    vocab = {
        "record_input_type": uniq(weights["RECORD_INPUT_TYPE"]),
        "sex": uniq(weights["SEX"]),
        "neutered": uniq(weights["NEUTERED"]),
        "major_category": uniq(weights["MAJOR_CATEGORY"]),
        "age_class": uniq(weights["AGE_CLASS"]),
        "canon_breed": uniq(weights["CANON_BREED"]),
        "pasture_state": uniq(paddock["PASTURE_STATE"]),
    }
    return vocab


def assign_move_features(days_df: pd.DataFrame, animal_moves: pd.DataFrame | None) -> pd.DataFrame:
    out = days_df[["day"]].copy()
    out["load_mj_per_day"] = np.nan
    out["paddock_land_size_ha"] = np.nan
    out["paddock_arable_land_size_ha"] = np.nan
    out["pasture_state"] = "Unknown"
    out["active_move_flag"] = 0.0

    if animal_moves is None or animal_moves.empty:
        return out

    source = animal_moves.sort_values(["start_day", "end_day", "PADDOCK_ID"]).reset_index(drop=True)

    merged = pd.merge_asof(
        out[["day"]].sort_values("day"),
        source,
        left_on="day",
        right_on="start_day",
        direction="backward",
        allow_exact_matches=True,
    )

    active = merged["end_day"].isna() | (merged["day"] < merged["end_day"])

    out.loc[active, "load_mj_per_day"] = merged.loc[active, "LOAD_MJ_PER_DAY"].astype(float).values
    out.loc[active, "paddock_land_size_ha"] = merged.loc[active, "paddock_land_size_ha"].astype(float).values
    out.loc[active, "paddock_arable_land_size_ha"] = (
        merged.loc[active, "paddock_arable_land_size_ha"].astype(float).values
    )
    out.loc[active, "pasture_state"] = (
        merged.loc[active, "pasture_state"].map(normalize_text_value).values
    )
    out.loc[active, "active_move_flag"] = 1.0
    return out


def compute_class_label(adg: pd.Series, threshold: float) -> np.ndarray:
    y = np.full(len(adg), np.nan)
    y[adg < -threshold] = 0
    y[(adg >= -threshold) & (adg <= threshold)] = 1
    y[adg > threshold] = 2
    return y


def build_summary(window: pd.DataFrame, horizon_days: int, adg_value: float, adg_class: int) -> str:
    last = window.iloc[-1]
    first = window.iloc[0]

    rain_total = float(window["daily_rain_mm"].fillna(0).sum())
    temp_mean = float(window["amb_temp_c"].mean()) if window["amb_temp_c"].notna().any() else np.nan
    load_mean = float(window["load_mj_per_day"].mean()) if window["load_mj_per_day"].notna().any() else np.nan

    weight_start = float(first["weight_last_obs_kg"]) if pd.notna(first["weight_last_obs_kg"]) else np.nan
    weight_end = float(last["weight_last_obs_kg"]) if pd.notna(last["weight_last_obs_kg"]) else np.nan

    trend = "stable"
    if pd.notna(weight_start) and pd.notna(weight_end):
        if weight_end > weight_start + 1:
            trend = "upward"
        elif weight_end < weight_start - 1:
            trend = "downward"

    class_name = {0: "decrease", 1: "neutral", 2: "increase"}.get(int(adg_class), "unknown")

    parts = [
        f"Over the last {len(window)} days, the animal's recent weight trajectory was {trend}.",
        f"Cumulative rainfall in this window was {rain_total:.1f} mm and the average ambient temperature was {temp_mean:.1f} C." if pd.notna(temp_mean) else
        f"Cumulative rainfall in this window was {rain_total:.1f} mm.",
        f"Average paddock load during active move periods was {load_mean:.1f} MJ/day." if pd.notna(load_mean) else
        "No active paddock load signal was recorded in this window.",
        f"Current animal context: sex={last['sex']}, age_class={last['age_class']}, record_input_type={last['record_input_type']}.",
        f"Model target framing: next {horizon_days}-day ADG class is {class_name} (adg_7d={adg_value:.3f} kg/day).",
    ]
    return " ".join(parts)


def build_samples(
    weights: pd.DataFrame,
    weather: pd.DataFrame,
    move_map: Dict[str, pd.DataFrame],
    vocab: Dict[str, List[str]],
    history_days: int,
    horizon_days: int,
    neutral_threshold: float,
) -> PreparedSamples:
    weather_by_farm = {
        farm_id: grp.set_index("DATE")[list(WEATHER_RENAME.values())]
        for farm_id, grp in weather.groupby("FARM_ID", sort=False)
    }

    weights = weights.sort_values(["LIVESTOCK_ID", "day"]).reset_index(drop=True)

    x_time_list: List[np.ndarray] = []
    x_missing_list: List[np.ndarray] = []
    x_text_list: List[str] = []
    y_class_list: List[int] = []
    y_adg_list: List[float] = []
    meta_rows: List[dict] = []

    cat_feature_names: List[str] = []
    for col in CAT_FEATURES:
        cat_feature_names.extend([f"{col}__{v}" for v in vocab[col]])

    for livestock_id, grp in weights.groupby("LIVESTOCK_ID", sort=False):
        grp = grp.sort_values("day").reset_index(drop=True)
        first_day = grp["day"].min()
        last_day = grp["day"].max()
        if pd.isna(first_day) or pd.isna(last_day):
            continue

        valid_anchor_rows = grp[
            (grp["day"] >= first_day + pd.Timedelta(days=history_days - 1))
            & (grp["day"] <= last_day - pd.Timedelta(days=horizon_days))
        ]
        if valid_anchor_rows.empty:
            continue

        anchor_days_set = set(valid_anchor_rows["day"].tolist())

        days = pd.date_range(first_day, last_day, freq="D")
        panel = pd.DataFrame({"day": days})
        panel["LIVESTOCK_ID"] = livestock_id
        panel["FARM_ID"] = normalize_text_value(grp["FARM_ID"].iloc[0], default="")

        wx = weather_by_farm.get(panel["FARM_ID"].iloc[0])
        if wx is None:
            for c in WEATHER_RENAME.values():
                panel[c] = np.nan
        else:
            panel = panel.merge(
                wx.reset_index(),
                left_on="day",
                right_on="DATE",
                how="left",
            ).drop(columns=["DATE"])

        keep_weight_cols = [
            "day",
            "WEIGHT",
            "DOB",
            "AGE_MONTHS",
            "RECORD_INPUT_TYPE",
            "SEX",
            "CANON_BREED",
            "MAJOR_CATEGORY",
            "AGE_CLASS",
            "NEUTERED",
            "OBSERVATION_DATE",
            "CREATION_DATE",
            "RECORD_ID",
        ]
        panel = panel.merge(grp[keep_weight_cols], on="day", how="left")

        move_features = assign_move_features(panel[["day"]], move_map.get(livestock_id))
        panel = panel.merge(move_features, on="day", how="left", suffixes=("", "_move"))

        panel["weight_obs_kg"] = panel["WEIGHT"].astype(float)
        panel["weight_last_obs_kg"] = panel["weight_obs_kg"].ffill()

        obs_day = panel["day"].where(panel["weight_obs_kg"].notna())
        last_obs_day = obs_day.ffill()
        panel["days_since_last_weight"] = (panel["day"] - last_obs_day).dt.days.astype(float)

        dob_series = panel["DOB"].ffill().bfill()
        age_from_dob = (panel["day"] - dob_series).dt.days / 30.4375
        age_from_record = panel["AGE_MONTHS"].astype(float).ffill().bfill()
        panel["age_months"] = age_from_dob.where(age_from_dob.notna(), age_from_record)

        panel["record_input_type"] = (
            panel["RECORD_INPUT_TYPE"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        )
        panel["sex"] = panel["SEX"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        panel["canon_breed"] = (
            panel["CANON_BREED"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        )
        panel["major_category"] = (
            panel["MAJOR_CATEGORY"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        )
        panel["age_class"] = (
            panel["AGE_CLASS"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        )
        panel["neutered"] = (
            panel["NEUTERED"].map(normalize_text_value).ffill().bfill().fillna("Unknown")
        )
        panel["pasture_state"] = panel["pasture_state"].map(normalize_text_value).fillna("Unknown")

        panel["weight_interp_kg"] = panel["weight_obs_kg"].interpolate(method="linear", limit_area="inside")
        panel["weight_plus_h_kg"] = panel["weight_interp_kg"].shift(-horizon_days)
        panel["adg_7d"] = (panel["weight_plus_h_kg"] - panel["weight_interp_kg"]) / float(horizon_days)
        panel["adg_class"] = compute_class_label(panel["adg_7d"], neutral_threshold)

        numeric_df = panel[NUMERIC_FEATURES].astype(float).copy()
        numeric_missing = numeric_df.isna().to_numpy(dtype=np.float32)

        numeric_df = numeric_df.ffill()
        rolling_med = numeric_df.rolling(window=7, min_periods=1).median()
        numeric_df = numeric_df.fillna(rolling_med)

        cat_arrays = []
        for col in CAT_FEATURES:
            values = panel[col].map(normalize_text_value).to_numpy()
            for category in vocab[col]:
                cat_arrays.append((values == category).astype(np.float32)[:, None])

        cat_matrix = np.concatenate(cat_arrays, axis=1) if cat_arrays else np.empty((len(panel), 0), dtype=np.float32)
        feature_matrix = np.concatenate([numeric_df.to_numpy(dtype=np.float32), cat_matrix], axis=1)

        panel["panel_idx"] = np.arange(len(panel), dtype=np.int32)
        anchor_mask = (
            panel["day"].isin(anchor_days_set)
            & panel["adg_class"].notna()
            & (panel["panel_idx"] >= history_days - 1)
        )

        anchor_positions = panel.loc[anchor_mask, "panel_idx"].to_numpy(dtype=np.int32)
        if anchor_positions.size == 0:
            continue

        for pos in anchor_positions:
            start = pos - (history_days - 1)
            stop = pos + 1
            x_time_list.append(feature_matrix[start:stop])
            x_missing_list.append(numeric_missing[start:stop])

            adg_value = float(panel.iloc[pos]["adg_7d"])
            adg_class = int(panel.iloc[pos]["adg_class"])
            y_adg_list.append(adg_value)
            y_class_list.append(adg_class)

            window = panel.iloc[start:stop]
            x_text_list.append(build_summary(window, horizon_days, adg_value, adg_class))

            meta_rows.append(
                {
                    "farm_id": panel.iloc[pos]["FARM_ID"],
                    "livestock_id": livestock_id,
                    "anchor_day": panel.iloc[pos]["day"],
                    "record_input_type": panel.iloc[pos]["record_input_type"],
                }
            )

    if not x_time_list:
        raise RuntimeError("No samples were generated. Check date coverage and source tables.")

    x_time_raw = np.stack(x_time_list).astype(np.float32)
    x_missing = np.stack(x_missing_list).astype(np.float32)
    # Save as unicode array to avoid pickle/NumPy-version coupling.
    x_text = np.array(x_text_list, dtype=str)
    y_class = np.array(y_class_list, dtype=np.int64)
    y_adg = np.array(y_adg_list, dtype=np.float32)

    sample_meta = pd.DataFrame(meta_rows)
    sample_meta.insert(0, "sample_id", np.arange(len(sample_meta), dtype=np.int64))

    feature_names_base = NUMERIC_FEATURES + cat_feature_names

    return PreparedSamples(
        x_time_raw=x_time_raw,
        x_missing=x_missing,
        x_text=x_text,
        y_class=y_class,
        y_adg=y_adg,
        sample_meta=sample_meta,
        feature_names_base=feature_names_base,
        numeric_feature_names=NUMERIC_FEATURES,
    )


def chronological_split(sample_meta: pd.DataFrame, train_ratio: float, test_ratio: float) -> Dict[str, np.ndarray]:
    order = np.argsort(sample_meta["anchor_day"].to_numpy())
    n = len(order)
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_val = n - n_train - n_test

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    split_labels = np.array(["" for _ in range(n)], dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    sample_meta["split"] = split_labels

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def finalize_features(
    samples: PreparedSamples,
    split_idx: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    x_time = samples.x_time_raw.copy()

    train = split_idx["train"]
    train_flat = x_time[train].reshape(-1, x_time.shape[-1])
    medians = np.nanmedian(train_flat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians).astype(np.float32)

    nan_mask = np.isnan(x_time)
    if nan_mask.any():
        feature_idx = np.where(nan_mask)[2]
        x_time[nan_mask] = medians[feature_idx]

    x_time_final = np.concatenate([x_time, samples.x_missing], axis=2).astype(np.float32)
    missing_feature_names = [f"{c}__missing" for c in samples.numeric_feature_names]
    feature_names = samples.feature_names_base + missing_feature_names

    return x_time_final, feature_names, medians


def save_outputs(
    output_dir: Path,
    samples: PreparedSamples,
    x_time_final: np.ndarray,
    feature_names: List[str],
    split_idx: Dict[str, np.ndarray],
    medians: np.ndarray,
    args: argparse.Namespace,
    weights_before: int,
    weights_after: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "x_time.npy", x_time_final)
    np.save(output_dir / "x_text.npy", samples.x_text, allow_pickle=False)
    np.save(output_dir / "y_class.npy", samples.y_class)
    np.save(output_dir / "y_adg_7d.npy", samples.y_adg)

    samples.sample_meta.to_parquet(output_dir / "sample_meta.parquet", index=False)

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    split_payload = {
        "train": split_idx["train"].tolist(),
        "val": split_idx["val"].tolist(),
        "test": split_idx["test"].tolist(),
        "train_ratio": args.train_ratio,
        "val_ratio": 1.0 - args.train_ratio - args.test_ratio,
        "test_ratio": args.test_ratio,
    }
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)

    class_counts = pd.Series(samples.y_class).value_counts().sort_index().to_dict()

    report = {
        "num_samples": int(len(samples.y_class)),
        "num_features": int(x_time_final.shape[2]),
        "history_days": int(args.history_days),
        "horizon_days": int(args.horizon_days),
        "neutral_threshold": float(args.neutral_threshold),
        "dedup_rule": {
            "key": ["LIVESTOCK_ID", "day"],
            "priority": ["OBSERVATION_DATE desc", "CREATION_DATE desc", "RECORD_ID desc"],
        },
        "weights_rows_before_dedup": int(weights_before),
        "weights_rows_after_dedup": int(weights_after),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "split_sizes": {
            "train": int(len(split_idx["train"])),
            "val": int(len(split_idx["val"])),
            "test": int(len(split_idx["test"])),
        },
        "global_train_medians": {
            feature_names[i]: float(medians[i]) for i in range(len(medians))
        },
    }

    with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    args = parse_args()
    tables = load_tables(args.input_dir)

    paddock = tables["paddock"]
    weather = prepare_weather(tables["weather"])

    weights_raw = tables["weights"]
    weights_before = len(weights_raw)
    weights = deduplicate_weights(weights_raw)
    weights_after = len(weights)

    max_day = weights["day"].max()
    move_map = prepare_moves(tables["moves"], paddock, max_day=max_day)
    vocab = build_category_vocab(weights, paddock)

    samples = build_samples(
        weights=weights,
        weather=weather,
        move_map=move_map,
        vocab=vocab,
        history_days=args.history_days,
        horizon_days=args.horizon_days,
        neutral_threshold=args.neutral_threshold,
    )

    split_idx = chronological_split(samples.sample_meta, args.train_ratio, args.test_ratio)
    x_time_final, feature_names, medians = finalize_features(samples, split_idx)

    save_outputs(
        output_dir=args.output_dir,
        samples=samples,
        x_time_final=x_time_final,
        feature_names=feature_names,
        split_idx=split_idx,
        medians=medians,
        args=args,
        weights_before=weights_before,
        weights_after=weights_after,
    )

    print("Prepared AgriWebb dataset successfully.")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples: {len(samples.y_class)}")
    print(f"x_time shape: {x_time_final.shape}")


if __name__ == "__main__":
    main()
