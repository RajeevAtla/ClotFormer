#!/usr/bin/env python3

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------
# Hardcoded paths / settings
# ----------------------------

CSV_PATH = Path("train.csv")
ORIGINAL_IMAGE_DIR = Path("train_original")

TRAIN_DIR = Path("train")
VAL_DIR = Path("val")
TEST_DIR = Path("test")

PATIENT_COL = "patient_id"
LABEL_COL = "label"
IMAGE_ID_COL = "image_id"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = [
    ".tif",
    ".tiff",
    ".svs",
    ".ndpi",
    ".mrxs",
    ".png",
    ".jpg",
    ".jpeg",
]


def check_required_columns(df: pd.DataFrame) -> None:
    required_cols = [IMAGE_ID_COL, PATIENT_COL, LABEL_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )


def check_one_label_per_patient(df: pd.DataFrame) -> None:
    label_counts = df.groupby(PATIENT_COL)[LABEL_COL].nunique(dropna=False)
    bad_patients = label_counts[label_counts > 1]

    if not bad_patients.empty:
        examples = bad_patients.index[:10].tolist()
        raise ValueError(
            "Some patients have more than one label. "
            "Resolve this before splitting. "
            f"Example patient IDs: {examples}"
        )


def make_patient_level_table(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = (
        df.groupby(PATIENT_COL, as_index=False)
        .agg(
            label=(LABEL_COL, "first"),
            n_images=(IMAGE_ID_COL, "count"),
        )
        .rename(columns={"label": LABEL_COL})
    )

    return patient_df


def assign_patient_splits(patient_df: pd.DataFrame) -> dict[str, str]:
    """
    Creates an 80/10/10 patient-level split.

    Step 1:
        80% train
        20% temporary

    Step 2:
        Split the temporary 20% into:
        10% validation
        10% test

    Stratification is done by patient label.
    """

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=patient_df[LABEL_COL],
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=temp_patients[LABEL_COL],
    )

    split_map: dict[str, str] = {}

    for patient_id in train_patients[PATIENT_COL]:
        split_map[str(patient_id)] = "train"

    for patient_id in val_patients[PATIENT_COL]:
        split_map[str(patient_id)] = "val"

    for patient_id in test_patients[PATIENT_COL]:
        split_map[str(patient_id)] = "test"

    return split_map


def verify_no_patient_leakage(df: pd.DataFrame) -> None:
    split_counts = df.groupby(PATIENT_COL)["split"].nunique(dropna=False)
    leaked_patients = split_counts[split_counts > 1]

    if not leaked_patients.empty:
        examples = leaked_patients.index[:10].tolist()
        raise RuntimeError(
            f"Patient leakage detected. Example patient IDs: {examples}"
        )


def find_image_file(image_id: str) -> Path:
    """
    Finds the WSI file inside train_original.

    This supports two cases:
    1. image_id already includes an extension
       Example: 006388_0.tif

    2. image_id does not include an extension
       Example: 006388_0
       The script tries common WSI/image extensions.
    """

    image_id = str(image_id)

    direct_path = ORIGINAL_IMAGE_DIR / image_id
    if direct_path.exists():
        return direct_path

    for ext in IMAGE_EXTENSIONS:
        candidate_path = ORIGINAL_IMAGE_DIR / f"{image_id}{ext}"
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find image for image_id={image_id} in {ORIGINAL_IMAGE_DIR}"
    )


def copy_split_images(df: pd.DataFrame) -> None:
    split_to_dir = {
        "train": TRAIN_DIR,
        "val": VAL_DIR,
        "test": TEST_DIR,
    }

    for split_dir in split_to_dir.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    missing_images: list[str] = []

    for _, row in df.iterrows():
        image_id = str(row[IMAGE_ID_COL])
        split = row["split"]
        output_dir = split_to_dir[split]

        try:
            source_path = find_image_file(image_id)
        except FileNotFoundError:
            missing_images.append(image_id)
            continue

        destination_path = output_dir / source_path.name

        if destination_path.exists():
            continue

        shutil.copy2(source_path, destination_path)

    if missing_images:
        preview = missing_images[:20]
        raise FileNotFoundError(
            f"{len(missing_images)} images were missing from {ORIGINAL_IMAGE_DIR}. "
            f"First few missing image_ids: {preview}"
        )


def save_split_csvs(df: pd.DataFrame) -> None:
    df.to_csv("train_with_patient_splits.csv", index=False)

    df[df["split"] == "train"].drop(columns=["split"]).to_csv(
        "train_split.csv",
        index=False,
    )

    df[df["split"] == "val"].drop(columns=["split"]).to_csv(
        "val_split.csv",
        index=False,
    )

    df[df["split"] == "test"].drop(columns=["split"]).to_csv(
        "test_split.csv",
        index=False,
    )


def print_split_stats(df: pd.DataFrame) -> None:
    print("\nImage / WSI counts by split:")
    print(df["split"].value_counts().reindex(["train", "val", "test"]))

    print("\nPatient counts by split:")
    print(
        df.groupby("split")[PATIENT_COL]
        .nunique()
        .reindex(["train", "val", "test"])
    )

    print("\nImage / WSI label counts by split:")
    print(pd.crosstab(df["split"], df[LABEL_COL]).reindex(["train", "val", "test"]))

    print("\nImage / WSI label proportions by split:")
    print(
        pd.crosstab(df["split"], df[LABEL_COL], normalize="index")
        .reindex(["train", "val", "test"])
        .round(4)
    )


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    if not ORIGINAL_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Could not find folder {ORIGINAL_IMAGE_DIR}")

    df = pd.read_csv(CSV_PATH)

    check_required_columns(df)

    df[PATIENT_COL] = df[PATIENT_COL].astype(str)
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].astype(str)

    check_one_label_per_patient(df)

    patient_df = make_patient_level_table(df)
    split_map = assign_patient_splits(patient_df)

    df["split"] = df[PATIENT_COL].map(split_map)

    if df["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a split.")

    verify_no_patient_leakage(df)

    save_split_csvs(df)
    copy_split_images(df)

    print_split_stats(df)

    print("\nDone.")
    print(f"Created folders: {TRAIN_DIR}, {VAL_DIR}, {TEST_DIR}")
    print("Created CSVs:")
    print("  train_with_patient_splits.csv")
    print("  train_split.csv")
    print("  val_split.csv")
    print("  test_split.csv")


if __name__ == "__main__":
    main()