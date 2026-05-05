# main.py

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ----------------------------
# Config
# ----------------------------

DATA_DIR = Path("data")

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = Path("swinv2-wsi-classifier")

MODEL_NAME = "microsoft/swinv2-tiny-patch4-window8-256"

SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

NUM_WORKERS = 4

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ----------------------------
# Dataset
# ----------------------------

class ImageFolderDataset(Dataset):
    """
    A simple image classification dataset.

    Expected structure:

    data/train/class_0/image1.png
    data/train/class_1/image2.png

    data/val/class_0/image3.png
    data/val/class_1/image4.png

    data/test/class_0/image5.png
    data/test/class_1/image6.png
    """

    def __init__(
        self,
        split_dir: Path,
        processor: AutoImageProcessor,
        label2id: dict[str, int],
    ) -> None:
        self.split_dir = split_dir
        self.processor = processor
        self.label2id = label2id

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Could not find split folder: {self.split_dir}")

        self.samples: list[tuple[Path, int]] = self._load_samples()

        if not self.samples:
            raise ValueError(f"No image files found in {self.split_dir}")

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        for class_name, label_id in self.label2id.items():
            class_dir = self.split_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class folder missing: {class_dir}"
                )

            for image_path in class_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, label_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image_path, label_id = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        processed = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# ----------------------------
# Helpers
# ----------------------------

def get_class_names(train_dir: Path) -> list[str]:
    if not train_dir.exists():
        raise FileNotFoundError(f"Could not find train folder: {train_dir}")

    class_names = sorted(
        path.name
        for path in train_dir.iterdir()
        if path.is_dir()
    )

    if not class_names:
        raise ValueError(
            f"No class folders found inside {train_dir}. "
            "Expected something like data/train/benign and data/train/malignant."
        )

    return class_names


def verify_split_classes(
    split_dir: Path,
    expected_class_names: list[str],
) -> None:
    found_class_names = sorted(
        path.name
        for path in split_dir.iterdir()
        if path.is_dir()
    )

    missing = sorted(set(expected_class_names) - set(found_class_names))
    extra = sorted(set(found_class_names) - set(expected_class_names))

    if missing:
        raise ValueError(
            f"{split_dir} is missing these class folders: {missing}"
        )

    if extra:
        raise ValueError(
            f"{split_dir} has extra class folders not present in train: {extra}"
        )


def data_collator(
    examples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits = eval_pred.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def make_training_args() -> TrainingArguments:
    """
    Handles both newer and older transformers versions.

    Newer versions use eval_strategy.
    Older versions use evaluation_strategy.
    """

    common_args = {
        "output_dir": str(OUTPUT_DIR),
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "dataloader_num_workers": NUM_WORKERS,
        "report_to": "none",
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        common_args["eval_strategy"] = "epoch"
    else:
        common_args["evaluation_strategy"] = "epoch"

    return TrainingArguments(**common_args)


def save_metrics(
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(TRAIN_DIR)

    verify_split_classes(VAL_DIR, class_names)
    verify_split_classes(TEST_DIR, class_names)

    label2id = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    id2label = {
        idx: class_name
        for class_name, idx in label2id.items()
    }

    print("Classes:")
    for class_name, label_id in label2id.items():
        print(f"  {label_id}: {class_name}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_dataset = ImageFolderDataset(
        split_dir=TRAIN_DIR,
        processor=processor,
        label2id=label2id,
    )

    val_dataset = ImageFolderDataset(
        split_dir=VAL_DIR,
        processor=processor,
        label2id=label2id,
    )

    test_dataset = ImageFolderDataset(
        split_dir=TEST_DIR,
        processor=processor,
        label2id=label2id,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = make_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(val_metrics)

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    final_model_dir = OUTPUT_DIR / "final_model"

    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    save_metrics(
        metrics=val_metrics,
        output_path=OUTPUT_DIR / "val_metrics.json",
    )

    save_metrics(
        metrics=test_metrics,
        output_path=OUTPUT_DIR / "test_metrics.json",
    )

    print("\nDone.")
    print(f"Best/final model saved to: {final_model_dir}")
    print(f"Validation metrics saved to: {OUTPUT_DIR / 'val_metrics.json'}")
    print(f"Test metrics saved to:       {OUTPUT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()