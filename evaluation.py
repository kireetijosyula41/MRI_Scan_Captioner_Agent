"""Evaluate the PyTorch MRI classifier on braintumordata/Testing.

Run ``python evaluation.py`` to write metrics.json, confusion_matrix.png,
and predictions.csv in the project directory.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torch_classifier import (
    load_classifier,
    predict_probabilities_batch,
    preprocess_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_FOLDERS = (
    ("glioma", "Glioma"),
    ("meningioma", "Meningioma"),
    ("notumor", "No tumor"),
    ("pituitary", "Pituitary"),
)
LABELS = [label for _, label in CLASS_FOLDERS]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_test_images(test_dir):
    images = []
    for class_index, (folder_name, _) in enumerate(CLASS_FOLDERS):
        folder = test_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing test class folder: {folder}")
        images.extend(
            (path, class_index)
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        raise ValueError(f"No MRI images found in {test_dir}")
    return images


def calculate_metrics(true_indices, predicted_indices):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (true_indices, predicted_indices), 1)

    true_positives = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    precision = np.divide(
        true_positives, predicted_count,
        out=np.zeros(len(LABELS), dtype=float), where=predicted_count != 0,
    )
    recall = np.divide(
        true_positives, support,
        out=np.zeros(len(LABELS), dtype=float), where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(len(LABELS), dtype=float), where=(precision + recall) != 0,
    )

    total = int(support.sum())
    metrics = {
        "total_images": total,
        "labels": LABELS,
        "accuracy": float(true_positives.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, matrix


def save_confusion_matrix(matrix, output_path):
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Images")
    axis.set(
        title="MRI classifier confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def display_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def evaluate(model_path, test_dir, output_dir, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    images = find_test_images(test_dir)
    model = load_classifier(model_path)
    true_indices = []
    predicted_indices = []
    rows = []

    for start in range(0, len(images), batch_size):
        batch_items = images[start:start + batch_size]
        batch = np.concatenate(
            [preprocess_image(path) for path, _ in batch_items], axis=0
        )
        probabilities = predict_probabilities_batch(model, batch)
        for (path, true_index), distribution in zip(batch_items, probabilities):
            predicted_index = int(np.argmax(distribution))
            true_indices.append(true_index)
            predicted_indices.append(predicted_index)
            rows.append({
                "image_path": display_path(path),
                "true_label": LABELS[true_index],
                "predicted_label": LABELS[predicted_index],
                "confidence": float(distribution[predicted_index]),
            })
        print(f"Evaluated {min(start + batch_size, len(images))}/{len(images)} images", end="\r", flush=True)

    metrics, matrix = calculate_metrics(true_indices, predicted_indices)
    metrics["model_path"] = display_path(model_path)
    metrics["test_directory"] = display_path(test_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "true_label", "predicted_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}; macro F1: {metrics['macro_f1']:.4f}; weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved metrics.json, predictions.csv, and confusion_matrix.png to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "brain_tumor.h5")
    parser.add_argument("--testing-dir", type=Path, default=PROJECT_ROOT / "braintumordata/Testing")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate(args.model, args.testing_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
