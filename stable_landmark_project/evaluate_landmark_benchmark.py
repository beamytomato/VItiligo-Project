"""
Run the landmark labeler on a small labeled benchmark and compare predictions
against extracted ground-truth landmark centers.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


Point = Tuple[float, float]


def _distance(p1: Point, p2: Point) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _read_prediction_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "landmark_id": float(row["landmark_id"]),
                        "image1_x": float(row["image1_x"]),
                        "image1_y": float(row["image1_y"]),
                        "image2_x": float(row["image2_x"]),
                        "image2_y": float(row["image2_y"]),
                    }
                )
            except (KeyError, ValueError):
                continue
        return rows


def evaluate_pair(ground_truth: Sequence[Dict[str, object]], predictions: Sequence[Dict[str, float]], tolerance: float):
    matched_prediction_indexes = set()
    matched_truth = []
    missed_truth = []

    for truth in ground_truth:
        truth_p1 = (float(truth["image1_x"]), float(truth["image1_y"]))
        truth_p2 = (float(truth["image2_x"]), float(truth["image2_y"]))
        best = None
        for index, prediction in enumerate(predictions):
            if index in matched_prediction_indexes:
                continue
            pred_p1 = (prediction["image1_x"], prediction["image1_y"])
            pred_p2 = (prediction["image2_x"], prediction["image2_y"])
            error1 = _distance(truth_p1, pred_p1)
            error2 = _distance(truth_p2, pred_p2)
            max_error = max(error1, error2)
            if max_error <= tolerance and (best is None or max_error < best[0]):
                best = (max_error, index, error1, error2)

        if best is None:
            missed_truth.append(truth)
        else:
            _max_error, index, error1, error2 = best
            matched_prediction_indexes.add(index)
            matched_truth.append(
                {
                    "label": truth["label"],
                    "prediction_id": int(predictions[index]["landmark_id"]),
                    "image1_error": round(error1, 2),
                    "image2_error": round(error2, 2),
                }
            )

    false_predictions = [
        predictions[index]
        for index in range(len(predictions))
        if index not in matched_prediction_indexes
    ]
    return matched_truth, missed_truth, false_predictions


def run_labeler(
    script_path: Path,
    image1: Path,
    image2: Path,
    output_dir: Path,
    extra_args: Sequence[str],
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--image1",
        str(image1),
        "--image2",
        str(image2),
        "--output-dir",
        str(output_dir),
        "--save-debug",
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark predicted landmarks against extracted ground truth.")
    parser.add_argument("--ground-truth-json", required=True, type=Path)
    parser.add_argument("--labeler", default=Path(__file__).with_name("stable_landmark_labeler_v2.py"), type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tolerance", default=45.0, type=float, help="Pixel tolerance in both images")
    parser.add_argument(
        "--labeler-arg",
        action="append",
        default=[],
        help="Additional argument passed through to the labeler; repeat for multiple args",
    )
    args = parser.parse_args()

    data = json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for pair in data["pairs"]:
        pair_name = pair["pair"]
        safe_pair_name = pair_name.replace(" ", "_").lower()
        output_dir = args.output_root / safe_pair_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {pair_name} ===")
        run_labeler(
            args.labeler,
            Path(pair["image1"]),
            Path(pair["image2"]),
            output_dir,
            args.labeler_arg,
        )

        predictions = _read_prediction_csv(output_dir / "landmark_matches.csv")
        matched, missed, false_predictions = evaluate_pair(pair["landmarks"], predictions, args.tolerance)
        print(
            f"matched {len(matched)}/{len(pair['landmarks'])}; "
            f"predictions={len(predictions)}; false_predictions={len(false_predictions)}"
        )
        if missed:
            print("missed:", ", ".join(str(item["label"]) for item in missed))

        summary.append(
            {
                "pair": pair_name,
                "ground_truth": len(pair["landmarks"]),
                "predictions": len(predictions),
                "matched": len(matched),
                "missed": len(missed),
                "false_predictions": len(false_predictions),
                "matched_details": matched,
                "missed_labels": [item["label"] for item in missed],
            }
        )

    total_truth = sum(item["ground_truth"] for item in summary)
    total_matched = sum(item["matched"] for item in summary)
    total_predictions = sum(item["predictions"] for item in summary)
    total_false = sum(item["false_predictions"] for item in summary)
    report = {
        "tolerance": args.tolerance,
        "total_ground_truth": total_truth,
        "total_predictions": total_predictions,
        "total_matched": total_matched,
        "total_missed": total_truth - total_matched,
        "total_false_predictions": total_false,
        "pairs": summary,
    }
    report_path = args.output_root / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"matched {total_matched}/{total_truth}")
    print(f"predictions={total_predictions}; false_predictions={total_false}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
