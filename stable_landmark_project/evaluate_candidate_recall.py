"""
Evaluate whether hand-labeled ground-truth landmarks are detected as candidates.

This does not evaluate final matching. It answers the earlier question:
    "Did the model find the true landmark at all?"

Candidate pools checked:
    - compact mole/freckle candidates
    - broader brown patch candidates
    - stable-spot keypoints used to augment SIFT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

import stable_landmark_labeler_v2 as labeler


Point = Tuple[float, float]


def _distance(p1: Point, p2: Point) -> float:
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def _build_mask(image: np.ndarray, auto_ring: bool) -> np.ndarray:
    h, w = image.shape[:2]
    roi_mask = np.full((h, w), 255, dtype=np.uint8)
    mask = labeler.build_automatic_quality_mask(image, roi_mask)
    if auto_ring:
        ring = labeler.detect_largest_ring_circle(image)
        if ring is not None:
            center, radius = ring
            mask, _inner = labeler.suppress_ring_annulus_from_mask(mask, center, radius)
    return mask


def _candidate_points_for_image(
    image: np.ndarray,
    mask: np.ndarray,
    max_moles: int,
    min_mole_score: float,
    max_brown_patches: int,
    brown_patch_min_area: int,
    brown_patch_max_area: int,
    brown_patch_min_score: float,
    stable_spot_keypoints: int,
    min_stable_spot_score: float,
) -> Dict[str, List[Dict[str, object]]]:
    mole_candidates = labeler.detect_mole_candidates(image, mask, max_moles, min_mole_score)
    brown_candidates = labeler.detect_brown_patch_candidates(
        image,
        mask,
        max_brown_patches,
        brown_patch_min_area,
        brown_patch_max_area,
        brown_patch_min_score,
    )
    stable_spots = labeler.detect_stable_spot_keypoints(
        image,
        mask,
        stable_spot_keypoints,
        min_stable_spot_score,
    )

    return {
        "mole": [
            {
                "x": candidate.center[0],
                "y": candidate.center[1],
                "score": candidate.score,
                "box_half": candidate.box_half,
            }
            for candidate in mole_candidates
        ],
        "brown_patch": [
            {
                "x": candidate.center[0],
                "y": candidate.center[1],
                "score": candidate.score,
                "box_half": candidate.box_half,
            }
            for candidate in brown_candidates
        ],
        "stable_spot": [
            {
                "x": spot.pt[0],
                "y": spot.pt[1],
                "score": float(spot.response),
                "box_half": max(8, int(round(spot.size / 2.0))),
            }
            for spot in stable_spots
        ],
    }


def _nearest_candidate(point: Point, candidates: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not candidates:
        return {
            "distance": None,
            "x": None,
            "y": None,
            "score": None,
        }
    best = min(candidates, key=lambda item: _distance(point, (float(item["x"]), float(item["y"]))))
    return {
        "distance": round(_distance(point, (float(best["x"]), float(best["y"]))), 2),
        "x": round(float(best["x"]), 2),
        "y": round(float(best["y"]), 2),
        "score": round(float(best["score"]), 3),
    }


def _draw_recall_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    truths: Sequence[Dict[str, object]],
    truth_side: str,
    candidates_by_type: Dict[str, List[Dict[str, object]]],
    tolerance: float,
) -> np.ndarray:
    output = image.copy()
    dim = output.copy()
    dim[mask == 0] = (0, 0, 80)
    output = cv2.addWeighted(dim, 0.25, output, 0.75, 0)

    colors = {
        "mole": (0, 255, 255),
        "brown_patch": (255, 0, 255),
        "stable_spot": (0, 180, 255),
    }
    for candidate_type, candidates in candidates_by_type.items():
        color = colors[candidate_type]
        for candidate in candidates:
            center = (int(round(float(candidate["x"]))), int(round(float(candidate["y"]))))
            cv2.circle(output, center, 6, color, 1, cv2.LINE_AA)

    for truth in truths:
        x = float(truth[f"{truth_side}_x"])
        y = float(truth[f"{truth_side}_y"])
        point = (x, y)
        found = any(
            (
                nearest["distance"] is not None
                and float(nearest["distance"]) <= tolerance
            )
            for nearest in (_nearest_candidate(point, candidates) for candidates in candidates_by_type.values())
        )
        color = (0, 255, 0) if found else (0, 0, 255)
        center = (int(round(x)), int(round(y)))
        cv2.rectangle(output, (center[0] - 24, center[1] - 24), (center[0] + 24, center[1] + 24), color, 3, cv2.LINE_AA)
        cv2.putText(
            output,
            str(truth["label"]),
            (center[0] + 28, max(18, center[1] - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            str(truth["label"]),
            (center[0] + 28, max(18, center[1] - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def evaluate_candidate_recall(
    ground_truth_json: Path,
    output_dir: Path,
    tolerance: float,
    auto_ring: bool,
    max_moles: int,
    min_mole_score: float,
    max_brown_patches: int,
    brown_patch_min_area: int,
    brown_patch_max_area: int,
    brown_patch_min_score: float,
    stable_spot_keypoints: int,
    min_stable_spot_score: float,
) -> Dict[str, object]:
    data = json.loads(ground_truth_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pair_summaries = []
    totals = {
        "truth_points": 0,
        "any_detected": 0,
        "mole_detected": 0,
        "brown_patch_detected": 0,
        "stable_spot_detected": 0,
    }

    for pair in data["pairs"]:
        pair_name = str(pair["pair"])
        pair_dir = output_dir / pair_name.replace(" ", "_").lower()
        pair_dir.mkdir(parents=True, exist_ok=True)

        img1 = labeler.load_image(Path(pair["image1"]))
        img2 = labeler.load_image(Path(pair["image2"]))
        mask1 = _build_mask(img1, auto_ring)
        mask2 = _build_mask(img2, auto_ring)
        cv2.imwrite(str(pair_dir / "image1_allowed_mask.png"), mask1)
        cv2.imwrite(str(pair_dir / "image2_allowed_mask.png"), mask2)

        candidates1 = _candidate_points_for_image(
            img1,
            mask1,
            max_moles,
            min_mole_score,
            max_brown_patches,
            brown_patch_min_area,
            brown_patch_max_area,
            brown_patch_min_score,
            stable_spot_keypoints,
            min_stable_spot_score,
        )
        candidates2 = _candidate_points_for_image(
            img2,
            mask2,
            max_moles,
            min_mole_score,
            max_brown_patches,
            brown_patch_min_area,
            brown_patch_max_area,
            brown_patch_min_score,
            stable_spot_keypoints,
            min_stable_spot_score,
        )

        side_counts = {
            "image1": {candidate_type: len(candidates) for candidate_type, candidates in candidates1.items()},
            "image2": {candidate_type: len(candidates) for candidate_type, candidates in candidates2.items()},
        }
        pair_detected = 0
        pair_truth = 0

        for landmark in pair["landmarks"]:
            for side, candidates_by_type in [("image1", candidates1), ("image2", candidates2)]:
                point = (float(landmark[f"{side}_x"]), float(landmark[f"{side}_y"]))
                nearest_by_type = {
                    candidate_type: _nearest_candidate(point, candidates)
                    for candidate_type, candidates in candidates_by_type.items()
                }
                detected_by_type = {
                    candidate_type: (
                        nearest["distance"] is not None
                        and float(nearest["distance"]) <= tolerance
                    )
                    for candidate_type, nearest in nearest_by_type.items()
                }
                any_detected = any(detected_by_type.values())

                totals["truth_points"] += 1
                pair_truth += 1
                if any_detected:
                    totals["any_detected"] += 1
                    pair_detected += 1
                for candidate_type, detected in detected_by_type.items():
                    if detected:
                        totals[f"{candidate_type}_detected"] += 1

                row = {
                    "pair": pair_name,
                    "label": landmark["label"],
                    "side": side,
                    "truth_x": round(point[0], 2),
                    "truth_y": round(point[1], 2),
                    "any_detected": any_detected,
                }
                for candidate_type, nearest in nearest_by_type.items():
                    row[f"{candidate_type}_detected"] = detected_by_type[candidate_type]
                    row[f"{candidate_type}_distance"] = nearest["distance"]
                    row[f"{candidate_type}_x"] = nearest["x"]
                    row[f"{candidate_type}_y"] = nearest["y"]
                    row[f"{candidate_type}_score"] = nearest["score"]
                rows.append(row)

        cv2.imwrite(
            str(pair_dir / "image1_candidate_recall_overlay.jpg"),
            _draw_recall_overlay(img1, mask1, pair["landmarks"], "image1", candidates1, tolerance),
        )
        cv2.imwrite(
            str(pair_dir / "image2_candidate_recall_overlay.jpg"),
            _draw_recall_overlay(img2, mask2, pair["landmarks"], "image2", candidates2, tolerance),
        )

        pair_summaries.append(
            {
                "pair": pair_name,
                "truth_points": pair_truth,
                "any_detected": pair_detected,
                "candidate_counts": side_counts,
            }
        )

    report = {
        "tolerance": tolerance,
        "totals": totals,
        "pairs": pair_summaries,
        "rows": rows,
    }
    (output_dir / "candidate_recall_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_dir / "candidate_recall_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure whether ground-truth landmarks are detected as candidates.")
    parser.add_argument("--ground-truth-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tolerance", default=45.0, type=float)
    parser.add_argument("--auto-detect-baseline-ring", action="store_true")
    parser.add_argument("--mole-candidates", default=80, type=int)
    parser.add_argument("--min-mole-score", default=16.0, type=float)
    parser.add_argument("--brown-patch-keypoints", default=160, type=int)
    parser.add_argument("--brown-patch-min-area", default=12, type=int)
    parser.add_argument("--brown-patch-max-area", default=12000, type=int)
    parser.add_argument("--brown-patch-min-score", default=7.0, type=float)
    parser.add_argument("--stable-spot-keypoints", default=160, type=int)
    parser.add_argument("--min-stable-spot-score", default=16.0, type=float)
    args = parser.parse_args()

    report = evaluate_candidate_recall(
        args.ground_truth_json,
        args.output_dir,
        args.tolerance,
        args.auto_detect_baseline_ring,
        args.mole_candidates,
        args.min_mole_score,
        args.brown_patch_keypoints,
        args.brown_patch_min_area,
        args.brown_patch_max_area,
        args.brown_patch_min_score,
        args.stable_spot_keypoints,
        args.min_stable_spot_score,
    )
    totals = report["totals"]
    print(
        "Candidate recall: "
        f"{totals['any_detected']}/{totals['truth_points']} truth points detected by at least one candidate pool"
    )
    print(f"Saved report: {args.output_dir / 'candidate_recall_report.json'}")
    print(f"Saved CSV: {args.output_dir / 'candidate_recall_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
