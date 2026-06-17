"""
Evaluate matching difficulty after candidate detection.

This script assumes ground-truth landmarks have already been extracted with
landmark_ground_truth_tools.py. It does not change the model. It diagnoses the
matching problem by asking:

1. Is a detected candidate available near the true landmark on both images?
2. If yes, how highly does the true candidate pair rank by appearance alone?
3. How highly does it rank by leave-one-out ground-truth geometry?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import stable_landmark_labeler_v2 as labeler
from evaluate_candidate_recall import _build_mask


Point = Tuple[float, float]


@dataclass
class MatchCandidate:
    source: str
    center: Point
    score: float
    area: int
    mean_lab: Tuple[float, float, float]
    box_half: int


def _distance(p1: Point, p2: Point) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _mean_lab_patch(image: np.ndarray, center: Point, radius: int = 6) -> Tuple[float, float, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    cx, cy = int(round(center[0])), int(round(center[1]))
    y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius + 1)
    patch = lab[y0:y1, x0:x1]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in np.mean(patch.reshape(-1, 3), axis=0))


def _candidate_pools(
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
) -> List[MatchCandidate]:
    candidates: List[MatchCandidate] = []

    for mole in labeler.detect_mole_candidates(image, mask, max_moles, min_mole_score):
        candidates.append(
            MatchCandidate(
                source="mole",
                center=mole.center,
                score=mole.score,
                area=mole.area,
                mean_lab=mole.mean_lab,
                box_half=mole.box_half,
            )
        )

    for patch in labeler.detect_brown_patch_candidates(
        image,
        mask,
        max_brown_patches,
        brown_patch_min_area,
        brown_patch_max_area,
        brown_patch_min_score,
    ):
        candidates.append(
            MatchCandidate(
                source="brown_patch",
                center=patch.center,
                score=patch.score,
                area=patch.area,
                mean_lab=patch.mean_lab,
                box_half=patch.box_half,
            )
        )

    for spot in labeler.detect_stable_spot_keypoints(image, mask, stable_spot_keypoints, min_stable_spot_score):
        measurement = labeler.stable_spot_measurement(image, spot.pt)
        candidates.append(
            MatchCandidate(
                source="stable_spot",
                center=(float(spot.pt[0]), float(spot.pt[1])),
                score=float(spot.response),
                area=max(3, int(round((spot.size / 2.0) ** 2))),
                mean_lab=_mean_lab_patch(image, spot.pt),
                box_half=max(8, int(round(spot.size / 2.0))),
            )
        )

    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: Sequence[MatchCandidate], distance_threshold: float = 5.0) -> List[MatchCandidate]:
    output: List[MatchCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score * math.sqrt(max(1, item.area)), reverse=True):
        if any(_distance(candidate.center, existing.center) < distance_threshold for existing in output):
            continue
        output.append(candidate)
    return output


def _nearest_candidate(point: Point, candidates: Sequence[MatchCandidate]) -> Tuple[Optional[int], Optional[float]]:
    if not candidates:
        return None, None
    distances = [_distance(point, candidate.center) for candidate in candidates]
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _appearance_score(candidate1: MatchCandidate, candidate2: MatchCandidate) -> float:
    lab_distance = float(np.linalg.norm(np.array(candidate1.mean_lab) - np.array(candidate2.mean_lab)))
    area_ratio = max(candidate1.area, candidate2.area) / max(1, min(candidate1.area, candidate2.area))
    source_penalty = 0.0 if candidate1.source == candidate2.source else 0.25
    prominence_bonus = min(candidate1.score, candidate2.score) / 120.0
    return (lab_distance / 65.0) + (0.55 * abs(math.log(area_ratio))) + source_penalty - (0.15 * prominence_bonus)


def _normalized_position_score(
    candidate1: MatchCandidate,
    candidate2: MatchCandidate,
    image1_shape: Tuple[int, int, int],
    image2_shape: Tuple[int, int, int],
) -> float:
    h1, w1 = image1_shape[:2]
    h2, w2 = image2_shape[:2]
    p1 = (candidate1.center[0] / max(1, w1), candidate1.center[1] / max(1, h1))
    p2 = (candidate2.center[0] / max(1, w2), candidate2.center[1] / max(1, h2))
    return _distance(p1, p2)


def _rank(index: int, scores: Sequence[float]) -> int:
    target = scores[index]
    return 1 + sum(1 for score in scores if score < target)


def _leave_one_out_affine(
    landmarks: Sequence[Dict[str, object]],
    held_out_index: int,
) -> Optional[np.ndarray]:
    src = []
    dst = []
    for index, landmark in enumerate(landmarks):
        if index == held_out_index:
            continue
        src.append([float(landmark["image1_x"]), float(landmark["image1_y"])])
        dst.append([float(landmark["image2_x"]), float(landmark["image2_y"])])
    if len(src) < 3:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        return None
    return np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float32)


def _project(point: Point, matrix: np.ndarray) -> Point:
    projected = cv2.perspectiveTransform(np.float32([[point]]).reshape(-1, 1, 2), matrix).reshape(2)
    return float(projected[0]), float(projected[1])


def evaluate_matching(
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
        "landmarks": 0,
        "both_sides_detected": 0,
        "appearance_top1": 0,
        "appearance_top3": 0,
        "appearance_plus_position_top1": 0,
        "appearance_plus_position_top3": 0,
        "geometry_top1": 0,
        "geometry_top3": 0,
    }

    for pair in data["pairs"]:
        pair_name = str(pair["pair"])
        image1 = labeler.load_image(Path(pair["image1"]))
        image2 = labeler.load_image(Path(pair["image2"]))
        mask1 = _build_mask(image1, auto_ring)
        mask2 = _build_mask(image2, auto_ring)
        candidates1 = _candidate_pools(
            image1,
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
        candidates2 = _candidate_pools(
            image2,
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

        pair_summary = {
            "pair": pair_name,
            "landmarks": len(pair["landmarks"]),
            "candidate_count_image1": len(candidates1),
            "candidate_count_image2": len(candidates2),
            "both_sides_detected": 0,
            "appearance_top3": 0,
            "geometry_top3": 0,
        }

        for landmark_index, landmark in enumerate(pair["landmarks"]):
            totals["landmarks"] += 1
            truth1 = (float(landmark["image1_x"]), float(landmark["image1_y"]))
            truth2 = (float(landmark["image2_x"]), float(landmark["image2_y"]))
            idx1, dist1 = _nearest_candidate(truth1, candidates1)
            idx2, dist2 = _nearest_candidate(truth2, candidates2)
            both_detected = (
                idx1 is not None
                and idx2 is not None
                and dist1 is not None
                and dist2 is not None
                and dist1 <= tolerance
                and dist2 <= tolerance
            )

            row = {
                "pair": pair_name,
                "label": landmark["label"],
                "both_sides_detected": both_detected,
                "image1_detection_distance": None if dist1 is None else round(dist1, 2),
                "image2_detection_distance": None if dist2 is None else round(dist2, 2),
                "image1_candidate_source": None if idx1 is None else candidates1[idx1].source,
                "image2_candidate_source": None if idx2 is None else candidates2[idx2].source,
                "appearance_rank": None,
                "appearance_plus_position_rank": None,
                "geometry_rank": None,
                "geometry_projected_distance": None,
            }

            if both_detected:
                totals["both_sides_detected"] += 1
                pair_summary["both_sides_detected"] += 1
                candidate1 = candidates1[idx1]
                appearance_scores = [_appearance_score(candidate1, candidate2) for candidate2 in candidates2]
                appearance_rank = _rank(idx2, appearance_scores)
                position_scores = [
                    _normalized_position_score(candidate1, candidate2, image1.shape, image2.shape)
                    for candidate2 in candidates2
                ]
                combined_scores = [
                    appearance_scores[index] + (1.65 * position_scores[index])
                    for index in range(len(candidates2))
                ]
                combined_rank = _rank(idx2, combined_scores)

                row["appearance_rank"] = appearance_rank
                row["appearance_plus_position_rank"] = combined_rank
                if appearance_rank == 1:
                    totals["appearance_top1"] += 1
                if appearance_rank <= 3:
                    totals["appearance_top3"] += 1
                    pair_summary["appearance_top3"] += 1
                if combined_rank == 1:
                    totals["appearance_plus_position_top1"] += 1
                if combined_rank <= 3:
                    totals["appearance_plus_position_top3"] += 1

                affine = _leave_one_out_affine(pair["landmarks"], landmark_index)
                if affine is not None:
                    projected = _project(candidate1.center, affine)
                    geometry_scores = [_distance(projected, candidate2.center) for candidate2 in candidates2]
                    geometry_rank = _rank(idx2, geometry_scores)
                    row["geometry_rank"] = geometry_rank
                    row["geometry_projected_distance"] = round(geometry_scores[idx2], 2)
                    if geometry_rank == 1:
                        totals["geometry_top1"] += 1
                    if geometry_rank <= 3:
                        totals["geometry_top3"] += 1
                        pair_summary["geometry_top3"] += 1

            rows.append(row)

        pair_summaries.append(pair_summary)

    report = {
        "tolerance": tolerance,
        "totals": totals,
        "pairs": pair_summaries,
        "rows": rows,
    }
    (output_dir / "candidate_matching_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output_dir / "candidate_matching_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose candidate-to-candidate matching difficulty.")
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

    report = evaluate_matching(
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
    print(f"Both sides detected: {totals['both_sides_detected']}/{totals['landmarks']}")
    print(f"Appearance top-1/top-3: {totals['appearance_top1']}/{totals['appearance_top3']}")
    print(
        "Appearance+position top-1/top-3: "
        f"{totals['appearance_plus_position_top1']}/{totals['appearance_plus_position_top3']}"
    )
    print(f"Leave-one-out geometry top-1/top-3: {totals['geometry_top1']}/{totals['geometry_top3']}")
    print(f"Saved report: {args.output_dir / 'candidate_matching_report.json'}")
    print(f"Saved CSV: {args.output_dir / 'candidate_matching_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
