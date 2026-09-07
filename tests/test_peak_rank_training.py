import numpy as np

from research.peak_rank_detection.train_synthetic_real_detector import (
    TrainingExample,
    audit_passed,
    augment_example,
    detection_average_precision,
    local_positive_metrics,
    normalize_frames,
    selection_passed,
)


def test_normalization_is_robust_and_finite() -> None:
    frames = np.arange(3 * 8 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8, 8)
    normalized = normalize_frames(frames)

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_augmentation_keeps_points_aligned_and_in_bounds() -> None:
    frames = np.zeros((3, 8, 8, 8), dtype=np.uint16)
    frames[:, 2, 3, 4] = 100
    example = TrainingExample(
        frames=frames,
        points=np.asarray([[2.0, 3.0, 4.0]], dtype=np.float32),
        division_points=np.empty((0, 3), dtype=np.float32),
        source="synthetic",
        identity="example",
    )
    augmented, points = augment_example(example, np.random.default_rng(7))

    assert augmented.shape == frames.shape
    assert points.shape == (1, 3)
    assert np.all(points >= 0) and np.all(points < 8)
    point = tuple(points[0].astype(int))
    assert augmented[1][point] > 0.5


def test_detection_average_precision_is_score_ordered() -> None:
    truth = np.asarray([[1, 1, 1], [6, 6, 6]], dtype=np.float32)
    predicted = np.asarray([[1, 1, 1], [0, 7, 0], [6, 6, 6]], dtype=np.float32)
    ap, recall = detection_average_precision(
        predicted, np.asarray([0.9, 0.8, 0.7]), truth, radius=1.0
    )

    assert recall == 1.0
    assert ap == (1.0 + 2.0 / 3.0) / 2.0


def test_local_positive_metrics_are_bounded() -> None:
    distances, successes = local_positive_metrics(
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
        np.asarray([0.8], dtype=np.float32),
        np.asarray([[1.5, 1.0, 1.0], [7.0, 7.0, 7.0]], dtype=np.float32),
        search_radius=4.0,
        success_radius=1.0,
    )

    assert distances == [0.5, 4.0]
    assert successes == [True, False]


def test_selection_and_audit_gates_are_explicit() -> None:
    metrics = {
        "synthetic": {
            "mean_average_precision": 0.81,
            "worst_average_precision": 0.66,
            "mean_recall": 0.91,
        },
        "real_positive_only": {
            "positive_peak_recall": 0.86,
            "mean_distance_voxels": 2.0,
            "p90_distance_voxels": 3.0,
        },
    }
    assert selection_passed(metrics)
    assert audit_passed(metrics)
    metrics["real_positive_only"]["positive_peak_recall"] = 0.79
    assert not selection_passed(metrics)
    assert not audit_passed(metrics)
