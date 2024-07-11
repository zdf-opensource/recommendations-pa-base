import numpy as np
import pytest

from pa_base.train.model_evaluation import MLmetricsCalculator


@pytest.fixture
def item_ids_count():
    return {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 100}


@pytest.fixture
def metrics():
    return ["ndcg_at_n", "hit_at_n", "hit_at_1", "mrr", "pop_q_1"]


@pytest.fixture
def calculator(metrics, item_ids_count):
    return MLmetricsCalculator(metrics, item_ids_count)


@pytest.mark.parametrize(
    "all_targets_item_ids, all_predictions_item_ids, n_for_ranking_metrics, expected_results",
    [
        (
            [[3], [1], [2]],
            [[3, 2, 1], [1, 2, 3], [1, 2, 3]],
            3,
            {
                "ndcg_at_n": 0.8769765845238192,
                "hit_at_n": 1.0,
                "hit_at_1": 0.6666666666666666,
                "mrr": 0.8333333333333334,
                "pop_q_1": 0.1111111111111111,
            },
        ),
        (
            [[1], [2], [3]],
            [[3, 2, 1], [1, 2, 3], [1, 2, 3]],
            3,
            {
                "ndcg_at_n": 0.5436432511904858,
                "hit_at_n": 1.0,
                "hit_at_1": 0.0,
                "mrr": 0.38888888888888884,
                "pop_q_1": 0.1111111111111111,
            },
        ),
    ],
)
def test_compute_offline_metrics(
    calculator, all_targets_item_ids, all_predictions_item_ids, n_for_ranking_metrics, expected_results
):
    results = calculator.compute_offline_metrics(all_targets_item_ids, all_predictions_item_ids, n_for_ranking_metrics)

    for metric, expected_value in expected_results.items():
        assert np.isclose(results[metric], expected_value)
