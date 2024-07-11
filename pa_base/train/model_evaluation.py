# Copyright (c) 2024, ZDF.
"""
Cross-validation functions and ML evaluation metrics to be used during model evaluation.
"""

import logging
from typing import Dict, List, Optional

import mlflow
import numpy as np
from mlflow.exceptions import MissingConfigException
from sklearn.model_selection import KFold, train_test_split

from pa_base.train.cloud_watch_metrics import CloudWatchMetrics


class MLmetricsCalculator:
    """This class computes different offline evaluation metrics."""

    def __init__(self, metrics: List[str], item_ids_count: Optional[Dict[int, int]] = None):
        """Constructor.
        : param metrics : List of offline metrics (ndcg_at_n, hit_at_n, hit_at_1, mrr, pop_q_1) to compute.
        : param item_ids_count: Item ids and their respective value count.
        """
        self.metrics = metrics
        self.item_ids_count = item_ids_count

        if self.item_ids_count and "pop_q_1" not in self.metrics:
            raise ValueError(
                "'item_id_count' provided, but 'pop_q_1' has to be specified within the metrics to calculate 'pop_q_1'"
            )

        elif "pop_q_1" in self.metrics and not self.item_ids_count:
            raise ValueError(
                "'pop_q_1' specified within the metrics, please provide item_ids_count to calculate 'pop_q_1'"
            )

        elif "pop_q_1" in self.metrics and self.item_ids_count:
            logging.info("Calculating popularity quantiles for all items ids")
            self.each_item_id_popularity_quantile: Dict[int, int] = {
                each_item_id: sum(
                    1
                    for other_item_id_occurence in self.item_ids_count.values()
                    if other_item_id_occurence < each_occurence
                )
                / len(self.item_ids_count)
                for each_item_id, each_occurence in self.item_ids_count.items()
            }

    def compute_offline_metrics(
        self,
        all_targets_item_ids: List[List[int]],
        all_predictions_item_ids: List[List[int]],
        n_for_ranking_metrics: int = 5,
    ) -> Dict[str, int]:
        """
        Computes the offline metrics for all the targets and predictions given at once.
        : param all_targets_item_ids: All targets or final item_ids for different user sequences.
          An example [[10], [100],..[nth seq target]] - n user sequences target items within the list, where in
          the example [10], [100] are the target item_ids for the first two test sequences.
        : param all_predictions_item_ids : All predictions obtained from the trained model for n user sequences.
          An examaple [[1,2,3,4,5,6], [10, 20, 30, 40], ...[nth seq predictions]]- n user sequencs predictions within the,
          list where [1,2,3,4,5,6] represents the prediction items ids from the first user sequence and
          [10, 20, 30, 40] represents the predicitons item ids for the second sequence.
        : param n_for_ranking_metrics : n to consider for calculation for "ndcg_at_n" and  "hit_at_n".
        : return: Calculation of stated metrics from ndcg_at_n, hit_at_n, hit_at_1, mrr, pop_q_1.
          An example: Metrics argument  specified as ['ndcg_at_n', 'hit_at_n', 'pop_q_1'], with the parameter 'n_for_ranking_metrics' taken as 5.
          final_offline_results - {"ndcg_at_5": 0.50, "hit_at_5": 0.70, 'pop_q_1':0.90}, here mrr, and hit_at_1 is not returned as it stated within metrics.

        """

        # Initialize array to store metric values based on targets
        metric_results = {metric: np.zeros(len(all_targets_item_ids), dtype=np.float64) for metric in self.metrics}

        for index, (target, pred) in enumerate(zip(all_targets_item_ids, all_predictions_item_ids)):
            # Calculate pop_quantile_1 based on the first prediction for each user
            if "pop_q_1" in self.metrics:
                metric_results["pop_q_1"][index] = self.each_item_id_popularity_quantile.get(pred[0], 0)

            # Convert predictions to a numpy array
            pred_ids = np.array(pred)

            # Find the rank of the target in the predictions
            rank = np.where(pred_ids == target[0])[0]

            # Compute the metrics based on the rank
            if len(rank) > 0:
                if "mrr" in self.metrics:
                    metric_results["mrr"][index] = 1.0 / (rank + 1)

                if "ndcg_at_n" in self.metrics:
                    metric_results["ndcg_at_n"][index] = (
                        1.0 / np.log2(rank + 2) if rank < n_for_ranking_metrics else 0.0
                    )
                if "hit_at_n" in self.metrics:
                    metric_results["hit_at_n"][index] = 1.0 if rank < n_for_ranking_metrics else 0.0

                if "hit_at_1" in self.metrics:
                    metric_results["hit_at_1"][index] = 1.0 if rank == 0 else 0.0

        # Compute final results
        final_offline_results = {metric: np.mean(metric_results[metric]) for metric in self.metrics}

        return final_offline_results


class MLmetricsLogger(CloudWatchMetrics):
    """This class is used for various logging purposes and also used for mlfow for hyper parameter tuning ."""

    def __init__(self, *, model_name: str, model_target: str = None, variant: str = None):
        if variant is not None:
            CloudWatchMetrics.__init__(
                self,
                model_name=f"{model_name}_{variant}",
                model_target=model_target,
                custom_namespace="MLmetrics",
            )
        else:
            CloudWatchMetrics.__init__(
                self,
                model_name=model_name,
                model_target=model_target,
                custom_namespace="MLmetrics",
            )

        # set the correct folder that is available in the docker container when running in sagemaker
        mlflow.set_tracking_uri("../ml/model")
        try:
            mlflow.set_experiment(model_name)
        except MissingConfigException:
            pass

        mlflow.start_run()
        mlflow.log_param("model", model_name)
        if variant is not None:
            mlflow.log_param("variant", variant)

    def __del__(self):
        mlflow.end_run()

    def log_metric(self, prefix, metric_name, value, step=None, log_cloud_watch=False):
        """Logs a metric both through logging and into mlflow metrics."""
        if step is not None:
            logging.info(f"{prefix}:{metric_name}_{step}={value:.8f}")
            mlflow.log_metric(f"{prefix}_{metric_name}_k", value, step)
        else:
            logging.info(f"{prefix}:{metric_name}={value:.8f}")
            mlflow.log_metric(f"{prefix}_{metric_name}", value)
            if log_cloud_watch:
                self._log_metric(metric_name=f"{prefix}_{metric_name}", value=value)

    def log_mlflow_dataset_metrics(self, dataset, dataset_name, step=0):
        """
        Logs the number of unique users, unique items, and the number of interactions as mlflow metrics.
        """
        mlflow.log_metric(f"{dataset_name}_user_items", len(dataset), step)
        mlflow.log_metric(f"{dataset_name}_users", len(dataset["user_id"].unique()), step)
        mlflow.log_metric(f"{dataset_name}_items", len(dataset["item_id"].unique()), step)


def leave_out_last_split(sequences):
    """Splits sequences into train and test data by leaving all but the last item in each sequence in the train set."""
    train = sequences[:-1]
    test = sequences[-1:]
    return train, test


def train_test_validation_split_by_users(data, test_size=0.25, with_validation_set=True, min_views_user=-1):
    """
    Splits the users with a train_test_split and then splits the click data assigning the corresponding data to the users in the train or test set.
    If with_validation_set==True, a validation set will be created of the same size as the test set.
    """

    users = data["user_id"].unique()

    # if a validation set is required, it has the same size as the test size: first we split up the train set and then divide the rest into validation and test
    if with_validation_set:
        test_size *= 2
    train, test = train_test_split(users, test_size=test_size, random_state=42, shuffle=True)
    train = data[data["user_id"].isin(train)]
    test = data[data["user_id"].isin(test)]
    # we can only use the items known to the model (available in train) when evaluating it
    test = test[test["externalid"].isin(train["externalid"])]
    if min_views_user > 1:
        test = test[test["user_id"].groupby(test["user_id"]).transform("size") >= min_views_user]

    if with_validation_set:
        test_users = test["user_id"].unique()
        validation_user_ids, test_user_ids = train_test_split(test_users, test_size=0.5, random_state=42, shuffle=True)
        validation = test[test["user_id"].isin(validation_user_ids)]
        validation = validation.sort_values("datetime_local", ascending=True)
        test = test[test["user_id"].isin(test_user_ids)]

        validation = validation[
            validation["user_id"].groupby(validation["user_id"]).transform("size") >= min_views_user
        ]

    train = train.sort_values("datetime_local", ascending=True)
    test = test.sort_values("datetime_local", ascending=True)

    if with_validation_set:
        logging.info(f"Splitting data by users: {1-test_size} : {test_size/2} : {test_size/2}")
        logging.info(
            f"Train size : {train.shape[0]}. Test size : {test.shape[0]}. Validation size: {validation.shape[0]}. Num users train: {len(train['user_id'].unique())}, test: {len(test['user_id'].unique())}, validation: {len(validation['user_id'].unique())}. Num items train: {len(train['item_id'].unique())}, test: {len(test['item_id'].unique())}, validation: {len(validation['item_id'].unique())}."
        )
        return train, test, validation
    else:
        logging.info(f"Splitting data by users: {1-test_size} : {test_size}")
        logging.info(
            f"Train size : {train.shape[0]}. Test size : {test.shape[0]}. Num users train: {len(train['user_id'].unique())}, test: {len(test['user_id'].unique())}.  Num items train: {len(train['item_id'].unique())}, test: {len(test['item_id'].unique())}."
        )
        return train, test, None


def train_test_validation_split_by_time(data, test_size=0.25, with_validation_set=True, min_views_user=-1):
    """Splits data into train and test sets, by splitting the sorted data at a certain point depending on the desired test set size."""

    if with_validation_set:
        test_size *= 2

    data = data.sort_values("datetime_local", ascending=True)
    split_index = len(data) - int(len(data) * test_size)

    train = data[:split_index]
    test = data[split_index:]

    validation = None

    if with_validation_set:
        split_index = len(test) - int(len(test) * 0.5)
        validation = test[:split_index]
        test = test[split_index:]

    if min_views_user > 1:
        train = train[train["user_id"].groupby(train["user_id"]).transform("size") >= min_views_user]

        test = test[test["user_id"].groupby(test["user_id"]).transform("size") >= min_views_user]

        if with_validation_set:
            validation = validation[
                validation["user_id"].groupby(validation["user_id"]).transform("size") >= min_views_user
            ]

    # we can only use the items known to the model (available in train) when evaluating it
    test = test[test["externalid"].isin(train["externalid"])]
    if min_views_user > 1:
        test = test[test["user_id"].groupby(test["user_id"]).transform("size") >= min_views_user]

    if with_validation_set:
        # we can only use the items known to the model (available in train) when evaluating it
        validation = validation[validation["externalid"].isin(validation["externalid"])]
        if min_views_user > 1:
            validation = validation[
                validation["user_id"].groupby(validation["user_id"]).transform("size") >= min_views_user
            ]
        logging.info(f"Splitting data by time: {1-test_size} : {test_size/2} : {test_size/2}")
        logging.info(
            f"Train size : {train.shape[0]}. Test size : {test.shape[0]}. Validation size: {validation.shape[0]}. Num users train: {len(train['user_id'].unique())}, test: {len(test['user_id'].unique())}, validation: {len(validation['user_id'].unique())}. Num items train: {len(train['item_id'].unique())}, test: {len(test['item_id'].unique())}, validation: {len(validation['item_id'].unique())}."
        )
        return train, test, validation
    else:
        logging.info(f"Splitting data by time: {1-test_size} : {test_size}")
        logging.info(
            f"Train size : {train.shape[0]}. Test size : {test.shape[0]}. Num users train: {len(train['user_id'].unique())}, test: {len(test['user_id'].unique())}. Num items train: {len(train['item_id'].unique())}, test: {len(test['item_id'].unique())}."
        )
        return train, test, None


def k_fold_split_by_users(data, k=5, min_views_user=-1):
    """
    Splits the users into k sets, taking k-1 sets as training-set and 1 set as test-set and then splits the click data assigning the corresponding data to the users in the train- and test-set.
    """
    users = data["user_id"].unique()

    k_fold = KFold(n_splits=k, random_state=42, shuffle=True)
    logging.info(f"Splitting data by users into {k} sets. ")

    for k, (train, test) in enumerate(k_fold.split(users)):
        train = data[data["user_id"].isin(train)]
        test = data[data["user_id"].isin(test)]

        train = train.sort_values("datetime_local", ascending=True)
        test = test.sort_values("datetime_local", ascending=True)

        # we can only use the items known to the model (available in train) when evaluating it
        test = test[test["externalid"].isin(train["externalid"])]
        if min_views_user > 1:
            test = test[test["user_id"].groupby(test["user_id"]).transform("size") >= min_views_user]

        logging.info(
            f"Train size : {train.shape[0]}. Test size : {test.shape[0]}. Num users train: {len(train['user_id'].unique())}, test: {len(test['user_id'].unique())}.  Num items train: {len(train['item_id'].unique())}, test: {len(test['item_id'].unique())}."
        )
        yield k, train, test
