# Copyright (c) 2024, ZDF.
"""
Cross-validation functions and ML evaluation metrics to be used during model evaluation.
"""

import logging
import operator

import mlflow
import numpy as np
import scipy.stats as st
from mlflow.exceptions import MissingConfigException
from sklearn.model_selection import KFold, train_test_split

from pa_base.train.cloud_watch_metrics import CloudWatchMetrics


class MLmetrics(CloudWatchMetrics):
    """
    This class computes ML metrics and logs them in logging, mlflow, and CloudWatch
    """

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

    def sequence_mrr_stream(self, target, predictions):
        """Computes the MRR score for a given target and unsorted scores predicted by a model."""
        # see spotlight.evaluation.sequence_mrr_score : https://maciejkula.github.io/spotlight/evaluation.html
        if len(target) == 0:
            return 0
        scores = -np.array(list(map(operator.itemgetter(1), predictions)))
        pred = np.array(list(map(operator.itemgetter(0), predictions)))
        itemindex = np.where(pred == target)[0]
        # target was not within the predictions
        if len(itemindex) == 0:
            return 0

        mrr = 1.0 / st.rankdata(scores)[itemindex]
        mrr = mrr.mean()
        return mrr

    def hit_rate_at_n_stream(self, target, predictions, n=10):
        """Computes the hit rate for a given target and predictions made by a model."""
        hit = 0
        best_n_item_ids = list(map(operator.itemgetter(0), predictions[:n]))
        if len(target) > 0 and target[0] in best_n_item_ids:
            hit += 1
        return hit

    def ndcg_at_n_stream(self, target, predictions, n=10):
        """Computes the NDCG rate for a given target and predictions made by a model."""
        ndcg = 0
        best_n_item_ids = np.array(list(map(operator.itemgetter(0), predictions[:n])))
        if len(target) > 0 and target[0] in best_n_item_ids:
            rank = np.where(best_n_item_ids == target)[0]
            ndcg += float(1.0 / np.log2(rank + 2))
        return ndcg

    def metrics_at_n_stream(self, target, predictions, n=10):
        """Computes the NDCG rate for a given target and predictions made by a model."""
        ndcg = 0
        hit = 0
        hit_at_1 = 0

        if len(target) == 0:
            return ndcg, hit, hit_at_1, 0

        scores = -np.array(list(map(operator.itemgetter(1), predictions)))
        pred = np.array(list(map(operator.itemgetter(0), predictions)))
        rank = np.where(pred == target)[0]
        # target was not within the predictions
        if len(rank) == 0:
            return ndcg, hit, hit_at_1, 0

        mrr = 1.0 / st.rankdata(scores)[rank]
        mrr = mrr.mean()

        if rank[0] < n:
            ndcg += float(1.0 / np.log2(rank + 2))
            hit += 1
            if rank == 0:
                hit_at_1 += 1
        return ndcg, hit, hit_at_1, mrr

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
