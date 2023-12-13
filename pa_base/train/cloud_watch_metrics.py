# Copyright (c) 2023, ZDF.
"""
Put metrics into CloudWatch.
"""

import logging
import typing as t

import boto3

from pa_base.configuration.config import DEFAULT_REGION, SITE

# silence logging for boto3/botocore
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)


class CloudWatchMetrics:
    def __init__(
        self,
        *,
        model_name: str,
        model_target: t.Optional[str] = None,
        custom_namespace=None,
        enabled: bool = True,
    ):
        """
        custom logger that logs certain metrics to AWS CloudWatch

        :param model_name: name of the model whose metrics are logged
        :param model_target: target of the model whose metrics are logged (pctablet, mobile, tv)
        :param custom_namespace: custom namespace for the metric in AWS CloudWatch
        :param enabled: enable or disable training job metric logging to cloud watch
        """
        self.model_name = model_name
        self.model_target = model_target
        if custom_namespace is not None:
            self.namespace = f"{custom_namespace}/{SITE}"
        else:
            self.namespace = f"modelgen/{SITE}"
        self._enabled = enabled

    @property
    def enabled(self):
        """whether training job metric logging to cloud watch is enabled"""
        # read-only property
        return self._enabled

    def _log_metric(self, *, metric_name: str, value: t.Union[int, float], unit: str = "None"):
        if not self.enabled:
            return
        try:
            # Get the current region from the active boto3 credentials
            sess = boto3.session.Session()
            region = sess.region_name

            # otherwise fall back to the default region
            if region is None:
                region = DEFAULT_REGION

            # Create CloudWatch client
            cloudwatch = boto3.client("cloudwatch", region_name=region)
            dimensions = [
                {
                    "Name": "Model",
                    "Value": self.model_name,
                },
            ]
            if self.model_target:
                dimensions.append(
                    {
                        "Name": "Target",
                        "Value": self.model_target,
                    }
                )
            # Put custom metrics
            cloudwatch.put_metric_data(
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Dimensions": dimensions,
                        "Unit": unit,
                        "Value": value,
                    }
                ],
                Namespace=self.namespace,
            )
        except Exception as exc:
            logging.error(
                f"Could not put S3 metric '{metric_name} = {value} {unit}' "
                f"for model '{self.model_name}' with target '{self.model_target}'.",
                exc_info=exc,
            )

    def log_metric(self, *, metric_name: str, value: t.Union[int, float], unit: str = "None"):
        self._log_metric(metric_name=metric_name, value=value, unit=unit)

    def sample_count(self, value):
        self._log_metric(metric_name="Samples", value=value)

    def prep_time(self, value):
        self._log_metric(metric_name="Preparation Time", value=value, unit="Seconds")

    def gen_time(self, value):
        self._log_metric(metric_name="Generation Time", value=value, unit="Seconds")

    def model_training_complete(self):
        self._log_metric(metric_name="ModelTrainingComplete", value=1)
