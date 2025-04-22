"""
MLflow automatic tracing for txtai
"""

import collections
import inspect

from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

import txtai

from mlflow_txtai.tracing import patchgenerator, patchmethod, FLAVOR_NAME


@experimental
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from txtai to MLflow. Currently, MLflow
    only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for txtai calls. If ``False``,
            no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the txtai autologging integration. If ``False``,
            enables the txtai autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during txtai
            autologging. If ``False``, show all events and warnings.
    """

    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    setup(log_traces, disable, silent)

    # Base mappings
    mappings = {
        txtai.archive.Archive: ["load", "save"],
        txtai.cloud.Cloud: ["load", "save"],
        txtai.Embeddings: ["batchsearch", "delete", "index", "load", "save", "upsert"],
        txtai.LLM: ["__call__"],
        txtai.RAG: ["__call__"],
        txtai.workflow.Task: ["__call__"],
        txtai.Workflow: ["__call__"],
        txtai.vectors.Vectors: ["vectorize"],
    }

    # Add agent logging only if agents are enabled
    if "__call__" in txtai.Agent.__dict__:
        mappings[txtai.Agent] = ["__call__"]

    # Add component mappings
    register(mappings)

    # Add autologging
    for target, methods in mappings.items():
        for method in methods:
            function = getattr(target, method)

            # Separate path for patching generator methods vs standard methods
            if inspect.isgeneratorfunction(function):
                patchgenerator(target, method, function)
            else:
                safe_patch(FLAVOR_NAME, target, method, patchmethod)


# pylint: disable=W0613
@autologging_integration(FLAVOR_NAME)
def setup(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    """
    The @autologging_integration annotation must be applied here, and the callback injection
    needs to happen outside the annotated function. This is because the annotated function is NOT
    executed when disable=True is passed. This prevents us from removing our callback and patching
    when autologging is turned off.
    """


def register(mappings):
    """
    Register mappings for txtai components.

    Args:
        mappings: where to add mappings
    """

    # ANN
    for target in vars(txtai.ann).values():
        if ischildclass(target, txtai.ann.ANN):
            mappings[target] = ["append", "delete", "index", "load", "save", "search"]

    # Database, Graph, Scoring
    for module, clss in [
        (txtai.database, txtai.database.Database),
        (txtai.graph, txtai.graph.Graph),
        (txtai.scoring, txtai.scoring.Scoring),
    ]:
        for target in vars(module).values():
            if ischildclass(target, clss):
                mappings[target] = ["delete", "insert", "load", "save", "search"]

                if hasattr(target, "index"):
                    mappings[target] += ["index", "upsert"]

    # Pipelines
    for target in vars(txtai.pipeline).values():
        if ischildclass(target, txtai.pipeline.Pipeline) and iscallable(target) and target not in mappings:
            mappings[target] = ["__call__"]


def iscallable(target):
    """
    Checks if target is a callable method.

    Args:
        target: method to check

    Returns:
        True if target is callable
    """

    return issubclass(target, collections.abc.Callable)


def ischildclass(target, check):
    """
    Checks if target is a subclass of check.

    Args:
        target: class to check
        check: validation check class

    Returns:
        True if target is a subclass of check
    """

    return inspect.isclass(target) and issubclass(target, check) and target != check
