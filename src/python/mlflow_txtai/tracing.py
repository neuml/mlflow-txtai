"""
MLflow automatic tracing for txtai
"""

import inspect
import json

from mlflow import start_span
from mlflow.entities import SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import STREAM_CHUNK_EVENT_NAME_FORMAT, STREAM_CHUNK_EVENT_VALUE_KEY
from mlflow.tracing.fluent import get_current_active_span, start_span_no_context
from mlflow.tracing.provider import safe_set_span_in_context
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

import txtai


# MLflow tracing name
FLAVOR_NAME = "txtai"


def patchgenerator(target, method, function):
    """
    Patches a generator method with trace logging.

    Args:
        target: target class
        method: target method
        function: target.method function
    """

    # pylint: disable=C0103
    def fn(self, *args, **kwargs):
        return patchclassgenerator(function, self, *args, **kwargs)

    # Add original function as attribute
    setattr(fn, "__wrapped__", function)
    setattr(target, method, fn)


def patchmethod(original, self, *args, **kwargs):
    """
    Patches a method with trace logging.

    Args:
        original: original method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        self.original result
    """

    config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)

    if config.log_traces:
        with start_span(name=spanname(original, self), span_type=spantype(self)) as span:
            # Set attributes
            for attribute, value in vars(self).items():
                span.set_attribute(attribute, value)

            # Set inputs
            span.set_inputs(inputs(original, self, *args, **kwargs))

            # Run method
            results = original(self, *args, **kwargs)

            # Set outputs
            outputs = results.__dict__ if hasattr(results, "__dict__") else results
            span.set_outputs(outputs)

            return results

    return None


def patchclassgenerator(original, self, *args, **kwargs):
    """
    Patches a generator method with trace logging.

    Args:
        original: generator method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        self.original result
    """

    config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)

    if config.log_traces:
        span = startspan(original, self, *args, **kwargs)

        # Start generator
        generator = original(self, *args, **kwargs)

        index, outputs = 0, []
        while True:
            try:
                # Set the span to active only when the generator is running
                with safe_set_span_in_context(span):
                    value = next(generator)
            except StopIteration:
                break
            except Exception as e:
                endspan(span, error=e)
                raise e
            else:
                outputs.append(value)
                spanchunk(span, value, index)
                yield value
                index += 1

        endspan(span, outputs)


def startspan(original, self, *args, **kwargs):
    """
    Starts a span.

    Args:
        original: original method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        span
    """

    # Start span
    return start_span_no_context(
        name=spanname(original, self),
        parent_span=get_current_active_span(),
        span_type=spantype(self),
        attributes=vars(self),
        inputs=inputs(original, self, *args, **kwargs),
    )


def spanchunk(span, chunk, index):
    """
    Adds a span event with a chunk of data.

    Args:
        span: span instance
        chunk: data chunk
        index: data chunk index
    """

    event = SpanEvent(
        name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=index),
        attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
    )
    span.add_event(event)


def endspan(span, outputs=None, error=None):
    """
    Ends a span.

    Args:
        span: span to end
        outputs: optional outputs to log
        error: error to log, if any
    """

    if error:
        span.add_event(SpanEvent.from_exception(error))
        span.end(status=SpanStatusCode.ERROR)
        return

    span.end(outputs=outputs)


def spanname(original, self):
    """
    Creates a span name for inputs.

    Args:
        original: original method
        self: object instance

    Returns:
        span name
    """

    name = self.__class__.__name__
    if not original.__name__.startswith("__"):
        name += f".{original.__name__}"

    return name


def spantype(instance):
    """
    Maps txtai objects to MLflow span types.

    Args:
        instance: txtai object instance

    Returns:
        SpanType
    """

    if isinstance(instance, txtai.Agent):
        return SpanType.AGENT

    if isinstance(instance, txtai.Embeddings):
        return SpanType.RETRIEVER

    if isinstance(instance, txtai.LLM):
        return SpanType.LLM

    if isinstance(instance, (txtai.pipeline.Pipeline, txtai.workflow.Task)):
        return SpanType.PARSER

    if isinstance(instance, (txtai.RAG, txtai.Workflow)):
        return SpanType.CHAIN

    if isinstance(instance, txtai.vectors.Vectors):
        return SpanType.EMBEDDING

    # Default to RETRIEVER
    return SpanType.RETRIEVER


def inputs(func, *args, **kwargs):
    """
    Constructs function inputs as a dictionary.

    Args:
        func: function
        args: arguments
        kwargs: keyword arguments

    Returns:
        dictionary of function inputs
    """

    # Get function arguments
    signature = inspect.signature(func)
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    # Avoid circular references
    return {k: v.__dict__ if hasattr(v, "__dict__") else v for k, v in arguments.items() if v is not None}
