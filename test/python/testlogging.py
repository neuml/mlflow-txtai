"""
Tests for mlflow-txtai logging
"""

import unittest

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import txtai

import mlflow
from mlflow.entities import SpanType
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID

import mlflow_txtai

# pylint: disable=W0613
def batchsearch(*args, **kwargs):
    """
    Mock method for embeddings batch search.
    """

    return [[{"id": 0, "text": "text search result", "score": 1.0}]]


def createagent(self, *args, **kwargs):
    """
    Mock method for creating an agent.
    """

    def run(s, *a, **kw):
        return "The Roman Empire ruled the Mediterranean and much of Europe"

    self.process = Mock()
    self.process.run = run


def createllm(self, *args, **kwargs):
    """
    Mock method for creating a LLM
    """

    def generator(s, *a, **kw):
        return "The Roman Empire ruled the Mediterranean and much of Europe"

    self.generator = generator


def gettraces():
    """
    Gets traces for the default experiment id.

    Returns:
        list of traces
    """

    return mlflow.MlflowClient().search_traces(experiment_ids=[DEFAULT_EXPERIMENT_ID])


def vectorsencode(*args, **kwargs):
    """
    Mock method for vector encoding.

    Returns:
        mock vectors
    """

    return np.random.rand(1, 10)


class TestLogging(unittest.TestCase):
    """
    Tests for mlflow-txtai logging
    """

    @patch("txtai.Agent.__init__", createagent)
    def testagent(self):
        """
        Test agent
        """

        mlflow_txtai.autolog()

        agent = txtai.Agent()
        response = agent("Tell me about the Roman Empire")

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.AGENT)
        self.assertEqual(span.inputs, {"text": "Tell me about the Roman Empire"})
        self.assertEqual(span.outputs, response)

    def testann(self):
        """
        Test ANN
        """

        mlflow_txtai.autolog()

        ann = txtai.ann.NumPy({})
        ann.backend = np.random.rand(1, 10)

        results = ann.search(np.random.rand(1, 10), 1)

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.RETRIEVER)
        self.assertEqual(span.inputs["limit"], 1)
        self.assertTrue(np.allclose(span.outputs[0], results))

    def testdatabase(self):
        """
        Test database
        """

        mlflow_txtai.autolog()

        database = txtai.database.SQLite({"content": True})
        database.insert([(0, "test", None)])
        results = database.search("select id, text from txtai")

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.RETRIEVER)
        self.assertEqual(span.inputs, {"query": "select id, text from txtai"})
        self.assertTrue(span.outputs, results)

    @patch("txtai.Embeddings.batchsearch", batchsearch)
    def testembeddings(self):
        """
        Test embeddings
        """

        mlflow_txtai.autolog()

        embeddings = txtai.Embeddings()
        results = embeddings.batchsearch("apple")

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.RETRIEVER)
        self.assertEqual(span.inputs, {"queries": "apple"})
        self.assertTrue(span.outputs, results)

    @patch("txtai.Embeddings.batchsearch", batchsearch)
    def testenable(self):
        """
        Test logging enable/disable
        """

        mlflow_txtai.autolog()

        # Get current number of traces
        traces = gettraces()
        count = len(traces)

        embeddings = txtai.Embeddings()
        embeddings.search("test query")

        traces = gettraces()
        self.assertEqual(len(traces), count + 1)

        mlflow_txtai.autolog(disable=True)
        embeddings.search("test query")

        # New trace should not be created
        traces = gettraces()
        self.assertEqual(len(traces), count + 1)

        mlflow_txtai.autolog(log_traces=False)
        embeddings.search("test query")

        # New trace should not be created
        traces = gettraces()
        self.assertEqual(len(traces), count + 1)

    def testerrors(self):
        """
        Test error reporting
        """

        def action(*args, **kwargs):
            raise IOError("Error")

        mlflow_txtai.autolog()

        # No-op workflow that returns inputs
        workflow = txtai.Workflow(tasks=[txtai.workflow.Task(action)])

        with self.assertRaises(IOError):
            list(workflow(["workflow input"]))

    @patch("txtai.LLM.__init__", createllm)
    def testllm(self):
        """
        Test LLM
        """

        mlflow_txtai.autolog()

        llm = txtai.LLM()
        response = llm("Tell me about the Roman Empire")

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.LLM)
        self.assertEqual(span.inputs, {"text": "Tell me about the Roman Empire"})
        self.assertEqual(span.outputs, response)

    def testpipeline(self):
        """
        Test pipeline
        """

        mlflow_txtai.autolog()

        segment = txtai.pipeline.Segmentation(lines=True)
        results = segment("abc\ndef")

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.PARSER)
        self.assertEqual(span.inputs, {"text": "abc\ndef"})
        self.assertEqual(span.outputs, results)

    def testscoring(self):
        """
        Test scoring
        """

        mlflow_txtai.autolog()

        scoring = txtai.scoring.BM25({"terms": True})
        scoring.index(documents=[(0, "test", None)])
        results = scoring.search("test", 1)

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 2)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.RETRIEVER)
        self.assertEqual(span.inputs, {"limit": 1, "query": "test"})
        self.assertTrue(np.allclose(span.outputs, results))

    @patch("txtai.vectors.Vectors.encode", vectorsencode)
    def testvectors(self):
        """
        Test vectors
        """

        mlflow_txtai.autolog()

        vectors = txtai.vectors.Vectors({}, None, None)
        vectors.dimensionality, vectors.qbits = None, None

        results = vectors.vectorize(["test"])

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 1)

        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.EMBEDDING)
        self.assertEqual(span.inputs, {"data": ["test"]})
        self.assertEqual(str(span.outputs), str(results))

    def testworkflow(self):
        """
        Test workflow
        """

        mlflow_txtai.autolog()

        # No-op workflow that returns inputs
        workflow = txtai.Workflow(tasks=[txtai.workflow.Task()])
        results = list(workflow(["workflow input"]))

        trace = gettraces()[0]
        self.assertIsNotNone(trace)
        self.assertEqual(trace.info.status, "OK")
        self.assertEqual(len(trace.data.spans), 2)

        # Check workflow
        span = trace.data.spans[0]
        self.assertEqual(span.span_type, SpanType.CHAIN)
        self.assertEqual(span.inputs, {"elements": results})
        self.assertEqual(span.outputs, results)

        # Check task
        span = trace.data.spans[1]
        self.assertEqual(span.span_type, SpanType.PARSER)
        self.assertEqual(span.inputs["elements"], results)
        self.assertEqual(span.outputs, results)
