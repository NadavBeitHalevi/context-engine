"""Offline unit tests for the Context Library pipeline, using fake Pinecone and OpenAI clients.

The fake index is eventually consistent like serverless Pinecone: writes become visible
only after a few reads, so the pipeline's wait logic is exercised for real.
"""

from types import SimpleNamespace as NS
from typing import Any

import pytest
from pinecone.exceptions import ServiceException

from config import Settings
from rag_pipeline import rag_context_librarian as pipeline
from rag_pipeline.context_library import ContextBlueprint, blueprint_metadata

NAMESPACE = "ContextLibrary"

BLUEPRINTS: list[ContextBlueprint] = [
    {"id": "bp_a", "description": "style A", "blueprint": {"instruction": "write like A"}},
    {"id": "bp_b", "description": "style B", "blueprint": {"instruction": "write like B"}},
]


class FakeIndex:
    """In-memory, eventually consistent stand-in for a Pinecone serverless index."""

    def __init__(self, lag: int = 0, stored: dict[str, dict[str, Any]] | None = None, fail_reads: int = 0):
        self.visible: dict[str, dict[str, Any]] = dict(stored or {})  # id -> metadata
        self.pending: list[tuple[int, Any]] = []  # (visible after this many reads, apply fn)
        self.lag = lag
        self.reads = 0
        self.fail_reads = fail_reads  # raise a transient 503 on the next N reads
        self.calls: list[str] = []

    def _write(self, apply) -> None:
        self.pending.append((self.reads + self.lag, apply))

    def _read(self) -> None:
        if self.fail_reads:
            self.fail_reads -= 1
            raise ServiceException("Service Unavailable", status_code=503)
        self.reads += 1
        ready = [fn for at, fn in self.pending if at <= self.reads]
        self.pending = [(at, fn) for at, fn in self.pending if at > self.reads]
        for fn in ready:
            fn()

    def upsert(self, *, vectors, namespace):
        assert namespace == NAMESPACE
        self.calls.append("upsert")
        self._write(lambda: self.visible.update({v["id"]: v["metadata"] for v in vectors}))

    def delete(self, *, ids=None, delete_all=False, namespace):
        assert namespace == NAMESPACE
        self.calls.append("delete_all" if delete_all else f"delete:{','.join(ids)}")
        if delete_all:
            self._write(self.visible.clear)
        else:
            self._write(lambda: [self.visible.pop(i, None) for i in ids])

    def fetch(self, *, ids, namespace):
        if not ids:
            raise ValueError("Pinecone rejects a fetch with no ids")
        self._read()
        return NS(vectors={i: NS(metadata=dict(self.visible[i])) for i in ids if i in self.visible})

    def list(self, *, namespace):
        self._read()
        yield NS(vectors=[NS(id=i) for i in self.visible])


class FakePinecone:
    def __init__(self, index: FakeIndex, exists: bool = True, dimension: int = 1536,
                 metric: str = "cosine", ready_after: int = 0):
        self.index, self.exists = index, exists
        self.dimension, self.metric = dimension, metric
        self.ready_after = ready_after  # describe_index calls before status.ready is True
        self.created: dict[str, Any] | None = None

    def has_index(self, name):
        return self.exists

    def describe_index(self, name):
        ready = self.ready_after <= 0
        self.ready_after -= 1
        return NS(dimension=self.dimension, metric=self.metric, status=NS(ready=ready))

    def create_index(self, **kwargs):
        self.created, self.exists = kwargs, True

    def Index(self, name):
        return self.index


FAKE_EMBEDDINGS = NS(embeddings=NS(create=lambda input, model: NS(data=[NS(embedding=[0.0, 0.1])])))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(pipeline, "POLL_INTERVAL_S", 0)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, pinecone_index_name="test-index", namespace_context=NAMESPACE,
                    embedding_model="text-embedding-3-small")


def run(settings, pc, **kwargs):
    return pipeline.run_pipeline(settings, pc=pc, embedding_client=FAKE_EMBEDDINGS, timeout_s=5, **kwargs)


def test_creates_missing_index_and_loads_blueprints(settings):
    index = FakeIndex()
    pc = FakePinecone(index, exists=False)

    result = run(settings, pc, blueprints=BLUEPRINTS)

    assert result.created_index and result.uploaded == 2 and result.removed == 0
    assert pc.created["dimension"] == 1536 and pc.created["metric"] == "cosine"
    assert index.visible == {b["id"]: blueprint_metadata(b) for b in BLUEPRINTS}


@pytest.mark.parametrize("field, value, message", [("dimension", 3072, "dimension"), ("metric", "dotproduct", "metric")])
def test_existing_index_that_does_not_fit_is_rejected(settings, field, value, message):
    pc = FakePinecone(FakeIndex(), **{field: value})

    with pytest.raises(ValueError, match=message):
        run(settings, pc, blueprints=BLUEPRINTS)


def test_waits_for_existing_index_to_become_ready(settings):
    pc = FakePinecone(FakeIndex(), ready_after=3)

    result = run(settings, pc, blueprints=BLUEPRINTS)

    assert not result.created_index and pc.ready_after < 0  # polled until ready


def test_rerun_waits_until_new_content_is_served(settings):
    # Old versions of both blueprints are already stored (a re-run), and writes lag.
    old = {b["id"]: {"description": "old", "blueprint_json": "{}"} for b in BLUEPRINTS}
    index = FakeIndex(lag=3, stored=old)

    run(settings, FakePinecone(index), blueprints=BLUEPRINTS)

    # run_pipeline only returns once the *new* content is what Pinecone serves.
    assert index.visible == {b["id"]: blueprint_metadata(b) for b in BLUEPRINTS}


def test_removes_blueprints_that_are_no_longer_defined(settings):
    stored = {"bp_removed": {"description": "gone", "blueprint_json": "{}"}}
    index = FakeIndex(lag=2, stored=stored)

    result = run(settings, FakePinecone(index), blueprints=BLUEPRINTS)

    assert result.removed == 1
    assert set(index.visible) == {"bp_a", "bp_b"}


def test_fresh_empties_the_namespace_before_uploading(settings):
    index = FakeIndex(stored={"bp_a": {"description": "old", "blueprint_json": "{}"}})

    run(settings, FakePinecone(index), fresh=True, blueprints=BLUEPRINTS)

    assert index.calls[:2] == ["delete_all", "upsert"]
    assert index.visible == {b["id"]: blueprint_metadata(b) for b in BLUEPRINTS}


def test_empty_library_skips_fetch_and_removes_everything(settings):
    index = FakeIndex(stored={"bp_a": {"description": "x", "blueprint_json": "{}"}})

    result = run(settings, FakePinecone(index), blueprints=[])

    assert result.uploaded == 0 and result.removed == 1
    assert "upsert" not in index.calls and index.visible == {}


def test_transient_pinecone_error_while_waiting_is_retried(settings):
    index = FakeIndex(fail_reads=2)

    result = run(settings, FakePinecone(index), blueprints=BLUEPRINTS)

    assert result.uploaded == 2
    assert index.visible == {b["id"]: blueprint_metadata(b) for b in BLUEPRINTS}


def test_gives_up_with_timeout_if_content_never_appears(settings):
    index = FakeIndex(lag=10_000)  # writes never become visible

    with pytest.raises(TimeoutError, match="serve the uploaded blueprints"):
        pipeline.run_pipeline(settings, pc=FakePinecone(index), embedding_client=FAKE_EMBEDDINGS,
                              blueprints=BLUEPRINTS, timeout_s=0)


def test_serves_current_blueprints_detects_mismatch():
    index = FakeIndex(stored={"bp_a": blueprint_metadata(BLUEPRINTS[0]),
                              "bp_b": {"description": "style B", "blueprint_json": '{"instruction": "old"}'}})

    assert not pipeline.serves_current_blueprints(index, NAMESPACE, BLUEPRINTS)
    assert pipeline.serves_current_blueprints(index, NAMESPACE, BLUEPRINTS[:1])
    assert pipeline.serves_current_blueprints(index, NAMESPACE, [])
