"""End-to-end qualification through the public SDK API and real native runtime."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

from vectoramp import VectorAmp

root = Path(
    os.environ.get(
        "VECTORAMP_EMBEDDED_E2E_ROOT",
        tempfile.mkdtemp(prefix="vectoramp-python-e2e-"),
    )
)
client = VectorAmp.embedded(root)
dataset = client.datasets.create(
    name="sdk-e2e",
    dim=4,
    metadata_schema={"tenant": "string", "ordinal": "i32"},
    tuning={"training_threshold": 256},
)
vectors = [
    {
        "id": vector_id,
        "values": [
            math.cos((vector_id - 1) * 0.01),
            math.sin((vector_id - 1) * 0.01),
            0.0,
            0.0,
        ],
        "metadata": {
            "tenant": "acme" if vector_id % 2 == 0 else "other",
            "ordinal": vector_id,
        },
    }
    for vector_id in range(1, 257)
]
assert dataset.insert(vectors) == {"inserted": 256}


def assert_results(results, expected_ids, query_angle=0.0):
    assert [int(result["id"]) for result in results] == expected_ids
    assert results[0]["distance"] >= 0.0
    assert results[1]["distance"] > 0.0
    assert all(
        left["distance"] <= right["distance"] for left, right in zip(results, results[1:])
    )
    for result in results:
        vector_id = int(result["id"])
        expected_distance = 1.0 - math.cos((vector_id - 1) * 0.01 - query_angle)
        assert abs(result["distance"] - expected_distance) < 1e-5
        assert result["score"] == result["distance"]
        assert result["metadata"]["ordinal"] == str(vector_id)
        assert result["metadata"]["tenant"] == (
            "acme" if vector_id % 2 == 0 else "other"
        )


query = [1.0, 0.0, 0.0, 0.0]
unfiltered = dataset.search(query, top_k=10, include_metadata=True)
assert_results(unfiltered["results"], list(range(1, 11)))
without_metadata = dataset.search(query, top_k=1, include_metadata=False)
assert "metadata" not in without_metadata["results"][0]
filtered = dataset.search(query, top_k=10, filters={"tenant": "acme"})
assert_results(filtered["results"], list(range(2, 21, 2)))
range_query_angle = 1.49
range_query = [math.cos(range_query_angle), math.sin(range_query_angle), 0.0, 0.0]
strict_filtered = dataset.search(
    range_query,
    top_k=10,
    filters={"tenant": "acme"},
    advanced_filters=[{"field": "ordinal", "op": "lt", "value": 100}],
    include_metadata=True,
)
assert_results(strict_filtered["results"], list(range(98, 79, -2)), range_query_angle)

deadline = time.monotonic() + 120
status = dataset.index_status()
while status["state"] not in {"indexed", "failed"} and time.monotonic() < deadline:
    time.sleep(0.25)
    status = dataset.index_status()
assert status["state"] == "indexed", status
assert status["vector_count"] == 256
assert dataset.optimize()["state"] == "indexed"
assert_results(dataset.search(query, top_k=10)["results"], list(range(1, 11)))
assert_results(
    dataset.search(
        range_query,
        top_k=10,
        filters={"tenant": "acme"},
        advanced_filters=[{"field": "ordinal", "op": "lt", "value": 100}],
    )["results"],
    list(range(98, 79, -2)),
    range_query_angle,
)
dataset_id = dataset.id
client.close()

reopened = VectorAmp.embedded(root)
restored = reopened.datasets.get(dataset_id)
assert reopened.datasets.stats(dataset_id)["vector_count"] == 256
post_reopen = restored.search(query, top_k=10, filters={"tenant": "acme"})
assert_results(post_reopen["results"], list(range(2, 21, 2)))
post_reopen_range = restored.search(
    range_query,
    top_k=10,
    filters={"tenant": "acme"},
    advanced_filters=[{"field": "ordinal", "op": "lt", "value": 100}],
    include_metadata=True,
)
assert_results(post_reopen_range["results"], list(range(98, 79, -2)), range_query_angle)
reopened.close()
