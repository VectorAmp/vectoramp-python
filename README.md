# VectorAmp Python SDK

Python client for the VectorAmp API. It is package-ready, typed, and defaults to `https://api.vectoramp.com`.

> Status: public SDK scaffold. Do not publish packages from this repository until a release is approved.

## Install

```bash
pip install vectoramp
```

For local development:

```bash
git clone git@gitlab.com:VectorAmp/SDK/Python.git
cd Python
pip install -e '.[dev]'
pytest
```

## Authentication

The SDK sends API keys with the `X-API-Key` header.

```python
from vectoramp import VectorAmp

client = VectorAmp(api_key="va_...")
```

Or use the environment variable:

```bash
export VECTORAMP_API_KEY=va_...
```

```python
client = VectorAmp()
```

Configure a non-production API host with `base_url`:

```python
client = VectorAmp(api_key="va_...", base_url="http://localhost:8080")
```

## Datasets

Dataset creation always requests the SABLE index. The SDK intentionally does **not** expose an `index_type` option.

```python
dataset = client.datasets.create(
    name="product-docs",
    dim=2560,
    metric="cosine",
    embedding_provider="vectoramp",
    embedding_model="Qwen/Qwen3-Embedding-4B",
)

dataset_id = dataset.id  # also available as dataset["id"] for compatibility
```

Create/get/list return `Dataset` resource objects. They keep the raw API payload and
carry the client/services needed for instance methods:

```python
dataset = client.datasets.get(dataset_id)
print(dataset.id, dataset.raw_data)
```

List methods return the API pagination envelope with `Dataset` objects in the `datasets` field:

```python
page = client.datasets.list(limit=50, offset=0)
for dataset in page["datasets"]:
    print(dataset.id, dataset.raw_data)
print(page["total"], page["limit"], page["offset"])
```

Service-style methods are still available for callers that prefer passing `dataset_id` explicitly.
Get or delete a dataset:

```python
client.datasets.get(dataset_id)
client.datasets.delete(dataset_id)
```

## Insert vectors and texts

Insert raw vectors:

```python
dataset.insert(
    [
        {
            "id": "doc-001",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"title": "Intro", "source": "manual"},
        }
    ],
)
```

Embed text with the dataset's configured model and insert the resulting vectors:

```python
dataset.add_texts(
    ["VectorAmp uses SABLE for high-performance vector search."],
    ids=["sable-note"],
    metadatas=[{"source": "readme"}],
)
```

## Search

Search by text:

```python
results = dataset.search(
    text="How does SABLE work?",
    top_k=10,
    include_documents=True,
)
```

Search by vector:

```python
results = dataset.search(vector=[0.1, 0.2, 0.3], top_k=5)
```

Hybrid and filtered search:

```python
results = dataset.search(
    text="wireless headphones",
    top_k=10,
    filters={"category": "electronics"},
    advanced_filters=[{"field": "price", "op": "lt", "value": 100}],
    hybrid=True,
    sparse_query="wireless headphones",
    alpha=0.7,
)
```

## Ingestion

Start ingestion from an existing source:

```python
job = dataset.ingest_source("source-uuid")
```

List jobs with pagination:

```python
jobs = client.ingestion.list_jobs(dataset_id=dataset_id, limit=50, offset=0)
```

Create a source:

```python
source = client.ingestion.create_source(
    name="docs-site",
    source_type="web",
    config={"start_urls": ["https://docs.example.com/"], "max_depth": 1, "type": "web"},
)
```

Upload local files through the REST upload flow. The SDK creates a `file_upload` source, initializes presigned uploads, uploads bytes to the returned URLs, and completes the upload job:

```python
job = dataset.ingest_files(
    paths=["./docs/whitepaper.pdf", "./docs/overview.txt"],
    source_name="product-docs-upload",
)
```

## Intelligence / RAG

Non-streaming query:

```python
answer = dataset.ask("What are the key product features?", top_k=5)
print(answer["answer"])
```

Streaming SSE query:

```python
for event in client.ask_stream("Summarize the docs", dataset_id=dataset_id):
    if event["chunk_type"] == "text":
        print(event["content"], end="")
```

## Transport abstraction

`VectorAmp` depends on a small transport interface. `RestTransport` is provided today; a future gRPC transport can implement the same `request`, `stream`, and `close` methods without changing resource UX.

## Development

```bash
pip install -e '.[dev]'
ruff check .
mypy src
pytest
```

GitLab CI runs Ruff, mypy, and pytest with coverage.
