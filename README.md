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
    embedding_model="VectorAmp-Embedding-2560",
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
dataset.add_texts("VectorAmp uses SABLE for high-performance vector search.")

# Optional IDs and metadata are still supported for batches.
dataset.add_texts(
    ["VectorAmp uses SABLE for high-performance vector search."],
    ids=["sable-note"],
    metadatas=[{"source": "readme"}],
)
```

## Search

Search by text:

```python
results = dataset.search("How does SABLE work?", top_k=10, include_documents=True)
```

Search by vector:

```python
results = dataset.search([0.1, 0.2, 0.3], top_k=5)
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

Or pass a typed source builder. The SDK creates the source, extracts its returned ID,
and starts the ingestion job for the dataset:

```python
from vectoramp import WebSource

job = dataset.ingest_source(
    WebSource(
        start_urls=["https://docs.example.com/"],
        max_depth=1,
    )
)
```

List jobs with pagination:

```python
jobs = client.ingestion.list_jobs(dataset_id=dataset_id, limit=50, offset=0)
```

Create sources with typed helpers. `client.sources` is an alias for the ingestion
source APIs, so existing `client.ingestion.create_source(...)` code still works:

```python
web = client.sources.create_web(
    start_urls=["https://docs.example.com/"],
    max_depth=1,
    include_assets=True,
    max_assets_per_page=5,
)

s3 = client.sources.create_s3(
    bucket="my-bucket",
    prefix="documents/",
    role_arn="arn:aws:iam::123456789012:role/vectoramp-ingestion",
)

gcs = client.sources.create_gcs(bucket="my-gcs-bucket", prefix="documents/")

jira = client.sources.create_jira(
    cloud_id="atlassian-cloud-id",
    project_keys=["ENG"],
    include_comments=True,  # default
)


gdrive = client.sources.create_google_drive(
    folder_ids=["drive-folder-id"],
    include_shared_drives=True,
)

upload_source = client.sources.create_file_upload()
```

The supported typed source classes are `WebSource`, `S3Source`, `GCSSource`,
`GoogleDriveSource` (`source_type="gdrive"`), `JiraSource`, and
`FileUploadSource` (`source_type="file_upload"`). Use `GenericSource` as an
escape hatch when the API supports a source type before the SDK has a dedicated
class:

```python
from vectoramp import GenericSource

source = client.sources.create(
    GenericSource(
        name="custom-source",
        source_type="custom",
        config={"any_api_field": "value"},
    )
)
```

The low-level create API is preserved:

```python
source = client.ingestion.create_source(
    name="docs-site",
    source_type="web",
    config={"start_urls": ["https://docs.example.com/"], "max_depth": 1},
)
```

Upload local files through the REST upload flow. The SDK creates a `file_upload` source with a generated name when `source_name` is omitted, initializes presigned uploads, uploads bytes to the returned URLs, and completes the upload job:

```python
job = dataset.ingest_files(["./docs/whitepaper.pdf", "./docs/overview.txt"])
```

Pass `source_name="product-docs-upload"` only when you want a specific source name.

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

## Dataset documents

Datasets expose retained source documents when ingestion stored originals:

```python
page = client.datasets.list_documents("dataset_id", limit=50, cursor=None, status="ready")
for document in page.get("documents", []):
    print(document["id"], document.get("file_name"))

content = client.datasets.download_document("dataset_id", "document_id")
open("document.bin", "wb").write(content)

# Resource-style calls work too:
dataset = client.datasets.get("dataset_id")
dataset.list_documents()
dataset.download_document("document_id")
```
