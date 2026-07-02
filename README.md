<div align="center">
  <a href="https://vectoramp.com/">
    <picture>
      <source media="(prefers-color-scheme: light)" srcset=".github/images/logo-full-light.svg">
      <source media="(prefers-color-scheme: dark)" srcset=".github/images/logo-full-dark.svg">
      <img alt="VectorAmp Logo" src=".github/images/logo-full-dark.svg" width="50%">
    </picture>
  </a>
</div>

# VectorAmp Python SDK

Python client for the VectorAmp API. It is package-ready, typed, and defaults to `https://api.vectoramp.com`.

Licensed under the [Apache License 2.0](LICENSE).

## Install

```bash
pip install vectoramp
```

For local development:

```bash
git clone https://github.com/VectorAmp/vectoramp-python.git
cd vectoramp-python
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

Dataset creation always requests the SABLE index. The SDK intentionally does **not** expose an `index_type` option. Built-in helpers infer dimensions for VectorAmp 4B and OpenAI `text-embedding-3-small`/`text-embedding-3-large`.

The only required argument is `name`. The default embedding is `VectorAmp-Embedding-4B` (`dim=2560`, `metric="cosine"`):

```python
dataset = client.datasets.create("product-docs")

dataset_id = dataset.id  # also available as dataset["id"] for compatibility
```

Enable hybrid (dense + sparse) indexing with `hybrid=True`:

```python
dataset = client.datasets.create("product-docs", hybrid=True)
```

Bring your own embedding model with the `openai` helper (dimension is inferred):

```python
from vectoramp import openai

dataset = client.datasets.create(
    "product-docs",
    embedding=openai("small"),  # or openai("large")
)
```

For a custom/unknown model you must pass `dim` explicitly:

```python
dataset = client.datasets.create(
    "product-docs",
    embedding={"provider": "acme", "model": "acme-embed-v1"},
    dim=1024,
)
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

Insert raw vectors. A record `id` may be a string **or** an integer; integer ids
are sent as JSON numbers, not coerced to strings:

```python
dataset.insert(
    [
        {
            "id": "doc-001",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"title": "Intro", "source": "manual"},
        },
        {
            "id": 42,  # preserved as a JSON number
            "values": [0.4, 0.5, 0.6],
            "metadata": {"title": "Appendix"},
        },
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
    rerank={"enabled": True},  # vectoramp / VectorAmp-Rerank-v1
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

The same one-liner works for any source type, e.g. Confluence:

```python
from vectoramp import ConfluenceSource

job = dataset.ingest_source(
    ConfluenceSource(
        base_url="https://acme.atlassian.net",
        username="user@example.com",
        api_token="…",
        spaces=["ENG"],
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

confluence = client.sources.create_confluence(
    cloud_id="atlassian-cloud-id",  # or base_url="https://acme.atlassian.net"
    username="user@example.com",
    api_token="…",                  # auth_mode defaults to "basic"
    spaces=["ENG", "DOCS"],         # empty/omitted = all accessible spaces
    include_attachments=True,       # default False
)

gdrive = client.sources.create_google_drive(
    folder_ids=["drive-folder-id"],
    include_shared_drives=True,
)

upload_source = client.sources.create_file_upload()
```

`sync_mode` is omitted unless you set it, so the server applies its default of
`"incremental"` for the connectors that support it. Pass `sync_mode="full"` to
force a full re-sync.

The supported typed source classes are `WebSource`, `S3Source`, `GCSSource`,
`GoogleDriveSource` (`source_type="gdrive"`), `JiraSource`, `ConfluenceSource`,
and `FileUploadSource` (`source_type="file_upload"`). Use `GenericSource` as an
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

CI runs Ruff, mypy, and pytest with coverage on every change.

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

### Intelligence sessions

```python
session = client.intelligence.create_session(title="Planning", dataset_id=dataset.id)
client.intelligence.append_message(session["id"], role="user", content="Summarize the docs")
messages = client.intelligence.list_messages(session["id"], limit=100)
```

Intelligence answers return `sources[]` and `chunks[]`. Inline `[1]` citations refer to `sources[0]`; `preview_ref` is an opaque preview token, not a storage key.

## Method reference

Both access styles work everywhere the SDK allows it:
`client.datasets.search(id, …)` and `dataset.search(…)`. Required arguments are
listed first; optional arguments show their default.

### Client (`VectorAmp`)

- `VectorAmp(api_key=None, *, base_url="https://api.vectoramp.com", timeout=30.0)` — `api_key` falls back to `VECTORAMP_API_KEY`.
- `client.ask(query, *, dataset_id=None, top_k=5, conversation_history=None, include_sources=True)`
- `client.ask_stream(query, *, dataset_id=None, top_k=5, conversation_history=None, include_sources=True)` — iterator of SSE chunks.
- `client.close()` (also a context manager).

### Datasets (`client.datasets` / `Dataset`)

- `create(name, *, dim=None, metric="cosine", embedding=None, embedding_provider="vectoramp", embedding_model="VectorAmp-Embedding-4B", hybrid=False, filters=None, metadata_schema=None, tuning=None)` → `Dataset`. Always SABLE. `dim` inferred for built-in models; required for custom models.
- `list(*, limit=50, offset=0)` → page with `Dataset` objects.
- `get(dataset_id)` → `Dataset`.
- `delete(dataset_id)` / `dataset.delete()`.
- `stats(dataset_id)` / `dataset.stats()`.
- `search(dataset_id, query=None, *, vector=None, text=None, search_text=None, top_k=10, filters=None, advanced_filters=None, embedding_provider=None, embedding_model=None, nprobe_override=None, rerank_depth_override=None, hybrid=None, sparse_query=None, alpha=None, include_embeddings=None, include_documents=None, include_metadata=None, rerank=None)` / `dataset.search(…)`. `query` accepts a string (text) or float sequence (vector); `top_k` defaults to 10.
- `insert(dataset_id, vectors)` and `insert_vectors(dataset_id, vectors)` / `dataset.insert(vectors)`. Record `id` may be `str` or `int` (integers stay JSON numbers).
- `embed(dataset_id, *, text=None, texts=None)` / `dataset.embed(…)`.
- `add_texts(dataset_id, texts, *, ids=None, metadatas=None)` / `dataset.add_texts(texts, …)`. Single string or list; ids auto-generated when omitted; copies the text into `metadata.text`.
- `list_documents(dataset_id, *, limit=50, cursor=None, status=None)` / `dataset.list_documents(…)`.
- `download_document(dataset_id, document_id)` / `dataset.download_document(document_id)` → bytes.
- `ensure_engine(dataset_id)`.
- `dataset.ask(query, *, top_k=5, conversation_history=None, include_sources=True)`.
- `dataset.ingest_source(source, *, pipeline_id=None)` — `source` is an id or a source builder.
- `dataset.ingest_files(paths, *, source_name=None, description=None)`.

### Sources / ingestion (`client.sources` is an alias of `client.ingestion`)

- `create(source)` / `create_source(source=None, *, name=None, source_type=None, config=None, description=None, metadata=None)`.
- `create_web(*, start_urls, name=None, max_depth=None, max_pages=None, allowed_domains=None, include_patterns=None, exclude_patterns=None, crawl_delay_seconds=None, include_assets=None, max_assets_per_page=None, sync_mode omitted via builder, description=None, metadata=None, config_extra=None)`.
- `create_s3(*, bucket, name=None, region="us-east-1", prefix=None, sync_mode=None, access_key_id=None, secret_access_key=None, role_arn=None, endpoint_url=None, file_patterns=None, max_file_size_mb=None, …)`.
- `create_gcs(*, bucket, name=None, prefix=None, project_id=None, credentials_json=None, sync_mode=None, file_patterns=None, max_file_size_mb=None, …)`.
- `create_jira(*, cloud_id, name=None, access_token=None, project_keys=None, jql=None, include_comments=True, sync_mode=None, …)`.
- `create_confluence(*, cloud_id=None, base_url=None, name=None, auth_mode="basic", username=None, api_token=None, oauth_credentials=None, spaces=None, include_attachments=False, sync_mode=None, …)` — requires `cloud_id` or `base_url`.
- `create_google_drive(*, name=None, folder_ids=None, file_ids=None, auth_mode="oauth", oauth_credentials=None, include_shared_drives=None, sync_mode=None, service_account_json=None, credentials_json=None, …)`.
- `create_file_upload(*, name="vectoramp-python-upload", storage_provider="s3", sync_mode="full", …)`.
- `list_sources(*, limit=50, offset=0)`, `get_source(source_id)`.
- `start_job(*, source_id, dataset_id, pipeline_id=None)`, `list_jobs(*, dataset_id=None, limit=50, offset=0)`, `get_job(job_id)`, `retry_job(job_id)`.
- `ingest_files(*, dataset_id, paths, source_name=None, description=None)`, `init_upload(source_id, files)`, `complete_upload(source_id, *, job_id, file_ids)`.

Builder classes: `WebSource`, `S3Source`, `GCSSource`, `GoogleDriveSource`, `JiraSource`, `ConfluenceSource`, `FileUploadSource`, `GenericSource` (escape hatch). `sync_mode` is omitted unless set, so the server default (`"incremental"`) applies.

### Schedules (`client.schedules`)

- `list(*, limit=50, offset=0)`, `get(schedule_id)`.
- `create(*, source_id, dataset_id, cron, timezone=None, pipeline_id=None, enabled=None, name=None, metadata=None)`.
- `update(schedule_id, *, cron=None, timezone=None, pipeline_id=None, enabled=None, name=None, metadata=None)` — only passed fields change.
- `delete(schedule_id)`, `trigger(schedule_id)`.

### Intelligence (`client.intelligence`)

- `query(query, *, dataset_id=None, top_k=5, conversation_history=None, include_sources=True)`.
- `stream(query, *, …)` — iterator of SSE chunks.
- `create_session(*, title=None, workspace_id=None, dataset_id=None, metadata=None)`.
- `list_sessions(*, limit=50)`, `get_session(session_id)`.
- `append_message(session_id, *, role, content, metadata=None)`, `list_messages(session_id, *, limit=100)`.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
