# Changelog

All notable changes to this project will be documented in this file.

This project follows semantic versioning.

## [Unreleased]

### Changed

- **Breaking:** Intelligence queries now scope with `dataset_ids` (a list) instead of the retired
  `dataset_id`. `client.ask(...)`, `client.ask_stream(...)`, `client.intelligence.query(...)` and
  `client.intelligence.stream(...)` take `dataset_ids=[...]`; `dataset.ask(...)` scopes itself to
  its own dataset. `POST /intelligence/query` answers any request still carrying `dataset_id` with
  a 400, so the field is never sent.
- The `dataset_id="all"` sentinel is retired. Omit `dataset_ids` (or pass an empty list) to search
  every dataset the API key can see.

## [0.4.0] - 2026-08-20

### Added

- Add `GitHubSource` and `GitLabSource` typed source builders.
- Add `client.sources.create_github(...)` and `client.sources.create_gitlab(...)`.

## [0.3.0] - 2026-07-20

### Added

- Add typed metadata-schema fields when creating datasets.
- Add metadata-schema merge/patch and full replacement operations.
- Document create, merge, and replace schema workflows.

## [0.2.0] - 2026-07-14

### Added

- Add dataset/vector deletion helpers for deleting vectors by id.
- Add organization secret helpers and OpenAI-backed dataset creation convenience flow.
- Document OpenAI secret setup and vector deletion examples.

## [0.1.0] - 2026-07-02

### Added

- Initial public-ready package baseline for VectorAmp SDK/CLI migration to GitHub.
- GitHub Actions CI workflow.
