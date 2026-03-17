# CLAUDE.md

## Project Overview

A DataFusion extension crate for vector similarity search (ANN) using USearch HNSW indices. Provides optimizer rules, UDFs (l2_distance, cosine_distance, negative_dot), and pluggable lookup providers (Parquet, SQLite) for retrieving non-embedding columns by key.

## Git Commits

Use conventional format: `<type>(<scope>): <subject>` where type = feat|fix|docs|style|refactor|test|chore|perf. Subject: 50 chars max, imperative mood ("add" not "added"), no period.

## Pre-Push Checklist

Always run these before pushing:

1. `cargo fmt --check` — fix any formatting issues
2. `cargo clippy -- -D warnings` — no warnings allowed
3. `cargo test --features sqlite-provider` — all tests must pass (sqlite-provider feature needed for SQLite provider tests)
