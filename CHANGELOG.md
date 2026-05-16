# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2026-05-17]

### Fixed

- **MSVC CRT linkage conflict in debug builds**: `ort-sys` precompiled static libraries use `/MD` (dynamic CRT), while `esaxx-rs` and `onig_sys` (compiled via the `cc` crate) default to `/MT` (static CRT). MSVC prohibits mixing `/MD` and `/MT` in a single binary, causing `LNK2005` / `LNK1169` errors in debug mode.
  - Root cause: Debug CRT (`/MTd` / `/MDd`) instantiates far more template symbols (checked iterators, debug heap at `_ITERATOR_DEBUG_LEVEL > 0`) than Release CRT, making symbol collisions inevitable. Release mode "passes" only because the overlap is smaller — it is not truly compatible.
  - Fix (`.cargo/config.toml`):
    1. `rustflags = ["-C", "target-feature=-crt-static"]` — makes Rust itself link against the dynamic CRT (`/MD`), matching `ort-sys`
    2. `CFLAGS` / `CXXFLAGS = { value = "/MD", force = true }` — forces the `cc` crate to also compile C/C++ sources with `/MD`, ensuring `esaxx-rs` and `onig_sys` use the same CRT variant
    - `force = true` is required to override any pre-existing `CFLAGS`/`CXXFLAGS` from the shell environment
