# Contributing to argus-engine

Thanks for your interest in contributing!

## Development

```bash
cargo build                              # default: opencl + profile
cargo test --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

Requires **Rust ≥ 1.94** (edition 2024); run `rustup update stable` if the first build
reports a `requires rustc 1.94` version error.

On Linux, building the default (OpenCL) feature needs `ocl-icd-opencl-dev`. A GPU is
not required to build. GPU correctness and on-device benchmarks require real hardware
and are run manually (CI builds only).

## Conventions

- **Module file style:** no `mod.rs` — a directory module's root is the sibling
  `foo.rs`. New/moved modules must follow this.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `type(scope): subject`, imperative mood.
- See [AGENTS.md](AGENTS.md) for the full working agreement and the domain vocabulary
  in [CONTEXT.md](CONTEXT.md).

## Submitting changes

1. **Fork** the repo and clone your fork.
2. **Branch** off `main`: `git checkout -b fix/short-description` (don't work on `main`).
3. **Keep changes surgical** — touch only what the task requires; match the surrounding
   style; don't reformat unrelated code.
4. **Verify locally** before pushing:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace -- -D warnings
   cargo test --workspace
   ```
5. **Open a PR** against `main` with a Conventional Commit title and a clear description;
   link any related issue. CI (fmt, clippy, build matrix, tests, cargo-deny) must pass.
6. Be responsive to review feedback.

## Extending the engine

New KV-cache stages/formats/read-stages are added as separate crates under
`crates/techniques/` that self-register via `linkme` — no engine-core edits needed.
The `example-*` crates are working templates.

## License of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you shall be dual licensed under **MIT OR Apache-2.0**,
without any additional terms or conditions.
