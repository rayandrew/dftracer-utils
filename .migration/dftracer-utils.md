# Migration Plan: dftracer-utils

Selected: 2026-07-30. Source: git@github.com:llnl/dftracer-utils.git (develop). Target: ssh://git@czgitlab.llnl.gov:7999/dftracer/dftracer-utils.git (repo exists on czgitlab).

## Steps (subagent fills findings + status)

1. [x] Latest develop (dfanalyzer freshly cloned; utils pulled)
2. [x] gitlab remote added + verified
3. [x] Convert .github/workflows → .gitlab-ci.yml on corona batch runner (.corona-batch template); publish jobs manual
4. [x] pages job for Sphinx docs (develop + temp gitlab-migration rule)
5. [x] In-place GitLab URL switch for deps — **not needed** (no dftracer-group deps; see findings)
6. [x] Local tests (YAML OK; sphinx build succeeded, 37 warnings; build/pytest smoke deferred — see findings)
7. [x] Commit on gitlab-migration; sync .migration/ into repo
8. [x] Push develop, tags, gitlab-migration
9. [ ] Pipeline: <https://czgitlab.llnl.gov/dftracer/dftracer-utils/-/pipelines>
10. [ ] User merges after green pipeline

## Findings

- Workflows converted: `ci.yml` (rocksdb prebuild + os/python test matrix, valgrind shards, web/npm, wheel/cibuildwheel) → single `test` job on `.corona-batch` running `make test` + `make test-py` with gcc/11.2.1 + python/3.13.2 + mvapich2/2.3.7 modules (matrices collapsed, coveralls/paths-filter/web dropped — GitHub-only, noted in header comment). `format-check.yaml` → `lint` job (pip-installs `clang-format==19.*` + `ruff`, runs `scripts/formatting/check-formatting.sh` + `ruff check`/`ruff format --check`). `python-publish.yaml` → `publish-pypi`, `when: manual` on tags; needs CI/CD variable `PYPI_TOKEN` (listed in `.gitlab-ci.yml` header). No docker on corona → cibuildwheel replaced by `python -m build` + twine.
- Sphinx source lives in `docs/source` (not `docs/`); pages job uses `sphinx-build -b html docs/source public` with `pip install -r docs/requirements.txt`. Doxygen/breathe steps in conf.py degrade gracefully when doxygen XML is absent.
- Dependency scan: NO dftracer-group GitHub deps (cpp-logger/brahma/dftracer/dfanalyzer/dftracer-agents) in pyproject.toml, setup.py, CMakeLists, or cmake/modules/Dependencies.cmake — all CPM deps are third-party. Only `github.com/LLNL/dftracer-utils` self-references (project metadata/docs). **Zero in-place changes made.**
- Build/pytest smoke not run locally: `make test` compiles RocksDB + full C++ tree from source (hours on a login node; GitHub CI uses prebuilt caches); left to the corona pipeline.

## Executed changes (what to undo on revert)

- Added `gitlab` remote: ssh://git@czgitlab.llnl.gov:7999/dftracer/dftracer-utils.git
- Branch `gitlab-migration` (from develop) with commits: `.gitlab-ci.yml` (ci commit) and `.migration/` copies (docs/migration commit) — SHAs in status log.
- Pushed to gitlab: `develop`, all tags, `gitlab-migration`. Nothing pushed to origin/GitHub; `.github/` untouched.
- **In-place changes: NONE** (nothing to fix on revert beyond the additive removals). Temp pages rule for `gitlab-migration` branch in `.gitlab-ci.yml` must be removed after merge.

## Status log

- 2026-07-30: plan created.
- 2026-07-30: executed. `.gitlab-ci.yml` written (lint/test/publish-pypi/pages on `.corona-batch`); YAML validated; sphinx built locally (success, 37 warnings); no dep URL switches needed; committed on `gitlab-migration`; pushed develop + tags + gitlab-migration to czgitlab.
