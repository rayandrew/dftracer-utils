#!/bin/bash
# Runs ON the allocated compute node (via
#   flux proxy <jobid> flux run -N 1 bash .gitlab/ci/lint-test-docs.sh)
# and executes the CI inside podman containers, mirroring the GitHub Actions
# workflows (ubuntu for lint/build/test, python:3.11 for docs).
set -ex

# Rootless podman needs node-local storage (overlayfs does not work on NFS).
PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"
PODMAN="podman --root $PODMAN_STORE --runroot $PODMAN_RUNROOT"

# Lint: clang-format 19 + ruff (format-check.yaml).
$PODMAN run --rm -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet "clang-format==19.*" ruff
  ./scripts/formatting/check-formatting.sh "$(command -v clang-format)"
  ruff check python/ tests/python/
  ruff format --check python/ tests/python/
'

# Build + test (ci.yml). APT::Sandbox::User=root: rootless podman has no mapped
# _apt uid, so apts privilege drop fails with "setgroups (22: Invalid argument)".
$PODMAN run --rm -v "$PWD:/ws" -w /ws -e CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  docker.io/library/ubuntu:22.04 bash -ec '
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o APT::Sandbox::User=root update -qq
  apt-get -o APT::Sandbox::User=root install -y -qq \
    build-essential cmake ninja-build git python3 python3-pip python3-venv \
    libmpich-dev mpich pkg-config
  pip3 install --quiet --upgrade pip
  pip3 install --quiet pytest
  make test
  make test-py
'

# Docs (sphinx source lives in docs/source).
$PODMAN run --rm -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet -r docs/requirements.txt
  sphinx-build -b html docs/source public
'
