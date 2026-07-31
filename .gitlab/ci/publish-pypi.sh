#!/bin/bash
# Manual PyPI publish, run inside podman on an allocated node.
# Requires the PYPI_TOKEN GitLab CI/CD variable.
set -ex

PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"
PODMAN="podman --root $PODMAN_STORE --runroot $PODMAN_RUNROOT"

$PODMAN run --rm -v "$PWD:/ws" -w /ws -e PYPI_TOKEN docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet build twine cmake ninja
  python -m build
  twine upload --username __token__ --password "$PYPI_TOKEN" dist/*
'
