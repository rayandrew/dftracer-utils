#!/bin/bash
# Usage: run-in-podman.sh <image> <phase-script> [args...]
# Runs a CI phase inside podman on the allocated node, mirroring the images the
# GitHub workflows use.
set -ex

IMAGE=$1
shift

# Rootless podman needs node-local storage; overlayfs does not work on NFS.
PODMAN_STORE=/var/tmp/$USER/podman-root
PODMAN_RUNROOT=/var/tmp/$USER/podman-run
mkdir -p "$PODMAN_STORE" "$PODMAN_RUNROOT"

# RocksDB is built once by the rocksdb phase and shared with every other one.
# Mounted at the path build_rocksdb.sh defaults to, so the prefix matches the
# one the GitHub jobs bake into their cache.
DEPS_DIR=/var/tmp/$USER/dftracer-deps
mkdir -p "$DEPS_DIR"

# Mounted at the same path so --find-links reads the same inside and out.
DFTRACER_DIST="${DFTRACER_DIST:-/usr/workspace/dldl/dftracer/distributions}"
DIST_MOUNT=()
if [ -d "$DFTRACER_DIST" ]; then
  DIST_MOUNT=(-v "$DFTRACER_DIST:$DFTRACER_DIST:ro")
fi

# --user 0:0: container root maps to the host user, so the bind-mounted
# checkout stays readable for images with a non-root USER.
podman --root "$PODMAN_STORE" --runroot "$PODMAN_RUNROOT" run --rm \
  --user 0:0 \
  -v "$PWD:/ws" -w /ws \
  -v "$DEPS_DIR:/opt/dftracer-deps" \
  "${DIST_MOUNT[@]}" \
  -e CI_PROJECT_DIR=/ws \
  -e DFTRACER_DIST \
  -e CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -e ROCKSDB_VERSION \
  "$IMAGE" bash "$@"
