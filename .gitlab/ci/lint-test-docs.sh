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

# --user 0:0: container root maps to the host user under rootless podman, so
# the bind-mounted checkout stays readable even for images with a non-root USER.

# Lint: clang-format 19 + ruff (format-check.yaml).
$PODMAN run --rm --user 0:0 -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet "clang-format==19.*" ruff
  ./scripts/formatting/check-formatting.sh "$(command -v clang-format)"
  ruff check python/ tests/python/
  ruff format --check python/ tests/python/
'

# Build + test (ci.yml). ubuntu:24.04 matches GitHub's ubuntu-latest: 22.04
# ships CMake 3.22, too old for this repo's CMakePresets.json (version 6 needs
# CMake >= 3.25) -> "Could not read presets: Unrecognized version field".
# APT::Sandbox::User=root: rootless podman has no mapped _apt uid, so apts
# privilege drop fails with "setgroups (22: Invalid argument)".
$PODMAN run --rm --user 0:0 -v "$PWD:/ws" -w /ws -e CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  docker.io/library/ubuntu:24.04 bash -ec '
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o APT::Sandbox::User=root update -qq
  apt-get -o APT::Sandbox::User=root install -y -qq \
    build-essential cmake ninja-build ccache pkg-config git \
    zlib1g-dev libzstd-dev libsqlite3-dev \
    libmpich-dev mpich \
    python3 python3-pip python3-venv python3-dev
  cmake --version
  # Ubuntu 24.04 marks its python as externally managed (PEP 668); this is a
  # throwaway container, so installing into it directly is fine.
  pip3 install --quiet --break-system-packages pytest
  # "make test" == these three commands. On Debian/Ubuntu GNUInstallDirs
  # resolves CMAKE_INSTALL_LIBDIR to the multiarch lib/x86_64-linux-gnu, but
  # the package-discovery tests only look in <prefix>/lib and <prefix>/lib64,
  # so all 15 fail with "...Config.cmake not found". Pinning the variable does
  # not work: a subproject re-includes GNUInstallDirs after changing the
  # prefix, which FORCE-overwrites it back to multiarch. Instead, pre-create
  # the install tree and point lib/{cmake,pkgconfig} at the multiarch dirs.
  cmake --preset tests
  cmake --build --preset tests
  LIBDIR=$(sed -n "s/^CMAKE_INSTALL_LIBDIR:PATH=//p" build/build-tests/CMakeCache.txt)
  PFX=$PWD/build/build-tests/test_install
  echo "install libdir: $LIBDIR"
  if [ "$LIBDIR" != "lib" ] && [ "$LIBDIR" != "lib64" ]; then
    mkdir -p "$PFX/$LIBDIR/cmake" "$PFX/$LIBDIR/pkgconfig" "$PFX/lib"
    ln -sfn "$PFX/$LIBDIR/cmake" "$PFX/lib/cmake"
    ln -sfn "$PFX/$LIBDIR/pkgconfig" "$PFX/lib/pkgconfig"
  fi
  ctest --preset tests
  make test-py
'

# Docs (sphinx source lives in docs/source).
$PODMAN run --rm --user 0:0 -v "$PWD:/ws" -w /ws docker.io/library/python:3.11 bash -ec '
  pip install --quiet --upgrade pip
  pip install --quiet -r docs/requirements.txt
  sphinx-build -b html docs/source public
'
