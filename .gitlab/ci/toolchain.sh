# Sourced by every phase script. Runs inside a podman container, no LC modules.
set -eo pipefail

# APT::Sandbox::User=root: rootless podman has no mapped _apt uid, so apt's
# privilege drop fails with "setgroups (22: Invalid argument)".
export DEBIAN_FRONTEND=noninteractive
apt-get -o APT::Sandbox::User=root update -qq
# OpenMPI, matching the github valgrind jobs: valgrind-mpi.supp is written
# against OpenMPI/PMIx, and under mpich every rank came up as a singleton.
apt-get -o APT::Sandbox::User=root install -y -qq \
  build-essential cmake ninja-build ccache pkg-config git \
  curl ca-certificates \
  zlib1g-dev libzstd-dev libsqlite3-dev libopenmpi-dev openmpi-bin \
  python3 python3-pip python3-venv python3-dev "$@"
cmake --version

# Read from the CMake pin rather than repeated here: build_rocksdb.sh and the
# prefix below have to agree with what the build will actually look for.
ROCKSDB_VERSION="${ROCKSDB_VERSION:-$(scripts/ci/rocksdb_version.sh)}"
export ROCKSDB_VERSION
echo "RocksDB pinned at ${ROCKSDB_VERSION}"

# Built once by the rocksdb phase; every other phase links against it instead
# of rebuilding it in its own container.
_rocksdb_prefix="/opt/dftracer-deps/rocksdb-${ROCKSDB_VERSION}"
if [ -x "${_rocksdb_prefix}/bin/ldb" ]; then
  export DFTRACER_UTILS_ROCKSDB_PREFIX="${_rocksdb_prefix}"
  echo "using prebuilt RocksDB at ${_rocksdb_prefix}"
fi
