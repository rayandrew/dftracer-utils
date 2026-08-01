#!/bin/bash
# ci.yml -> rocksdb. Builds RocksDB once into the shared deps mount; the phases
# that follow link against it rather than each building their own copy. The
# script is a no-op when the prefix is already populated.
set -eo pipefail
cd "$CI_PROJECT_DIR"
source .gitlab/ci/toolchain.sh
./scripts/ci/build_rocksdb.sh
