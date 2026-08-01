#!/usr/bin/env bash
#
# Print the RocksDB version the build pins. This is the only place CI reads it
# from: the version is part of the install prefix, so a stale copy elsewhere
# would silently send find_package looking at a path the prebuild never wrote.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
deps="${root}/cmake/modules/Dependencies.cmake"

version=$(grep -A2 'set(DFTRACER_UTILS_ROCKSDB_VERSION' "$deps" |
	grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

[ -n "$version" ] || {
	echo "ERROR: no DFTRACER_UTILS_ROCKSDB_VERSION in $deps" >&2
	exit 1
}
echo "$version"
