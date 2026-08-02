#!/bin/bash
# ci.yml -> wheel. Runs on the node, not in a container: cibuildwheel drives
# podman itself. The CIBW_* settings mirror the wheel job's env.
set -eo pipefail
cd "$CI_PROJECT_DIR"
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true

# The rootless store defaults to $HOME, which on NFS cannot hold image-layer
# xattrs. A PATH wrapper is needed: podman 4.9 ignores CONTAINERS_STORAGE_CONF
# for the rootless graphroot.
STATE=/var/tmp/$USER/dftracer-utils-wheels
mkdir -p "$STATE/bin" "$STATE/storage" "$STATE/run" "$STATE/ccache"
cat >"$STATE/bin/podman" <<WRAP
#!/usr/bin/env bash
exec "$(command -v podman)" --root "$STATE/storage" --runroot "$STATE/run" "\$@"
WRAP
chmod +x "$STATE/bin/podman"
export PATH="$STATE/bin:$PATH"

# scikit-build-core never runs setup.py, so the .postN.dev0 scheme dftracer and
# pydftracer set there cannot be expressed here: a custom version_scheme needs a
# registered entry point. Compute it and hand it over instead, so all three
# publish the same shape. pyproject's no-guess-dev stays as the local fallback.
described=$(git describe --tags --long --match 'v*')
tag=${described%-*-g*}
distance=${described#"$tag"-}
distance=${distance%-g*}
tag=${tag#v}
if [ "$distance" = "0" ]; then
  VERSION="$tag"
else
  VERSION="$tag.post$distance.dev0"
fi
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DFTRACER_UTILS="$VERSION"
echo "building version $VERSION"

export CIBW_BUILD="${CIBW_BUILD:-cp38-* cp39-* cp310-* cp311-* cp312-* cp313-* cp314-*}"
export CIBW_SKIP="*-win32 *-manylinux_i686 *-musllinux_* *-manylinux_ppc64le *-manylinux_s390x"
export CIBW_CONFIG_SETTINGS="cmake.define.DFTRACER_UTILS_LOCAL_PACKAGES=OFF"
export CIBW_BEFORE_ALL_LINUX="yum install -y epel-release ccache || dnf install -y epel-release ccache || true"
export CIBW_BEFORE_BUILD="ccache -s || echo 'WARNING: ccache unavailable - build is uncached'"
export CIBW_CONTAINER_ENGINE="podman; create_args: --volume=$STATE/ccache:/ccache"
export CIBW_ENVIRONMENT_LINUX="SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DFTRACER_UTILS=$VERSION CCACHE_DIR=/ccache CCACHE_BASEDIR=/project CCACHE_COMPILERCHECK=content CCACHE_NOHASHDIR=true CCACHE_MAXSIZE=900M CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,pch_defines,system_headers,locale"

# Same release as the pinned action, read from it rather than repeated: 4.x
# dropped the cp38 that CIBW_BUILD above still asks for.
CIBW_VERSION="${CIBW_VERSION:-$(grep -oE 'pypa/cibuildwheel@v[0-9]+\.[0-9]+\.[0-9]+' \
  .github/workflows/ci.yml | head -1 | sed 's/.*@v//')}"
[ -n "$CIBW_VERSION" ] || { echo "ERROR: no pinned cibuildwheel in .github/workflows/ci.yml"; exit 1; }
echo "using cibuildwheel $CIBW_VERSION"

rm -rf wheelhouse
python3 -m venv "$STATE/venv"
"$STATE/venv/bin/pip" install -q --upgrade pip "cibuildwheel==$CIBW_VERSION"
"$STATE/venv/bin/cibuildwheel" --output-dir wheelhouse
ls -1 wheelhouse
