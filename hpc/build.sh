#!/usr/bin/env bash
# Build the STEP environment image and push it to GHCR.
#
#   ./hpc/build.sh           # build + push :<short-sha> and :latest
#   ./hpc/build.sh --no-push # build only, for local testing
#
# Needs docker and a gh token with write:packages (gh auth refresh -h github.com -s write:packages).
# The package must be PUBLIC (https://github.com/users/ilsenatorov/packages/container/step/settings): execute
# nodes pull anonymously, and a private package fails every job with "manifest unknown".
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/ilsenatorov/step}"
cd "$(dirname "$0")/.."

# Tag by content (lock file + Dockerfile), not the commit: the image only depends on those two
TAG="env-$(cat uv.lock hpc/Dockerfile | sha256sum | cut -c1-12)"

echo "==> building ${IMAGE}:${TAG}"
docker build -f hpc/Dockerfile -t "${IMAGE}:${TAG}" -t "${IMAGE}:latest" .

if [[ "${1:-}" == "--no-push" ]]; then
    echo "==> built, not pushing (--no-push)"
    exit 0
fi

if ! gh auth status 2>&1 | grep -q "write:packages"; then
    echo "ERROR: gh token lacks write:packages; run: gh auth refresh -h github.com -s write:packages" >&2
    exit 1
fi
gh auth token | docker login ghcr.io -u "$(gh api user --jq .login)" --password-stdin

docker push "${IMAGE}:${TAG}"
docker push "${IMAGE}:latest"

# Submit files pin the exact tag: execute nodes cache images and do not re-pull a tag they already have, so a
# rebuilt :latest would silently keep running the old image there
sed -i "s#^docker_image .*#docker_image            = ${IMAGE}:${TAG}#" hpc/common.sub
echo "Pushed ${IMAGE}:${TAG} (and :latest) and pinned it in hpc/common.sub - commit that and git pull on the cluster."
