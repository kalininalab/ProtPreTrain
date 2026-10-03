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

# Tag by the lock file's content, not the commit: the image only depends on the environment
TAG="env-$(sha256sum uv.lock | cut -c1-12)"

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
echo "Pushed ${IMAGE}:${TAG} and :latest. Submit files use :latest; pin with docker_image = ${IMAGE}:${TAG}."
