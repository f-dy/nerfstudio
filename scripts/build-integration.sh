#!/usr/bin/env bash
# Rebuild the `all-open-prs-integration` branch from upstream main + open PRs.
#
# This branch is a DERIVED ARTIFACT. Do not develop on it directly: add/adjust
# the PR list below and re-run this script. See INTEGRATION.md for the manifest.
#
# After rebuilding, re-apply the manifest commit (INTEGRATION.md + this script)
# and verify the code tree is unchanged before force-pushing:
#   git diff --stat <old-tip> HEAD -- ':!INTEGRATION.md' ':!scripts/build-integration.sh'
set -euo pipefail

BRANCH="all-open-prs-integration"
# Pinned upstream base (nerfstudio-project/nerfstudio main). Bump deliberately.
BASE="50e0e3c70c775e89333256213363badbf074f29d"  # #3685
UPSTREAM="${UPSTREAM_REMOTE:-upstream}"           # nerfstudio-project/nerfstudio
FORK="${FORK_REMOTE:-origin}"                     # f-dy/nerfstudio

# PRs hosted on the f-dy fork, as "PR:branch" (merged in this order).
FORK_PRS=(
  "3775:fix-docs-python-version"
  "3774:fix-numpy2-struct-packing"
  "3612:k4-allowed-for-fisheye"
  "3773:splatrendermode-ply-comment"
  "2059:nsrender-interpolate-crop"
)
# External PRs, merged via upstream GitHub pull refs, as "PR".
EXTERNAL_PRS=("3653")

REPO_URL="https://github.com/nerfstudio-project/nerfstudio/pull"

git fetch "$UPSTREAM"
git fetch "$FORK"
git checkout -B "$BRANCH" "$BASE"

for pb in "${FORK_PRS[@]}"; do
  pr="${pb%%:*}"; br="${pb##*:}"
  git fetch "$FORK" "$br"
  git merge --no-ff -m "Merge PR #${pr} (${br}) from f-dy: ${REPO_URL}/${pr}" "$FORK/$br"
done

for pr in "${EXTERNAL_PRS[@]}"; do
  git fetch "$UPSTREAM" "pull/${pr}/head:pr-${pr}"
  git merge --no-ff -m "Merge PR #${pr}: ${REPO_URL}/${pr}" "pr-${pr}"
done

echo "Rebuilt ${BRANCH}. PR merges:"
git --no-pager log --merges --first-parent --oneline "${BASE}..HEAD"
