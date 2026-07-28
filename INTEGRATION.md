# `all-open-prs-integration` — integration branch

**This branch is a derived artifact, not a place for original development.**
It is upstream `main` at a pinned base commit with a set of open pull requests
merged on top. Every *code* change traces to one of the PRs listed below; the
only non-PR commit is the one that adds this manifest and
`scripts/build-integration.sh`. Regenerate the branch with that script.

- Upstream: <https://github.com/nerfstudio-project/nerfstudio>
- Fork: <https://github.com/f-dy/nerfstudio>
- Base: `main` @ `50e0e3c7` ("Add image tiling option to ColmapDataParser to use less memory", #3685)

## Included PRs (in merge order)

| PR | Title | Source branch | Head |
|----|-------|---------------|------|
| [#3775](https://github.com/nerfstudio-project/nerfstudio/pull/3775) | ci: build docs on Python 3.11 (fastjsonschema dropped 3.9) | `f-dy:fix-docs-python-version` | `10544dd` |
| [#3774](https://github.com/nerfstudio-project/nerfstudio/pull/3774) | fix: coerce numpy scalars in colmap `write_next_bytes` for NumPy 2.x | `f-dy:fix-numpy2-struct-packing` | `7c93989` |
| [#3612](https://github.com/nerfstudio-project/nerfstudio/pull/3612) | colmap_dataparser: COLMAP `k4` is valid for the fisheye model | `f-dy:k4-allowed-for-fisheye` | `4b828b1` |
| [#3773](https://github.com/nerfstudio-project/nerfstudio/pull/3773) | Record antialiased mode as a `SplatRenderMode` PLY comment on export | `f-dy:splatrendermode-ply-comment` | `b7be448` |
| [#2059](https://github.com/nerfstudio-project/nerfstudio/pull/2059) | Fix `ns-render interpolate` (render input views, crop box, `fixed_intrinsics`) | `f-dy:nsrender-interpolate-crop` | `10747be` |
| [#3653](https://github.com/nerfstudio-project/nerfstudio/pull/3653) | Support `torch-2.8.0+cu128` (Blackwell / RTX50xx) | `nicogorlo:main` | `80003ca` |

## Provenance

Because the branch is built purely by merging the PRs above, git history is the
source of truth:

```bash
git log --merges --first-parent --oneline <base>..all-open-prs-integration
```

shows exactly one merge per PR, each referencing the PR number and URL.

## Regenerate

```bash
./scripts/build-integration.sh
```

This recreates `all-open-prs-integration` from the pinned base by merging each PR
ref. Confirm the code tree is unchanged before force-pushing:

```bash
git diff --stat <old-tip> all-open-prs-integration -- ':!INTEGRATION.md' ':!scripts/build-integration.sh'
```
