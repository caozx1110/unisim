# Release Runbook

[English](release.md) | [中文](../zh/release.md)

This package is published as `unisim-core` and imported as `unisim`. Production PyPI publishing is automated by `.github/workflows/release.yml` after a matching version tag is pushed and the tagged commit has a successful `ci.yml` run. The workflow publishes the source distribution plus one universal pure-Python wheel using GitHub trusted publishing (OIDC); no PyPI token is stored in the repository. It does not select a Python-version or OS matrix: the base package is pure Python, depends only on NumPy, and ships exactly one `unisim_core-<version>-py3-none-any.whl` that covers every supported CPython 3.10–3.13 platform. The universal base wheel does not imply every optional backend is available on every platform — optional engine SDKs remain extras and keep their own per-platform support matrices — and the wheel complements the sdist, which continues to be published. After a successful PyPI publish, the workflow's `github-release` job creates the GitHub Release for the tag with the sdist and wheel attached and the matching `CHANGELOG.md` section as its notes.

The repository administrator must configure a PyPI trusted publisher once with owner `unilabsim`, repository `unisim`, workflow `release.yml`, and environment `pypi`. Keep the GitHub environment protected with the desired reviewer rules.

## TestPyPI

1. Update `[project].version` in `pyproject.toml`, `CHANGELOG.md`, and the affected English and Chinese documentation.
2. Run `make check` and `make package`.
3. Run `uvx --from twine twine check dist/unisim_core-<version>.tar.gz dist/unisim_core-<version>-py3-none-any.whl`.
4. Upload the sdist and wheel to TestPyPI using credentials from `~/.pypirc`; never print, copy, or commit that file.
5. Install the exact version in an isolated environment with TestPyPI as the package index — including a wheel-only install (`uv pip install --only-binary :all:`) — and verify `import unisim` without loading `unilab` or engine SDK modules.
6. Record the published URLs, artifact hashes, gate results, and rollback decision in the release PR or release tracking issue.

Use the package-owned distribution and import names in all commands, and never print, copy, or commit `~/.pypirc`.

## Production PyPI

1. Confirm that the working tree is clean, `make check` passes, and the changelog contains the release entry.
2. Wait for the three cross-platform `ci.yml` test jobs, the `typecheck` job, and the pre-release package job (which now also waits on `typecheck`) to pass for the commit you will tag.
3. Create and push an annotated tag that exactly matches the package version, for example `git tag -a v0.1.13 -m "release: unisim-core 0.1.13"` followed by `git push origin v0.1.13`. The release workflow verifies the tag and successful CI run, builds and smoke-tests one sdist and one universal pure-Python wheel on `ubuntu-latest` — the wheel is installed with `uv pip install --only-binary :all:` so the smoke test can never fall back to the sdist — and publishes both after the checks succeed. The runner only executes the build and does not constrain the artifacts. There is no release-time Python or OS matrix; manual dispatch runs verification only and cannot publish.
4. Inspect the workflow, the PyPI artifact metadata, and the created GitHub Release. A failed run may be re-run, but an already published version must never be overwritten. Fix the source and release a new patch version when an artifact is wrong.

## Compatibility and removal policy

Breaking removals are hard cuts at release boundaries: minor releases may cut internal surfaces, while the stable public names pinned for UniLab consumers (`unisim.SimBackend`, `unisim.create_backend`, `unisim.ADAPTER_SPECS`, the exported adapter classes, and `unisim.backend.subprocess_ipc`) are removed only at a major release. A published release is never retroactively edited. Two bounded compatibility windows are allowed:

- Warn-and-ignore shims, such as removed factory options that still parse but warn, live until the next major release; the removal milestone is named in `CHANGELOG.md` when the shim is introduced.
- Reader compatibility for persisted data, such as legacy pickle state keys that cross process boundaries, is documented in code with the producing version window and a named removal boundary — by default the next major release.

An announced "removed next major" must name the version, and the removal ships no earlier than its named milestone.
