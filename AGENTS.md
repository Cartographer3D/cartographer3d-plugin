# AGENTS.md — Cartographer3D Plugin

## Project Overview

A Python plugin for [Klipper](https://github.com/Klipper3d/klipper) 3D printer firmware that integrates **Cartographer3D** probe hardware. Provides contactless scanning, contact probing, bed mesh calibration, temperature compensation, and G-code macros.

**Must support three firmware targets:**
- [Klipper (mainline)](https://github.com/Klipper3d/klipper)
- [Kalico](https://github.com/KalicoCrew/kalico) — Klipper fork with extended APIs
- [SimpleAF](https://github.com/pellcorp/klipper) — community fork, largely follows mainline Klipper; `scipy` is **not available** in this environment

Runtime detection lives in `src/cartographer/runtime/environment.py`. Adapter-specific code is isolated under `src/cartographer/adapters/{klipper,kalico}/`. The loader in `src/cartographer/runtime/loader.py` picks the correct adapter at startup. Any new firmware-specific behaviour must go through this adapter pattern — never import Klipper/Kalico modules directly in core logic.

## Architecture

```
src/cartographer/
├── extra.py              ← Klipper entry point (load_config)
├── core.py              ← PrinterCartographer orchestrator
├── adapters/            ← Firmware-specific implementations
│   ├── klipper/         ← Mainline + pellcorp Klipper
│   └── kalico/          ← Kalico fork
├── runtime/             ← Environment detection & loader
├── interfaces/          ← Abstract contracts (Printer, Mcu, Toolhead, etc.)
├── probe/               ← Probe modes (scan, touch) + calibration models
├── macros/              ← G-code command implementations
├── config/              ← Configuration parsing & validation
├── coil/                ← Temperature compensation
├── mcu/                 ← MCU command protocol + streaming
├── lib/                 ← Utilities (filters, CSV, scipy wrappers)
└── toolhead.py          ← Backlash-compensating toolhead wrapper
```

Key flow: Klipper calls `extra.py:load_config()` → detects environment → creates adapters → instantiates `PrinterCartographer` → registers macros and endstop.

## Tech Stack

| Area | Tool |
|------|------|
| Language | Python 3.8–3.13 |
| Package manager | `uv` |
| Build backend | `hatchling` + `hatch-vcs` |
| Linter/Formatter | `ruff` (v0.14+) |
| Type checker | `basedpyright` (target: 3.8) |
| Tests | `pytest` + `pytest-mock` |
| Task runner | `justfile` |
| VCS | Jujutsu (`jj`) over Git |

## Commands

```bash
# Install
uv sync --locked --all-extras --dev

# Lint
uv run --only-group lint ruff check
uv run --only-group lint ruff format --check

# Format
uv run --only-group lint ruff format

# Type check
uv run basedpyright

# Test
uv run pytest
uv run pytest tests/test_core.py  # specific file

# Update lockfile after dependency changes
uv sync --dev --all-extras --all-groups
```

## Coding Standards

- **Line length**: 120 chars (docstring code blocks: 80)
- **Imports**: Absolute only — no relative imports (enforced by ruff `TID`)
- **Type hints**: Always. Use `from __future__ import annotations` in every file.
- **TYPE_CHECKING**: Imports only needed for type annotations go inside `if TYPE_CHECKING:` blocks
- **Strings**: f-strings preferred
- **Logging**: `logging.getLogger(__name__)`
- **Config objects**: dataclasses with validation
- **No suppression pragmas**: Do not use `# pyright: ignore`, `# type: ignore`, or `# noqa`. Fix the underlying issue instead.
- **No Klipper runtime imports in core**: All Klipper interactions go through abstract interfaces

## Testing

- All Klipper/printer interfaces are mocked — never import actual firmware code in tests
- Fixtures in `tests/conftest.py` provide standard mocks (toolhead, mcu, config, probe, etc.)
- Custom mocks in `tests/mocks/`
- Tests must pass on Python 3.8+

## Critical Constraints

1. **Python 3.8 compatibility** — No walrus operator in non-guarded code, no `match` statements, use `typing_extensions` backports for newer typing features.
2. **No relative imports** — Ruff enforces this; use `from cartographer.module import X`.
3. **Adapter isolation** — Firmware-specific code never leaks into `core.py`, `probe/`, `macros/`, or `interfaces/`. If a feature behaves differently across firmware targets, add it to the adapter/integrator layer.
4. **Klipper is not a dependency** — It's the host runtime. Type stubs live in `typings/`. Tests mock everything.
5. **scipy is optional** — Not available on SimpleAF; may be absent elsewhere. Guarded behind `try/except ImportError` or the optional dependency group. All core functionality must work without it.

## CI/CD

- PRs: auto-labeled via `.github/scripts/auto-label-pr.js`
- Integration: tests run in CI (`.github/workflows/integration.yml`)
- Delivery: tag push (`v*`) triggers build → PyPI publish → GitHub Release with Sigstore signatures
- Release notes auto-generated from PR labels (see `.github/release.yml`)

## PR Checklist

1. `uv run --only-group lint ruff check` passes
2. `uv run --only-group lint ruff format --check` passes
3. `uv run basedpyright` passes
4. `uv run pytest` passes
5. New functionality has tests
6. No relative imports, no untyped public APIs
