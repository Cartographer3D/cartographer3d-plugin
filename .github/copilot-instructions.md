# Copilot Instructions for Cartographer3D Plugin

## Project Overview

This is a Python plugin for Klipper that integrates with Cartographer3D, a 3D printing probe system. The plugin provides probe functionality, mesh calibration, temperature compensation, and various macros for 3D printer control.

## Project Structure

- `src/cartographer/` - Main source code
  - `adapters/` - Adapter implementations
  - `coil/` - Coil-related functionality including temperature compensation
  - `config/` - Configuration validation and management
  - `interfaces/` - Abstract interfaces for printer components
  - `lib/` - Utility libraries
  - `macros/` - G-code macros (bed mesh, probing, calibration, etc.)
  - `probe/` - Probe modes (scan mode, touch mode)
  - `runtime/` - Runtime environment detection (Klipper version/fork detection)
- `tests/` - Test suite with unit tests and mocks
- `scripts/` - Helper scripts
- `typings/` - Type stubs for dependencies

## Development Setup

### Prerequisites
- Python 3.8 or higher (targets 3.8-3.13)
- `uv` package manager (recommended) or `pip`

### Installation
```bash
# Install with uv (recommended)
uv sync --locked --all-extras --dev

# Or with pip
pip install -e ".[scipy]"
pip install -e ".[test,lint,typecheck,dev,scripts]"
```

## Build and Test Commands

### Linting
```bash
uv run --only-group lint ruff check
uv run --only-group lint ruff format --check
```

### Type Checking
```bash
uv run basedpyright
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_core.py
```

### Formatting
```bash
uv run --only-group lint ruff format
```

## Coding Standards

### Code Style
- **Line length**: 120 characters maximum
- **Docstring code blocks**: 80 characters maximum
- **Import style**: Absolute imports only (no relative imports)
- **Type hints**: Always use type hints with `from __future__ import annotations`
- **String formatting**: Prefer f-strings

### Linting Rules
This project uses Ruff with the following rule sets:
- pycodestyle (E)
- Pyflakes (F)
- pyupgrade (UP)
- flake8-future-annotations (FA)
- flake8-bugbear (B)
- flake8-simplify (SIM)
- isort (I)
- pep8-naming (N)
- flake8-errmsg (EM)
- flake8-logging (LOG, G)
- flake8-tidy-imports (TID)
- flake8-type-checking (TC)

### Type Checking
- Uses `basedpyright` (pyright fork)
- Target Python version: 3.8
- Type stubs are required for dependencies
- `TYPE_CHECKING` imports should be used to avoid runtime import cycles

### Testing Conventions
- Unit tests in `tests/` directory
- Mocks in `tests/mocks/` for external dependencies
- Use `pytest` fixtures extensively
- Mock Klipper interfaces (printer, mcu, toolhead, etc.)

## Important Implementation Details

### Klipper Integration
- This is a Klipper plugin, not a standalone application
- Klipper imports will be missing during development (by design)
- Mock all Klipper interfaces for testing
- Configuration uses Klipper's config file format

### Key Components
1. **Probe modes**: Scan mode (contactless) and Touch mode (contact probing)
2. **Temperature compensation**: Coil temperature affects measurements
3. **Bed mesh**: Custom mesh calibration implementation
4. **Models**: Calibration models for scan and touch modes (stored in config)
5. **Toolhead**: Backlash compensation wrapper
6. **Macros**: G-code command implementations

### Dependencies
- Core: `typing-extensions~=4.12`
- Optional: `scipy~=1.9` (for advanced calculations)
- Test: `pytest`, `pytest-mock`
- Lint: `ruff~=0.14`
- Type checking: `basedpyright~=1.27`

## Code Patterns

### Module Structure
```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, final

# Main imports
from cartographer.config.model_validator import validate_models

if TYPE_CHECKING:
    from cartographer.interfaces.printer import Macro
```

### Dataclasses for Configuration
Use dataclasses with validation for configuration objects:
```python
from dataclasses import dataclass

@dataclass
class ScanModeConfiguration:
    speed: float
    samples: int
    # ...
```

### Logging
Use standard Python logging:
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Starting calibration")
```

## Pull Request Guidelines

1. Ensure all linting passes: `uv run --only-group lint ruff check`
2. Format code: `uv run --only-group lint ruff format`
3. Run type checking: `uv run basedpyright`
4. Run tests: `uv run pytest`
5. Add tests for new functionality
6. Update documentation if adding public APIs

## Common Pitfalls

1. **Relative imports**: Always use absolute imports (enforced by ruff)
2. **Type checking imports**: Use `TYPE_CHECKING` block for imports only needed for types
3. **Line length**: Keep code lines under 120 characters
4. **Klipper dependencies**: Don't try to install or import actual Klipper code
5. **Test isolation**: Use mocks for all Klipper interfaces in tests

## Additional Resources

- [Project Documentation](https://docs.cartographer3d.com/)
- [Repository](https://github.com/Cartographer3D/cartographer3d-plugin)
- [Homepage](https://cartographer3d.com/)
