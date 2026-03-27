# Development

## Run tests

```bash
pytest
```

## Lint

```bash
ruff check src tests
```

## Build distribution

```bash
python -m build
```

## Release checklist

1. Bump version in `pyproject.toml`.
2. Add release notes in `CHANGELOG.md`.
3. Create git tag (for example `v0.1.1`).
4. Build and publish artifacts.
