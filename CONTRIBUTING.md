# Contributing

## Setup

```bash
git clone <repo-url>
cd sep-cfe-dice
pip install -e ".[dev,notebooks]"
```

## Development workflow

1. Create a feature branch.
2. Make focused changes.
3. Run checks locally:

```bash
ruff check src tests
pytest
```

4. Update docs and notebook examples if public behavior changed.
5. Open a pull request with a clear summary.

## Pull request checklist

- [ ] Code builds and imports cleanly.
- [ ] New behavior is documented.
- [ ] Tests added or rationale provided.
- [ ] Changelog updated (if user-facing change).
