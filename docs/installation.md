# Installation

## Requirements

- Python 3.9+
- pip

## Local editable install

```bash
pip install -e .
```

## Install with development tools

```bash
pip install -e ".[dev]"
```

## Verify

```bash
python -c "import PGCE as m; print(sorted(m.__all__)[:5])"
```
