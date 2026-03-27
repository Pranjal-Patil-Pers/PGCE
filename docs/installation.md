# Installation

## Requirements

- Python 3.9+
- pip

## Local editable install

```bash
pip install -e .
```

## Install with notebook extras

```bash
pip install -e ".[notebooks]"
```

## Install with development tools

```bash
pip install -e ".[dev,notebooks]"
```

## Verify

```bash
python -c "import SEP_CFE_DiCE as m; print(sorted(m.__all__)[:5])"
```
