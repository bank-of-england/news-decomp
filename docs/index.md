# News Decomposition

`news_decomp` analyses forecast levels and revisions without depending on the
model that produced them. It accepts a validated `decompositions` table and
provides summary metrics, reporting tables, and plots.

## Installation

```bash
pip install news-decomp
```

Install the development and documentation tools with:

```bash
pip install -e ".[dev,docs]"
```

## Quick start

```python
from news_decomp import NewsData
from news_decomp.sample import simulate

news_data = NewsData(simulate()["decompositions"])
news_data.summary()
```

Read the [API Reference](api.md) for public classes and methods. Read the
[News Decomposition data contract](news_decomp.md) for the input-table schema,
identities, and worked examples. The [Example Data notebook](notebooks/example_data.md)
walks through simulation, inspection, and summary output.
