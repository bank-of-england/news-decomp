---
title: Example Data Marimo
marimo-version: 0.24.0
---

```python {.marimo}
import marimo as mo
```

# News Decomposition Example

This notebook shows how to simulate decomposition data, inspect it, and
analyse it with news_decomp.
<!---->
## 1. Import the libraries

```python {.marimo}
from news_decomp.news_decomp import NewsData
from news_decomp.sample import plot, simulate
```

## 2. Simulate and visualise the data

```python {.marimo}
# Generate the sample data and its decomposition table.
data = simulate()
news_decomposition_data = data["decompositions"]
plot(data)
```

## 3. Inspect the decomposition table

```python {.marimo}
print(news_decomposition_data)
```

## 4. Create NewsData and print a summary

```python {.marimo}
# Validate the table and print its dimensions.
news_data = NewsData(news_decomposition_data)
news_data.summary()
```

## 5. Analyse the data

Add analysis calls here.

```python {.marimo}
# Add analysis calls here.
```