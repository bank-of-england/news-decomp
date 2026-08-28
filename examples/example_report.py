"""Create a nowcast report from the sample data.

Run with::

    python -m examples.example_report

The report contains two panels:

1. **Nowcast evolution**: a line showing the GDP nowcast as releases arrive.
2. **Nowcast decomposition**: bars showing each indicator's contribution.

The script also prints a data-flow table of model updates and impacts.
"""

import matplotlib.pyplot as plt

from news_decomp import NewsData
from news_decomp.sample import simulate

# 1. Load the sample data. Replace this call with your own table in practice.
data = simulate()
news_data = NewsData(data["decompositions"])

# 2. Generate the report and its summary tables.
news_data.report()

plt.show()
