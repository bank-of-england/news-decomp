import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # News Decomposition Example

    This notebook shows how to simulate decomposition data, inspect it, and
    analyse it with news_decomp.
    """)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Import the libraries
    """)


@app.cell
def _():
    from news_decomp.news_decomp import NewsData
    from news_decomp.sample import plot, simulate

    return NewsData, plot, simulate


@app.cell
def _(mo):
    mo.md("""
    ## 2. Simulate and visualise the data
    """)


@app.cell
def _(plot, simulate):
    # Generate the sample data and its decomposition table.
    data = simulate()
    news_decomposition_data = data["decompositions"]
    plot(data)
    return (news_decomposition_data,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Inspect the decomposition table
    """)


@app.cell
def _(news_decomposition_data):
    print(news_decomposition_data)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Create NewsData and print a summary
    """)


@app.cell
def _(NewsData, news_decomposition_data):
    # Validate the table and print its dimensions.
    news_data = NewsData(news_decomposition_data)
    news_data.summary()


@app.cell
def _(mo):
    mo.md("""
    ## 5. Analyse the data

    Add analysis calls here.
    """)


@app.cell
def _():
    # Add analysis calls here.
    return


if __name__ == "__main__":
    app.run()
