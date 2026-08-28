from news_decomp.news_decomp import NewsData
from news_decomp.sample import plot, simulate

# Generate the sample data.
data = simulate()
news_decomposition_data = data["decompositions"]
plot(data)
print(news_decomposition_data)

# Create a NewsData instance and print its summary.
news_data = NewsData(news_decomposition_data)
news_data.summary()

# Add analysis calls here.
