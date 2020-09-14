# %% [markdown]
import seaborn as sns
import matplotlib as mpl

print(f"seaborn version: {sns.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"matplotlib backend: {mpl.get_backend()}")

iris_data = sns.load_dataset("iris")

sns.scatterplot(data=iris_data, x="sepal_length", y="sepal_width")

# make a new column called int(0)
iris_data[0] = iris_data["sepal_length"].copy()

sns.scatterplot(data=iris_data, x=0, y="sepal_width")

