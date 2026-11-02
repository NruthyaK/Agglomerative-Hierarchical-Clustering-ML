# Agglomerative Clustering: From Iris Data to a Dendrogram 🌿

> **Can a computer discover groups in a dataset without being told the answers?**
>
> This notebook explores that question with **agglomerative hierarchical clustering**. It scales the Iris measurements, visualizes their hierarchy with a dendrogram, and compares cluster plots made with two and three clusters.

[![Open the notebook](https://img.shields.io/badge/Open-Jupyter%20Notebook-orange?logo=jupyter)](./agglomerative_clustering.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-Iris-4c8c4a)](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
[![Method](https://img.shields.io/badge/Clustering-Hierarchical-6f42c1)](#how-agglomerative-clustering-works)

---

## 👀 What you’ll explore

This is a compact, hands-on introduction to **unsupervised learning**—learning patterns from features without using known class labels to train the model.

In the notebook, you will:

1. Load the classic Iris dataset from scikit-learn.
2. Standardize its four measurements so they are on comparable scales.
3. Use **Ward linkage** to create a hierarchical clustering tree.
4. Read a dendrogram as a visual aid for thinking about cluster count.
5. Fit agglomerative clustering with `n_clusters=2`, then with `n_clusters=3`.
6. Plot the resulting labels using two Iris measurements.

**The main idea:** the model sees the measurements—not the Iris species labels—when forming clusters. The notebook loads the target labels, but does not use them to fit or evaluate the clustering.

<details>
<summary>✨ Quick project snapshot</summary>

| | |
|---|---|
| **Learning type** | Unsupervised learning |
| **Algorithm** | Agglomerative hierarchical clustering |
| **Example data** | Iris measurements |
| **Preprocessing** | `StandardScaler` |
| **Hierarchy method** | Ward linkage |
| **Cluster counts compared** | 2 and 3 |
| **Main visualizations** | Dendrogram and two scatter plots |
| **Notebook** | [`agglomerative_clustering.ipynb`](./agglomerative_clustering.ipynb) |

</details>

## 🧭 Contents

- [What is the goal?](#-what-is-the-goal)
- [How agglomerative clustering works](#-how-agglomerative-clustering-works)
- [What the notebook does](#-what-the-notebook-does)
- [Run it yourself](#-run-it-yourself)
- [How to read the plots](#-how-to-read-the-plots)
- [Interactive experiments](#-interactive-experiments)
- [Important notes](#-important-notes)
- [Project structure](#-project-structure)

## 🎯 What is the goal?

The Iris dataset contains measurements of iris flowers. The exercise asks whether a clustering algorithm can group flowers with similar measurements **without being given their species during fitting**.

The notebook uses four numeric features:

- sepal length
- sepal width
- petal length
- petal width

It then draws the cluster assignments in two dimensions: **sepal length** on the horizontal axis and **petal length** on the vertical axis. This makes the output easy to inspect, while the actual clustering uses all four standardized features.

## 🌱 How agglomerative clustering works

Agglomerative clustering is a **bottom-up hierarchical** method:

1. Start with every sample as its own cluster.
2. Find the two closest clusters and merge them.
3. Repeat until the hierarchy contains one cluster—or until a chosen cluster count is reached.

The meaning of “closest” depends on the linkage rule. This notebook uses **Ward linkage**, which chooses merges to minimize the increase in within-cluster variation. Ward linkage is commonly used with Euclidean features and is sensitive to feature scale, which is why scaling is an important part of this example.

### Reading the dendrogram

A dendrogram records the sequence of merges:

- The leaves represent individual samples.
- Each branch represents a merge.
- The merge height indicates the distance or cost associated with that merge.
- A large vertical jump can suggest a useful place to cut the tree into groups.

> **A dendrogram suggests; it does not decide.** The notebook’s comment is an important reminder: a dendrogram is a visual aid for choosing a cluster count, not proof that there is one objectively correct answer. The appropriate number also depends on the data and the purpose of the analysis.

## 🧪 What the notebook does

The full, executable walkthrough is in [`agglomerative_clustering.ipynb`](./agglomerative_clustering.ipynb). Here is its sequence:

### 1. Load the libraries and data

The notebook imports scikit-learn’s `load_iris`, plotting tools from Matplotlib and Seaborn, and then loads the Iris data. `X` contains the measurements; `y` contains the dataset’s target labels.

### 2. Standardize the measurements

`StandardScaler` transforms each feature to have a mean near zero and a standard deviation near one:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

This prevents a feature from having undue influence just because its numeric scale is larger than another feature’s scale.

### 3. Build and draw a Ward dendrogram

The notebook computes the hierarchical tree from the scaled measurements:

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X_scaled, method="ward")
dendrogram(Z)
```

The dendrogram is used to inspect possible groupings before choosing a cluster count for the model.

### 4. Fit and visualize two clusters

The first model asks for two groups:

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2)
labels = model.fit_predict(X_scaled)
```

The notebook plots the assigned labels against the first and third Iris features (`X[:, 0]` and `X[:, 2]`).

### 5. Fit and visualize three clusters

The same workflow is repeated with `n_clusters=3`. Comparing the two scatter plots shows how changing the requested number of clusters changes the assignments.

## ▶️ Run it yourself

### Requirements

- Python
- Jupyter Notebook or JupyterLab
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`

Install the packages with:

```bash
python -m pip install jupyter scikit-learn scipy matplotlib seaborn
```

Then open the notebook:

```bash
jupyter notebook agglomerative_clustering.ipynb
```

Run the cells from top to bottom. In Jupyter, you can use **Run All** or execute each cell individually to see how the data, hierarchy, and plots build on one another.

## 📊 How to read the plots

### Dendrogram

Look for relatively large jumps between merge heights. One possible way to reason about cluster count is to imagine drawing a horizontal line through the tree: the number of branches that line crosses corresponds to the number of groups at that cut. Try more than one cut and ask whether the grouping is meaningful for the task.

### Cluster scatter plots

The scatter plots show only two of the four measurements, while labels were learned from all four standardized measurements. Therefore:

- Points that appear close in the 2D chart may still differ on the other features.
- A 2D plot cannot show every distinction the model used.
- Cluster IDs such as `0`, `1`, and `2` are arbitrary names, not rankings or species names.
- The colors indicate the model’s assigned cluster labels, not the Iris target labels.

## 🕹️ Interactive experiments

Ready to explore? Make a copy of the notebook first, then try one change at a time.

<details>
<summary>🔢 Experiment 1: What changes when you choose a different cluster count?</summary>

Change `n_clusters` and rerun the model and plot cells. Compare two, three, and four clusters.

**Think about:** Do the groups become easier to interpret, or are they simply split into smaller pieces? Does the dendrogram support your choice?

</details>

<details>
<summary>📐 Experiment 2: What does scaling change?</summary>

Compare the clustering from `X_scaled` with a model fitted to the original `X`. Keep the cluster count fixed so that you are changing only the input representation.

**Think about:** Which features have the greatest numeric ranges? Does the grouping change when measurements are not standardized?

</details>

<details>
<summary>🎨 Experiment 3: Can you see the clusters from another angle?</summary>

Change the scatter plot’s `x` and `y` columns to plot different feature pairs. For example, compare sepal width with petal width.

**Think about:** Which feature pair makes the assignments easiest to inspect? Does one view hide patterns visible in another?

</details>

<details>
<summary>🧠 Experiment 4: How could you compare clusters with known species?</summary>

The Iris dataset includes target labels in `y`, but the notebook does not use them to fit the clustering. As a separate analysis, compare the predicted cluster labels with `y` using a metric intended for clustering, such as the adjusted Rand index.

**Think about:** Why is comparing clusters with known labels different from training the clustering algorithm on those labels?

</details>


## ⚠️ Important notes

- **Cluster count is a modeling choice.** The notebook compares two values; it does not automatically select a best value.
- **Unsupervised clusters are not class predictions.** Cluster labels do not inherently correspond to species or have stable meanings across model runs or settings.
- **The plotted view is only two-dimensional.** The clustering uses all four scaled features.
- **There is no quantitative evaluation in the notebook.** The current workflow visualizes the groups; it does not calculate a clustering score or compare assignments with the known targets.
- **Dendrogram interpretation takes judgment.** Treat suggested cuts as hypotheses to investigate, rather than definitive answers.

## 📁 Project structure

```text
Agglomerative Clustering/
├── README.md
└── agglomerative_clustering.ipynb
```

## 🙌 Have fun exploring

Start with the notebook, make a prediction about what you expect to see, then run the cells and compare your prediction with the plots. The most useful question is not only **“How many clusters can I make?”** but also **“Do these clusters help explain the data?”**
