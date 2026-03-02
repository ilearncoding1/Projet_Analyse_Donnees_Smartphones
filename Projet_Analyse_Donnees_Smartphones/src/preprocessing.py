import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load dataset
df = pd.read_csv("../data/smartphones.csv")

# Display first rows
print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nStatistical description:")
print(df.describe())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())



# ----------------------------
# DATA VISUALIZATION
# ----------------------------

# 1. Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Price_DH"], bins=5, kde=True)
plt.title("Distribution des prix des smartphones")
plt.xlabel("Prix (DH)")
plt.ylabel("Nombre")
plt.show()

# 2. Price by brand
plt.figure(figsize=(7,4))
sns.boxplot(x="Brand", y="Price_DH", data=df)
plt.title("Prix des smartphones par marque")
plt.xticks(rotation=45)
plt.show()

# 3. RAM vs Price
plt.figure(figsize=(6,4))
sns.scatterplot(x="RAM_GB", y="Price_DH", hue="Brand", data=df)
plt.title("Relation entre RAM et Prix")
plt.show()


# ----------------------------
# DATA CLEANING & PREPARATION
# ----------------------------

# 1. Encode categorical variables (Brand)
df_encoded = pd.get_dummies(df, columns=['Brand'], drop_first=True)

# 2. Normalize numerical features
numerical_features = ['Price_DH', 'RAM_GB', 'Storage_GB', 'Screen_Size_Inch', 'Battery_mAh']
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# 3. Check result
print("\nDataset after encoding & normalization:")
print(df_encoded.head())

# ----------------------------
# DIMENSIONALITY REDUCTION
# ----------------------------

# Features for dimensionality reduction (exclude non-numeric columns)
features = df_encoded.select_dtypes(include=['float64', 'int64', 'bool'])

# 1️⃣ PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)
print("\nExplained variance ratio (PCA):", pca.explained_variance_ratio_)

plt.figure(figsize=(6,4))
plt.scatter(pca_result[:,0], pca_result[:,1], c='blue')
plt.title("PCA - 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 2️⃣ t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(features)

plt.figure(figsize=(6,4))
plt.scatter(tsne_result[:,0], tsne_result[:,1], c='green')
plt.title("t-SNE - 2D Projection")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()
