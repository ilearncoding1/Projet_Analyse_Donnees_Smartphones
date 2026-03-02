import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Load preprocessed dataset
df = pd.read_csv("../data/smartphones.csv")

# ----------------------------
# Feature selection
# ----------------------------

# Convert Brand to dummies
df_encoded = pd.get_dummies(df, columns=['Brand'], drop_first=True)

# Features (exclude Model & Price)
X = df_encoded.drop(['Model', 'Price_DH'], axis=1)
y = df_encoded['Price_DH']  # Target: Price

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# Decision Tree Regressor
# ----------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Evaluate
score = dt.score(X_test, y_test)
print(f"Decision Tree R^2 score on test set: {score:.2f}")

# Visualize tree
plt.figure(figsize=(12,6))
plot_tree(dt, feature_names=X.columns, filled=True, rounded=True)
plt.show()
