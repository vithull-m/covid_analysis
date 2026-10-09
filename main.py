import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load data
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []
if not files:
    files = sorted(BASE.glob("*.csv"))
if not files:
    raise FileNotFoundError(
        f"No CSV dataset found in {DATA_DIR} or {BASE}."
    )
if len(files) > 1:
    raise ValueError(
        f"Multiple CSV files found in {files[0].parent}. "
        "Keep only the intended dataset in that folder."
    )

df = pd.read_csv(files[0], low_memory=False)

# 2. Prepare numerical data
X = df.select_dtypes(include=np.number)
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X),
                 columns=X.columns)

# 3. Scale data
scaled = StandardScaler().fit_transform(X)

# 4. Apply PCA
pca = PCA().fit(scaled)
variance = np.cumsum(pca.explained_variance_ratio_)
print("Components for 95% variance:", np.searchsorted(variance, .95) + 1)

plt.plot(variance * 100, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance (%)")
plt.show()

# 5. Reduce to 2D and 3D
two = PCA(2).fit_transform(scaled)
three = PCA(3).fit_transform(scaled)

print("2D variance:", round(PCA(2).fit(scaled).explained_variance_ratio_.sum()*100, 2), "%")
print("3D variance:", round(PCA(3).fit(scaled).explained_variance_ratio_.sum()*100, 2), "%")

plt.scatter(two[:, 0], two[:, 1], s=5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(three[:, 0], three[:, 1], three[:, 2], s=5)
plt.show()

# 6. Save reduced data
out = BASE / "outputs"
out.mkdir(exist_ok=True)
pd.DataFrame(three, columns=["PC1", "PC2", "PC3"]).to_csv(
    out / "reduced_data.csv", index=False)

print("PCA completed!")