import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter

file_path = r"C:\Users\cc\Desktop\UCB\STAT214\image_data"
labeled_graph = ["O013257.npz", "O013490.npz", "O012791.npz"]
data_list = []

for file in labeled_graph:
    full_path = os.path.join(file_path, file)
    if os.path.exists(full_path):
        data = np.load(full_path)["arr_0"]
        df = pd.DataFrame(data)
        df["filename"] = file
        data_list.append(df)

if not data_list:
    print("Error")
    exit()

df_all = pd.concat(data_list, ignore_index=True)

new_col_names = ["X", "Y", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "Label"]
df_all.columns = new_col_names + ["filename"]

plt.figure(figsize=(12, 5))
for file in labeled_graph:
    subset = df_all[df_all["filename"] == file]
    plt.scatter(subset["X"], subset["Y"], c=subset["Label"], alpha=0.5, label=file)
    
"""
plt.colorbar(label="Cloud Binary Variable for 1=Cloud, -1=No Cloud, 0=Unlabeled")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Expert Cloud Labels")
plt.legend()
plt.show()
"""

X = df_all.drop(columns=["X", "Y", "Label", "filename"])
Y = df_all["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=50)
rf.fit(X_scaled, Y)

feature_order = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feature_order = feature_order.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(x="Importance", y="Feature", data=feature_order)
plt.title("Ordered feature importance")
plt.xlabel("Importance score")
plt.ylabel("Feature")
plt.show()

top3 = feature_order["Feature"][:3].values  #Select the top three features that we might interested about: SD, NDAI, AN

plt.figure(figsize=(12, 5))
for i, feature in enumerate(top3):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data=df_all, x=feature, hue="Label", bins=30, kde=True, alpha=0.5)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

