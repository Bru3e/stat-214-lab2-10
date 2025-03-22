import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,6)

def load_npz(filename):
    data = np.load(filename)
    key = list(data.keys())[0]
    return data[key]

filenames = ["O013257.npz", "O012791.npz", "O013490.npz"]
columns = ["y", "x", "NDAI", "SD", "CORR", "Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN", "Label"]

dfs = []
for file in filenames:
    data = load_npz(file)
    df = pd.DataFrame(data, columns=columns)
    dfs.append(df)
    print(f"{file} loaded, shape: {df.shape}")

def plot_expert_labels(df, title):
    df_labeled = df[df["Label"] != 0]
    plt.figure(figsize=(8,6))
    colors = {-1: "blue", 1: "red"}
    for label, color in colors.items():
        subset = df_labeled[df_labeled["Label"] == label]
        plt.scatter(subset["x"], subset["y"], s=1, c=color, label="Cloud" if label==1 else "No Cloud")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.legend()
    plt.show()

for i, df in enumerate(dfs):
    plot_expert_labels(df, f"Expert Labels for Image {filenames[i]}")

radiance_cols = ["Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN"]
feature_cols = ["NDAI", "SD", "CORR"]

def plot_radiance_pairplot(df, title):
    df_labeled = df[df["Label"] != 0]

    df_sample = df_labeled.copy()
    df_sample["Label_str"] = df_sample["Label"].apply(lambda x: "Cloud" if x == 1 else "No Cloud")
    sns.pairplot(df_sample, vars=radiance_cols, hue="Label_str", 
                 plot_kws={'s': 10}, palette={"Cloud": "red", "No Cloud": "blue"})
    plt.suptitle(title, y=1.02)
    plt.show()

def plot_feature_distribution(df, feature):
    df_labeled = df[df["Label"] != 0]
    plt.figure(figsize=(8,6))
    sns.histplot(data=df_labeled, x=feature, hue="Label", bins=50, 
                 palette={-1:"blue", 1:"red"}, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()

for i, df in enumerate(dfs):
    plot_radiance_pairplot(df, f"Radiance Angles Pairplot for Image {filenames[i]}")

for i, df in enumerate(dfs):
    for feature in feature_cols:
        plot_feature_distribution(df, feature)

def compute_correlations(df, features):
    df_labeled = df[df["Label"] != 0]
    corr_dict = {}
    for feature in features:
        corr_val = df_labeled[feature].corr(df_labeled["Label"])
        corr_dict[feature] = corr_val
    return corr_dict

for i, df in enumerate(dfs):
    corrs = compute_correlations(df, feature_cols)
    print(f"Feature-label correlations for {filenames[i]}: {corrs}")

train_df = pd.concat([dfs[0][dfs[0]["Label"] != 0], dfs[1][dfs[1]["Label"] != 0]], ignore_index=True)
test_val_df = dfs[2][dfs[2]["Label"] != 0].copy()
test_val_df = test_val_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(0.5 * len(test_val_df))
val_df = test_val_df.iloc[:split_index].copy()
test_df = test_val_df.iloc[split_index:].copy()

print("Size of training set:", train_df.shape)
print("Size of validation set:", val_df.shape)
print("Size of test set:", test_df.shape)


plt.rcParams["figure.figsize"] = (7,5)


def boxplot_by_class(df, features, label_col="Label"):
    """
    Create boxplots for each feature, comparing cloud vs. no-cloud distributions.
    """
    df_labeled = df[df[label_col] != 0].copy()
    df_labeled["Label_str"] = df_labeled[label_col].apply(lambda x: "Cloud" if x == 1 else "No Cloud")
    
    for feature in features:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df_labeled, x="Label_str", y=feature,
                    palette={"Cloud":"red", "No Cloud":"blue"})
        plt.title(f"Boxplot of {feature} by Cloud / No Cloud")
        plt.show()

def correlation_heatmap(df, columns, title="Correlation Heatmap"):
    """
    Compute and plot a correlation heatmap for the given columns.
    """
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title(title)
    plt.show()


def correlation_by_class(df, columns, label_col="Label"):
    """
    Split the data into cloud vs. no-cloud, compute correlation matrices separately.
    Returns (corr_cloud, corr_nocloud).
    """
    df_labeled = df[df[label_col] != 0].copy()
    df_cloud = df_labeled[df_labeled[label_col] == 1]
    df_nocloud = df_labeled[df_labeled[label_col] == -1]
    
    corr_cloud = df_cloud[columns].corr()
    corr_nocloud = df_nocloud[columns].corr()
    return corr_cloud, corr_nocloud

def plot_two_heatmaps(corr1, corr2, title1="Cloud", title2="No Cloud"):
    """
    Plot two correlation heatmaps
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    sns.heatmap(corr1, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=axes[0])
    axes[0].set_title(title1)
    sns.heatmap(corr2, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=axes[1])
    axes[1].set_title(title2)
    plt.tight_layout()
    plt.show()

radiance_cols = ["Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN"]
feature_cols = ["NDAI", "SD", "CORR"]
all_cols = radiance_cols + feature_cols

boxplot_by_class(dfs[0], radiance_cols)
correlation_heatmap(dfs[0][dfs[0]["Label"] != 0], all_cols, "Correlation Heatmap (Image 1)")
corr_cloud, corr_nocloud = correlation_by_class(dfs[0], all_cols)
plot_two_heatmaps(corr_cloud, corr_nocloud, "Cloud (Image 1)", "No Cloud (Image 1)")


def loading(filename):
    '''
    Load the data from the given .npz file and return as a DataFrame.
    '''
    data = np.load(filename)
    key = list(data.keys())[0]
    columns = ["y", "x", "NDAI", "SD", "CORR", 
               "Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN", "Label"]

    df = pd.DataFrame(data[key], columns=columns)
    
    return df

filenames = ["O013257.npz", "O012791.npz", "O013490.npz"]
dataframes = {}

for file in filenames:
    df = loading(file)
    dataframes[file] = df 

combined_df = pd.concat(dataframes.values(), ignore_index=True)
print("Combined DataFrame shape:", combined_df.shape)
print(combined_df.head())  

for file, df in dataframes.items():
    base_name = os.path.basename(file)
    output_name = base_name.replace('.npz', '_cleaned.npz')
    np.savez_compressed(output_name, data=df.to_numpy())
    print(f"Saved cleaned data to {output_name}")


combined_df_no_0 = combined_df[combined_df["Label"] != 0]


features = ["NDAI", "SD", "CORR", 
            "Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN"]


# Compute Pearson correlation between each feature and the label
correlations = {}
for feature in features:
    corr_val = combined_df_no_0[feature].corr(combined_df_no_0["Label"])
    correlations[feature] = corr_val
    print(f"Correlation between {feature} and Label: {corr_val:.3f}")

corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["Correlation"])
corr_df["abs_corr"] = corr_df["Correlation"].abs()
corr_df = corr_df.sort_values(by="abs_corr", ascending=False)

# Plot the correlation values
plt.figure(figsize=(10,6))
sns.barplot(x=corr_df.index, y=corr_df["Correlation"], palette="viridis")
plt.xticks(rotation=45)
plt.title("Pearson Correlation between Features and Label")
plt.ylabel("Correlation Coefficient")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()

top_features = corr_df.index[:3].tolist()  
for feature in top_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Label", y=feature, data=combined_df_no_0, palette="Set2")
    plt.title(f"Distribution of {feature} by Label")
    plt.xlabel("Label (-1: No Cloud, +1: Cloud)")
    plt.tight_layout()
    plt.show()

X = combined_df_no_0[features]
y = combined_df_no_0["Label"]

f_scores, p_values = f_classif(X, y)

f_score_df = pd.DataFrame({"Feature": features, "F-score": f_scores, "p-value": p_values})
f_score_df = f_score_df.sort_values(by="F-score", ascending=False)

print(f_score_df)

combined_df_no_0.groupby("Label")[["SD", "NDAI", "Radiance_AN"]].agg(["mean", "std"])

# Plot the distribution of the features by label
for feature in top_features:
    plt.figure(figsize=(8,5))
    sns.histplot(data=combined_df_no_0, x=feature, hue="Label", kde=True, bins=30, palette="coolwarm", alpha=0.6)
    plt.title(f"Distribution of {feature} by Label")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.legend(title="Label", labels=["No Cloud (-1)", "Cloud (+1)"])
    plt.tight_layout()
    plt.show()

X = combined_df_no_0[features]
y = combined_df_no_0["Label"]

# Another way to compute feature importances using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state= 918)
rf.fit(X, y)
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({'Feature': features, 'Importance': rf_importances})
rf_df = rf_df.sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importances:")
print(rf_df)

plt.figure(figsize=(8,6))
sns.barplot(x="Feature", y="Importance", data=rf_df)
plt.title("Random Forest Feature Importances")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.rcParams["figure.figsize"] = (8,6)

def make_data(patch_size=9):
    """
    Load the image data from .npz files and create patches.
    For expert-labeled images (with 11 columns), the label is preserved.
    Returns:
        images_long: A list of numpy arrays of the original images (with label if present).
        patches: A list of lists of patches for each image.
    """

    # We only use the data with labels
    filepaths = glob.glob("C:\\Users\\lenovo\\Desktop\\stats214\\lab2wzh\\testdatalab2\\O013490.npz") 
    images_long = []
    for fp in filepaths:
        npz_data = np.load(fp, allow_pickle=True)
        key = list(npz_data.files)[0]
        data = npz_data[key]
        # DO NOT remove labels
        images_long.append(data)

    all_y = np.concatenate([img[:, 0] for img in images_long]).astype(int)
    all_x = np.concatenate([img[:, 1] for img in images_long]).astype(int)
    global_miny, global_maxy = all_y.min(), all_y.max()
    global_minx, global_maxx = all_x.min(), all_x.max()
    height = int(global_maxy - global_miny + 1)
    width = int(global_maxx - global_minx + 1)

    has_label = (images_long[0].shape[1] == 11)
    nchannels = images_long[0].shape[1] - 3 if has_label else images_long[0].shape[1] - 2

    images = []
    for img in images_long:
        y = img[:, 0].astype(int)
        x = img[:, 1].astype(int)
        y_rel = y - global_miny
        x_rel = x - global_minx
        image = np.zeros((nchannels, height, width))
        valid_mask = (y_rel >= 0) & (y_rel < height) & (x_rel >= 0) & (x_rel < width)
        y_valid = y_rel[valid_mask]
        x_valid = x_rel[valid_mask]
        img_valid = img[valid_mask]
        for c in range(nchannels):
            image[c, y_valid, x_valid] = img_valid[:, c + 2]
        images.append(image)
    print('done reshaping images')

    images = np.array(images)
    pad_len = patch_size // 2
    means = np.mean(images, axis=(0, 2, 3))[:, None, None]
    stds = np.std(images, axis=(0, 2, 3))[:, None, None]
    images = (images - means) / stds

    patches = []
    for i in range(len(images_long)):
        if i % 10 == 0:
            print(f'working on image {i}')
        patches_img = []
        img_mirror = np.pad(
            images[i],
            ((0, 0), (pad_len, pad_len), (pad_len, pad_len)),
            mode="reflect",
        )
        ys = images_long[i][:, 0].astype(int)
        xs = images_long[i][:, 1].astype(int)
        for y, x in zip(ys, xs):
            y_idx = int(y - global_miny + pad_len)
            x_idx = int(x - global_minx + pad_len)
            patch = img_mirror[
                :,
                y_idx - pad_len : y_idx + pad_len + 1,
                x_idx - pad_len : x_idx + pad_len + 1,
            ]
            patches_img.append(patch.astype(np.float32))
        patches.append(patches_img)

    return images_long, patches

def compute_patch_entropy(patch, bins=10):
    flat_patch = patch.flatten()
    hist, _ = np.histogram(flat_patch, bins=bins, density=True)
    hist = hist + 1e-8  # avoid log(0)
    return entropy(hist)

def extract_range_entropy_features(patches):
    """
    Extract range and entropy features for each patch.
    Returns a list of numpy arrays.
    """
    # Entropy is not good
    feature_list = []
    for patches_img in patches:
        ranges = []
        entropies = []
        for patch in patches_img:
            r = np.max(patch) - np.min(patch)
            e = compute_patch_entropy(patch, bins=10)
            ranges.append(r)
            entropies.append(e)
        feature_array = np.column_stack((ranges, entropies))
        feature_list.append(feature_array)
    return feature_list

images_long, patches = make_data(patch_size=9)
new_features = extract_range_entropy_features(patches)

augmented_dfs = []
for i, data in enumerate(images_long):
    df = combined_df_no_0
    df = pd.DataFrame(data, columns=["y", "x", "NDAI", "SD", "CORR",
                                       "Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN", "Label"])
    df["patch_range"] = new_features[i][:, 0]
    df["patch_entropy"] = new_features[i][:, 1]
    augmented_dfs.append(df)

combined_augmented_df = pd.concat(augmented_dfs, ignore_index=True)
print("Combined augmented DataFrame shape:", combined_augmented_df.shape)
print(combined_augmented_df.head())

corr_range = combined_augmented_df["patch_range"].corr(combined_augmented_df["Label"])
corr_entropy = combined_augmented_df["patch_entropy"].corr(combined_augmented_df["Label"])
print(f"Pearson correlation (patch_range vs Label): {corr_range:.3f}")
print(f"Pearson correlation (patch_entropy vs Label): {corr_entropy:.3f}")


plt.figure(figsize=(8,6))
sns.boxplot(x="Label", y="patch_range", data=combined_augmented_df, palette="Set2")
plt.title("Boxplot of patch_range by Label")
plt.xlabel("Label (-1: No Cloud, +1: Cloud)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Label", y="patch_entropy", data=combined_augmented_df, palette="Set2")
plt.title("Boxplot of patch_entropy by Label")
plt.xlabel("Label (-1: No Cloud, +1: Cloud)")
plt.tight_layout()
plt.show()



# Using PCA to compute explained variance ratio of a patch
def compute_patch_pca_feature(patch):
    channels, _ , _ = patch.shape
    patch_reshaped = patch.reshape(channels, -1).T  
    pca = PCA(n_components=1)
    pca.fit(patch_reshaped)
    return pca.explained_variance_ratio_[0]

def compute_patch_grad_mean(patch):
    channels = patch.shape[0]
    grad_means = []
    for c in range(channels):
        gx = ndimage.sobel(patch[c], axis=0)
        gy = ndimage.sobel(patch[c], axis=1)
        grad = np.hypot(gx, gy)
        grad_means.append(np.mean(grad))
    return np.mean(grad_means)

images_long, patches = make_data(patch_size=9)

pca_features_list = [] 
grad_features_list = [] 
for patches_img in patches:
    pca_feats = []
    grad_feats = []
    for patch in patches_img:
        pca_feat = compute_patch_pca_feature(patch)
        grad_feat = compute_patch_grad_mean(patch)
        pca_feats.append(pca_feat)
        grad_feats.append(grad_feat)
    pca_features_list.append(np.array(pca_feats))
    grad_features_list.append(np.array(grad_feats))


augmented_dfs = []
for i, data in enumerate(images_long):
    df = pd.DataFrame(data, columns=["y", "x", "NDAI", "SD", "CORR",
                                       "Radiance_DF", "Radiance_CF", "Radiance_BF", "Radiance_AF", "Radiance_AN", "Label"])
    df["pca_feature"] = pca_features_list[i]
    df["grad_mean"] = grad_features_list[i]
    df["patch_range"] = new_features[i][:, 0]
    df["patch_entropy"] = new_features[i][:, 1]
    augmented_dfs.append(df)

combined_augmented_df = pd.concat(augmented_dfs, ignore_index=True)
print("Combined augmented DataFrame shape:", combined_augmented_df.shape)
print(combined_augmented_df.head())

csv_filename = "13490.csv"
combined_augmented_df.to_csv(csv_filename, index=False)
print(f"Saved data to {csv_filename}")

corr_pca = combined_augmented_df["pca_feature"].corr(combined_augmented_df["Label"])
corr_grad = combined_augmented_df["grad_mean"].corr(combined_augmented_df["Label"])
print(f"Pearson correlation (pca_feature vs Label): {corr_pca:.3f}")
print(f"Pearson correlation (grad_mean vs Label): {corr_grad:.3f}")

plt.figure(figsize=(8,6))
sns.boxplot(x="Label", y="pca_feature", data=combined_augmented_df, palette="Set2")
plt.title("Boxplot of PCA Feature by Label")
plt.xlabel("Label (-1: No Cloud, +1: Cloud)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="Label", y="grad_mean", data=combined_augmented_df, palette="Set2")
plt.title("Boxplot of Gradient Mean by Label")
plt.xlabel("Label (-1: No Cloud, +1: Cloud)")
plt.tight_layout()
plt.show()


def heatmap_expert(df, title_labels="Expert Labels", title_grad="Mean Gradient Heatmap"):
    df_filtered = df[df["Label"] != 0].copy()
    grad_mean_avg = df_filtered["grad_mean"].mean()
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    colors = {-1: "blue", 1: "red"}
    for label, color in colors.items():
        subset = df_filtered[df_filtered["Label"] == label]
        ax1.scatter(subset["x"], subset["y"], s=1, c=color, label="Cloud" if label == 1 else "No Cloud")
    ax1.set_title(title_labels)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")
    ax1.legend()
    ax2 = axes[1]
    red = df_filtered[df_filtered["grad_mean"] > grad_mean_avg]
    blue = df_filtered[df_filtered["grad_mean"] <= grad_mean_avg]
    ax2.scatter(red["x"], red["y"], s=1, c='red', label="Above Avg")
    ax2.scatter(blue["x"], blue["y"], s=1, c='blue', label="Below Avg")
    ax2.set_title(title_grad)
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.legend()

    plt.tight_layout()
    plt.show()

heatmap_expert(combined_augmented_df)