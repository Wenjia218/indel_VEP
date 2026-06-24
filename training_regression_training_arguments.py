from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import glob
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--batched", type=str, required=True)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--output_filename", type=str, required=True)
parser.add_argument("--NaN", type=str, default="mean")
parser.add_argument("--frac", type=str, default="False")
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()


if_batched = args.batched == "True"
file_path = args.input
output_file_path = args.output
output_filename = args.output_filename
NaN = args.NaN
frac = args.frac == "True"
regularization = args.C

os.makedirs(output_file_path, exist_ok=True)

if if_batched:
    files = sorted(glob.glob(file_path))[:4]
    all_gene_features_merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
else:
    all_gene_features_merged = pd.read_parquet(file_path)


df_allele = pd.read_parquet("/s/project/benchmark-lm/selection_inference/benchmarklm/sample_variants/data/gnomad_v4_observed_indels_5_to_20_length_vep_scored_downsampled.parquet")
df_allele["variant_id"] = (
    df_allele["Chromosome"].astype(str)
    + ":"
    + df_allele["pos"].astype(str)
    + ":"
    + df_allele["ref"].astype(str)
    + ">"
    + df_allele["alt"].astype(str)
)

df_allele["rare"] = (
    df_allele["AF"] < 0.01
).astype(int) 

df_train = df_allele[["variant_id", "rare"]].merge(
    all_gene_features_merged,
    on="variant_id",
    how="inner"
)

del df_allele


# train on a smaller sample if OOM

if frac:
    df_train = df_train.sample(frac=0.1, random_state=42)

X = df_train.drop(
    columns=["variant_id", "rare"]
)

if NaN == "mean":
    X = X.fillna(X.mean())
else:
    X = X.fillna(0) # other methods for later


y = df_train["rare"]

print(f"X shape: {X.shape}, memory: {X.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"Class balance: {y.mean():.3f} rare")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# save test set
df_test_out = X_test.copy()
df_test_out["rare"] = y_test
df_test_out["variant_id"] = df_train.loc[X_test.index, "variant_id"]
df_test_out.to_parquet(output_file_path + "df_test_" + output_filename + ".parquet", index=False)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver="saga",
        penalty="l1",
        C=regularization,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(
    model, 
    X_train, 
    y_train, 
    cv=cv, 
    scoring="roc_auc", 
    n_jobs=4
)
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")

# fit on full training set and evaluate on test set
model.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test AUC: {test_auc:.3f}")


output_name = output_file_path + "cv_results_" + output_filename + ".csv"

# save CV scores
pd.DataFrame({"fold": range(1, 11), "auc": scores}).to_csv(
    output_name, index=False
)

output_name = output_file_path + "best_model_" + output_filename + ".pkl"
joblib.dump(model, output_name)

print("Model and CV results saved!")