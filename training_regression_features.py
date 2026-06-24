import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

with open("results_logistic/gene_list.txt", "r") as f:
    gene_list = [line.strip() for line in f.readlines()]


base_path = "/s/project/benchmark-lm/data/orthology_primates/preds"
important_columns = ["track_name", "strand", "variant_id", "log10_median_of_exon_medians_alt"]


cols_needed = ["human_ncbi_id", "species", "track_name", "strand", "log10_median_of_exon_medians"]
df_expression_agg = pd.read_parquet("results_logistic/species_reference.parquet", columns=cols_needed)

df_expression_agg["feature_name"] = (
    df_expression_agg["species"] + "_" + 
    df_expression_agg["track_name"] + "_" + 
    df_expression_agg["strand"]
)

df_expression_agg = df_expression_agg.drop(columns=["species", "track_name", "strand"])

df_gene_features = df_expression_agg.pivot_table(
    index="human_ncbi_id",
    columns="feature_name",
    values="log10_median_of_exon_medians",
    aggfunc="mean"
)

threshold = 0.2 * len(df_gene_features)
df_gene_features = df_gene_features.dropna(axis=1, thresh=int(threshold))


expr_columns = df_gene_features.columns.tolist()
output_path = "results_logistic/VS_per_track_strand_expression_log10medianalt_per_track_species.parquet"

all_gene_features = []

for i, gene in enumerate(tqdm(gene_list, desc="Processing genes")):
    f = os.path.join(
        base_path,
        f"gnomad_indels_scores/all_tracks_variant_gene_scores_all_species_{gene}.parquet"
    )
    if not os.path.exists(f):
        continue

    df = pd.read_parquet(f, columns=important_columns)
    df["track_strand"] = df["track_name"] + "_" + df["strand"]
    del df["track_name"], df["strand"]

    df_wide = df.pivot_table(
        index="variant_id",
        columns="track_strand",
        values="log10_median_of_exon_medians_alt",
        aggfunc="mean"
    ).reset_index()
    del df
    gc.collect()

    # fast O(1) lookup + concat broadcast (no fragmentation)
    if gene in df_gene_features.index:
        expr_row = df_gene_features.loc[gene, expr_columns]
    else:
        expr_row = pd.Series(np.nan, index=expr_columns)

    expr_df = pd.DataFrame(
        [expr_row.values] * len(df_wide),
        columns=expr_columns,
        index=df_wide.index
    )
    df_wide = pd.concat([df_wide, expr_df], axis=1)
    del expr_df, expr_row

    all_gene_features.append(df_wide)
    del df_wide

all_gene_features_merged = pd.concat(all_gene_features, ignore_index=True)
del all_gene_features
gc.collect()

print(f"Final shape: {all_gene_features_merged.shape}")
print(f"NaN rate: {all_gene_features_merged.isna().mean().mean():.1%}")
all_gene_features_merged.to_parquet(output_path, index=False)
print(f"Done! Saved to {output_path}")