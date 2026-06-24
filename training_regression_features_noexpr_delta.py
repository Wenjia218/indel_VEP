import pandas as pd
from tqdm import tqdm
import os
import gc
import time

with open("results_logistic/gene_list.txt", "r") as f:
    gene_list = [line.strip() for line in f.readlines()]



base_path = (
    "/s/project/benchmark-lm/data/orthology_primates/preds"
)
important_columns = ["track_name", "strand", "variant_id", "delta_median_of_exon_medians"]


all_gene_features = []



for i, gene in enumerate(tqdm(gene_list, desc="Processing genes"), start=1):
    f = os.path.join(
        base_path,
        f"gnomad_indels_scores/all_tracks_variant_gene_scores_all_species_{gene}.parquet"
    )

    if not os.path.exists(f):
        print("  Skipping, file not found:", f)
        continue
    
    df = pd.read_parquet(f, columns=important_columns)
    
    # remove strand = -
    #df = df[df["strand"] != "-"]

    df = df.drop(columns=["strand"])

    # Pivot track_name and strand as columns
    df_wide = df.pivot_table(
        index="variant_id",
        columns="track_name",
        values=["delta_median_of_exon_medians"],
        aggfunc="mean"
    )
    df_wide.columns = [f"{track}_{val.split('_')[-1]}" for val, track in df_wide.columns]
    df_wide = df_wide.reset_index()
    
    all_gene_features.append(df_wide)

    del df, df_wide
    gc.collect()

# Stack vertically — safe since variants never overlap
all_gene_features_merged = pd.concat(all_gene_features, axis=0).reset_index()
print(f"Final shape: {all_gene_features_merged.shape}")
print(f"NaN rate: {all_gene_features_merged.isna().mean().mean():.1%}")

# output to file
output_path = "results_logistic/VS_per_track_strand_delta_with-.parquet"
print("Saving to parquet...")
all_gene_features_merged.to_parquet(output_path, index=False)
print(f"Done! Saved to {output_path}")