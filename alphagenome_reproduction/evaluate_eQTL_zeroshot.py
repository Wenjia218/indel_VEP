from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import pandas as pd
# @title Imports

import functools
import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score

# @title Helper functions.

def calculate_tissue_weighted_metric(
    predictions_df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    prediction_col: str = 'prediction',
    target_col: str = 'target',
    tissue_col: str = 'tissue'
) -> float:
    """
    Calculates a tissue-weighted mean for a given metric function from a DataFrame.

    Args:
        predictions_df: DataFrame containing predictions, targets, and tissue info.
        metric_fn: A function that accepts two np.ndarrays (y_true, y_pred)
                   and returns a single float metric.
                   Note: For scipy functions that return tuples (e.g., spearmanr),
                   you must wrap them in a lambda function (see example).
        prediction_col: Name of the column with prediction scores.
        target_col: Name of the column with target labels/values.
        tissue_col: Name of the column indicating the tissue/group.

    Returns:
        The calculated tissue-weighted mean metric, or np.nan if unable to calculate.
    """
    if predictions_df is None or predictions_df.empty:
        print("Warning: predictions_df is empty, returning NaN.")
        return float('nan')

    required_cols = [prediction_col, target_col, tissue_col]
    if not all(col in predictions_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    tissue_results = []

    # Group by tissue and calculate the metric for each one.
    for tissue_name, group_df in predictions_df.groupby(tissue_col):

        # Drop rows where target or prediction is NaN.
        clean_group = group_df.dropna(subset=[target_col, prediction_col])

        num_variants = len(clean_group)

        # Skip tissues with < 2 variants (can't calculate metrics).
        if num_variants < 2:
            # print(f"Skipping tissue {tissue_name} (variants={num_variants})")
            continue

        try:
            # Calculate the metric for this one tissue using the passed function
            metric_value = metric_fn(
                clean_group[target_col], clean_group[prediction_col]
            )

            # Ensure metric is a valid number.
            if not np.isfinite(metric_value):
                print(f"Skipping tissue {tissue_name} (metric_fn returned non-finite value)")
                continue

            tissue_results.append({
                'tissue': tissue_name,
                'metric_value': metric_value,
                'count': num_variants  # This is the "weight".
            })
        except ValueError as e:
            # This will catch single-class errors for AUROC/AUPRC
            # and other potential issues from metric_fn.
            print(f"Could not calculate metric for tissue {tissue_name}: {e}")
            continue # Skip tissue if metric calculation fails.

    if not tissue_results:
        print("Warning: No tissues had scorable metric values, returning NaN.")
        return float('nan')

    # Create a DataFrame of the per-tissue metrics.
    metrics_df = pd.DataFrame(tissue_results)

    # Calculate the final weighted mean.
    weighted_sum = (metrics_df['metric_value'] * metrics_df['count']).sum()
    total_count = metrics_df['count'].sum()

    if total_count == 0:
         print("Warning: Total count for weighting is zero, returning NaN.")
         return float('nan')

    weighted_mean_metric = weighted_sum / total_count
    return weighted_mean_metric


def paqtl_auprc(df):
  SEED = 0

  # 1. Separate positives and negatives.
  pos = df[df['target'] == 1]
  neg = df[df['target'] == 0]

  # 2. Merge on 'PI' to find all valid matched pairs.
  matched = pd.merge(
      pos,
      neg,
      on='PI',
      how='inner',
      suffixes=('_pos', '_neg')
  )

  # 3. Group by 'PI' and sample exactly one pair per group.
  # This ensures 1:1 matching controlled by PI.
  sampled_pairs = matched.groupby('PI').sample(n=1, random_state=SEED)

  # 4. Reconstruct a single dataframe for AUPRC calculation
  # We stack the positive and negative parts of the sampled pairs back together.
  df_sampled = pd.concat([
      sampled_pairs[['prediction_pos', 'target_pos']].rename(
          columns={'prediction_pos': 'prediction', 'target_pos': 'target'}
      ),
      sampled_pairs[['prediction_neg', 'target_neg']].rename(
          columns={'prediction_neg': 'prediction', 'target_neg': 'target'}
      )
  ])
  return metrics.average_precision_score(
      df_sampled['target'], df_sampled['prediction'])

auroc_fn = roc_auc_score
auprc_fn = average_precision_score

spearman_fn = lambda y_true, y_pred: spearmanr(y_true, y_pred)[0]

dna_model = dna_client.create("AIzaSyAJiVhyZcj7HQsrUIQx_TOIlFtTsrgcT2c")


def score_variant(variant_id): 
    parts = variant_id.split("_")
    if len(parts) == 5:
        chrom, pos, ref, alt, ext = parts
    elif len(parts) == 4:
        chrom, pos, ref, alt = parts
        ext = None
    else:
        raise ValueError(f"Unexpected variant_id format: {variant_id}")
    # build variant object
    variant = genome.Variant(
        chromosome=chrom,
        position=int(pos),
        reference_bases=ref,
        alternate_bases=alt,
    )
    # Create a 1MB interval centered on the variant.
    interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']
    variant_scores = dna_model.score_variant(
        interval=interval, variant=variant, variant_scorers=[variant_scorer]
    )
    variant_scores = variant_scorers.tidy_scores(variant_scores)
    
    return variant_scores

urls = ["https://storage.googleapis.com/alphagenome/evals/eqtl_variant_borzoi_coefficient_human_predictions.feather",
        "https://storage.googleapis.com/alphagenome/evals/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather"]

dfs = [pd.read_feather(url) for url in urls]

df_zeroshot = dfs[1]
df_zeroshot = df_zeroshot.dropna(subset=['target'])

# take half of variants to reduce runtime

unique_variants = df_zeroshot["variant_id"].unique()

# run half of the variants
unique_variants = unique_variants[len(unique_variants) // 2 + 1:len(unique_variants)]
variant_scores_dic = {}


from tqdm import tqdm
for variant in tqdm(unique_variants):
    print("Scoring variant:", variant, flush=True)
    variant_scores_dic[variant] = score_variant(variant)

df_zeroshot_results = []

for row_nr in range(len(df_zeroshot)):
    row = df_zeroshot.iloc[row_nr]
    variant = row.variant_id
    if variant not in variant_scores_dic:
        continue
    variant_scores = variant_scores_dic[variant]
    mask = (
        (variant_scores["gene_id"] == row.gene_id.split('.')[0]) &
        (variant_scores["gtex_tissue"] == row.tissue)
    )
    filtered = variant_scores.loc[mask, "raw_score"]
    new_prediction = filtered.iloc[0] if not filtered.empty else np.nan
    # add prediction from model to the row and then collect row in a new dataframe for all variants
    new_row = row.copy()
    new_row['prediction_new'] = new_prediction

    df_zeroshot_results.append(new_row)
    

df_results = pd.DataFrame(df_zeroshot_results)
df_results.to_feather("results/client/df_zeroshot_results_part2.feather")
