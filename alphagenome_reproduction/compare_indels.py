import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
print(jax.devices())
print("default backend:", jax.default_backend())

from alphagenome.data import genome
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
from alphagenome.models import variant_scorers
from alphagenome.models import dna_client

import matplotlib.pyplot as plt
import functools
import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score

# @title Eval configs.
url = "https://storage.googleapis.com/alphagenome/evals/clinvar_noncoding_predictions.feather"
df = pd.read_feather(url)

df.head()

def parse_variant(variant_id):
    parts = variant_id.split("_")
    if len(parts) >= 4:
        chrom, pos, ref, alt = parts[:4]
        return ref, alt
    return None, None

def classify_variant(variant_id):
    ref, alt = parse_variant(variant_id)
    
    if ref is None:
        return "unknown"
    
    if len(ref) == len(alt) == 1:
        return "snp"
    elif len(alt) > len(ref):
        return "insertion"
    elif len(ref) > len(alt):
        return "deletion"
    else:
        return "other"
    
df["variant_type"] = df["variant_id"].apply(classify_variant)

insertions = df[df["variant_type"] == "insertion"].drop_duplicates("variant_id").head(100)
deletions = df[df["variant_type"] == "deletion"].drop_duplicates("variant_id").head(100)

variant_insertions = insertions["variant_id"].tolist()
variant_deletions = deletions["variant_id"].tolist()

variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ'] 

# Flags to improve determinism.
os.environ['XLA_FLAGS'] = ' '.join([
    '--xla_gpu_deterministic_ops',
    '--xla_gpu_enable_scatter_determinism_expander=True',
    '--xla_gpu_enable_triton_gemm=False',
])
# Increase GPU and CPU memory to reduce out of memory errors.
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

model = dna_model.create_from_huggingface('all_folds')

def score_variant(variant_id: str) -> pd.DataFrame: 

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
    interval = variant.reference_interval.resize(2**20)

    variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ'] 
    variant_scores = model.score_variant(
        interval=interval, 
        variant=variant, 
        variant_scorers=[variant_scorer],
        organism=dna_client.Organism.HOMO_SAPIENS,
    )
    variant_scores = variant_scorers.tidy_scores(variant_scores)
    
    return variant_scores


unique_variants = insertions["variant_id"]

variant_scores_dic = {}

from tqdm import tqdm
for variant in tqdm(unique_variants):
    variant_scores_dic[variant] = score_variant(variant)

# only take one gene for each variant

insertions_results_local = []

for variant in variant_scores_dic:
    df = variant_scores_dic[variant]
    if df is None or df.empty:
        continue
    insertions_results_local.append(df)

# convert list of rows → DataFrame
insertions_results_local = pd.concat(insertions_results_local, ignore_index=True)
    
cols = ["variant_id", "scored_interval", "ontology_curie", "gene_id", "raw_score"]

df_save = insertions_results_local[cols].copy()

df_save["variant_id"] = df_save["variant_id"].astype(str)
df_save["scored_interval"] = df_save["scored_interval"].astype(str)

df_save.to_feather("results/indel_compare/local_insertions.feather")