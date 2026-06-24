import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.variant_scoring import (
    VariantScoringModel, Variant, Interval,
    CenterMaskScorer, OutputType, AggregationType,
    get_recommended_scorers,
)
from alphagenome_pytorch.variant_scoring.scorers.gene_mask import GeneMaskMode



# Load model (track_means are bundled in weights from convert_weights.py)
model = AlphaGenome.from_pretrained('data/model_related/model_all_folds.safetensors', device='cuda')

# Create scoring wrapper
scoring_model = VariantScoringModel(
    model,
    fasta_path='data/model_related/hg38.fa',
    gtf_path='data/annotations/gencode.v46.annotation.parquet',
    polya_path='data/annotations/gencode.v46.polyAs.linked.parquet',  # Optional, for PolyadenylationScorer
    default_organism='human',
)
scoring_model.load_all_metadata('track_metadata.parquet')

# load scorere that alphagenome used for eQTL scoring in the paper
from alphagenome_pytorch.variant_scoring import GeneMaskLFCScorer

rna_scorer = GeneMaskLFCScorer(
    requested_output=OutputType.RNA_SEQ,
    mask_mode=GeneMaskMode.EXONS,  # EXONS or BODY
    resolution=1,  # Default 128bp for gene-level
)

def score_variant_pytorch(variant_id):
    variant = Variant.from_str(variant_id, format="gtex")

    interval = Interval.centered_on(
        variant.chromosome,
        variant.position
    )

    scores = scoring_model.score_variant(
        interval=interval,
        variant=variant,
        scorers=[rna_scorer],
        to_cpu=True,
    )

    return scoring_model.tidy_scores(scores)

urls = [
    "https://storage.googleapis.com/alphagenome/evals/eqtl_variant_borzoi_coefficient_human_predictions.feather",
    "https://storage.googleapis.com/alphagenome/evals/eqtl_variant_catalogue_causality_gene_balanced_human_predictions.feather",
]

dfs = [pd.read_feather(url) for url in urls]


df_zeroshot = dfs[1]
df_zeroshot = df_zeroshot.dropna(subset=["target"])


import gc
import torch
from tqdm import tqdm

unique_variants = df_zeroshot["variant_id"].unique()
#unique_variants = unique_variants[:10]
unique_variants = unique_variants[:len(unique_variants) // 20]
variant_scores_dic = {}

for variant in tqdm(unique_variants):
    try:
        with torch.inference_mode():
            result = score_variant_pytorch(variant)
        variant_scores_dic[variant] = result
        
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA OOM at {variant}")
        torch.cuda.empty_cache()
        gc.collect()
        variant_scores_dic[variant] = None
        continue

    except ValueError as e:
        print(f"Skipping {variant}: {e}")
        variant_scores_dic[variant] = None

    except Exception as e:
        print(f"Failed {variant}: {type(e).__name__}: {e}")
        variant_scores_dic[variant] = None


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

# output dataframe to file
import pandas as pd

df_results = pd.DataFrame(df_zeroshot_results)
df_results.to_feather("results/pytorch/zeroshot_part1.feather")