#!/usr/bin/env python3
"""Indel-aware AlphaGenome track aggregation helpers.

This module mirrors the exon aggregation logic from
`compute_all_tracks_exon_medians.py`, but sources REF/ALT predictions from
`predict_variant` so insertions and deletions are handled with AlphaGenome's
own variant machinery and ALT-to-REF alignment.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from compute_all_tracks_exon_medians import (
    DEFAULT_ASSAY_TITLES,
    OUTPUT_TYPE_SPECS,
    _compute_chip_histone_nearest_bin_aggregations,
    _compute_exon_aggregations,
    _parse_half_open_intervals,
    normalize_requested_output_types,
)
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome_research.model.variant_scoring import (
    gene_mask_extractor as research_gene_mask_extractor,
)
from alphagenome_research.model.variant_scoring import (
    variant_scoring as research_variant_scoring,
)


DEFAULT_LOG_FC_PSEUDOCOUNT = 1e-3


def alphagenome_hf_repo_id(model_name: str) -> str:
    return f"google/alphagenome-{str(model_name).replace('_', '-').lower()}"


def load_model_from_hf_snapshot(
    *,
    dna_model_module: Any,
    model_name: str = "all_folds",
    hf_cache_dir: Optional[Path] = None,
    hf_local_files_only: bool = False,
) -> Any:
    import huggingface_hub

    snapshot_kwargs: Dict[str, Any] = {
        "repo_id": alphagenome_hf_repo_id(model_name),
    }
    if hf_cache_dir is not None:
        snapshot_kwargs["cache_dir"] = str(hf_cache_dir)
    if hf_local_files_only:
        snapshot_kwargs["local_files_only"] = True

    checkpoint_path = huggingface_hub.snapshot_download(**snapshot_kwargs)
    return dna_model_module.create(checkpoint_path)


def coerce_model_metadata_to_pandas_object(model: Any) -> Any:
    """Converts Arrow-backed string metadata to plain pandas object dtype.

    Some environments expose AlphaGenome metadata columns as Arrow extension
    arrays, which can break downstream code paths that rely on NumPy-style
    broadcasting (for example `score_variant(...).finalize_variant(...)`).
    """

    def _coerce_table(table: Any) -> None:
        if table is None:
            return
        for column in list(table.columns):
            if not pd.api.types.is_extension_array_dtype(table[column].dtype):
                continue
            table[column] = table[column].astype(object)

    metadata_by_organism = getattr(model, "_metadata", {})
    for organism_metadata in metadata_by_organism.values():
        for output_type in dna_output.OutputType:
            _coerce_table(organism_metadata.get(output_type))

    output_metadata_by_organism = getattr(model, "_output_metadata_by_organism", {})
    for output_metadata in output_metadata_by_organism.values():
        for output_type in dna_output.OutputType:
            _coerce_table(output_metadata.get(output_type))

    return model


def forward_oriented_exon_intervals(
    exon_intervals: Any,
    *,
    sequence_length: int,
    gene_strand: str = "+",
) -> List[Tuple[int, int]]:
    """Converts gene-oriented exon windows to forward genomic orientation."""
    parsed = _parse_half_open_intervals(exon_intervals, pd_module=pd)
    if str(gene_strand) != "-":
        return parsed

    oriented = []
    for start, end in parsed:
        oriented.append((int(sequence_length) - int(end), int(sequence_length) - int(start)))
    oriented.sort()
    return oriented


def build_forward_interval_from_orthology_row(row: Mapping[str, Any]) -> genome.Interval:
    chromosome = row.get("mapped_contig") if row.get("mapped_contig") is not None else row.get("contig")
    if chromosome is None or pd.isna(chromosome):
        raise ValueError("The orthology row is missing 'mapped_contig'/'contig'.")

    actual_start = row.get("actual_extract_start")
    actual_end = row.get("actual_extract_end")
    if pd.isna(actual_start) or pd.isna(actual_end):
        raise ValueError("The orthology row is missing 'actual_extract_start'/'actual_extract_end'.")

    return genome.Interval(
        chromosome=str(chromosome),
        start=int(actual_start) - 1,
        end=int(actual_end),
        strand=".",
    )


def build_variant_from_row(row: Mapping[str, Any]) -> genome.Variant:
    chromosome = row.get("Chromosome") or row.get("chromosome")
    position = row.get("pos") or row.get("position")
    reference_bases = row.get("ref") or row.get("reference_bases")
    alternate_bases = row.get("alt") or row.get("alternate_bases")
    if chromosome is None or position is None or reference_bases is None or alternate_bases is None:
        raise ValueError(
            "Variant row must contain chromosome/pos/ref/alt information."
        )

    variant_id = row.get("variant_id")
    return genome.Variant(
        chromosome=str(chromosome),
        position=int(position),
        reference_bases=str(reference_bases).upper(),
        alternate_bases=str(alternate_bases).upper(),
        name=None if variant_id is None else str(variant_id),
    )


def _prepare_assay_selection_for_strand(
    assay_data: Any,
    assay_type: str,
    assay_titles: Optional[Sequence[str]],
    *,
    gene_strand: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    meta = assay_data.metadata
    if meta is None or meta.empty:
        return None

    mask = np.ones(len(meta), dtype=bool)
    if assay_titles and ("Assay title" in meta.columns):
        mask &= meta["Assay title"].isin(list(assay_titles)).to_numpy()
    if gene_strand in {"+", "-"} and ("strand" in meta.columns):
        mask &= meta["strand"].isin([str(gene_strand), "."]).to_numpy()

    keep_idx = np.flatnonzero(mask)
    if keep_idx.size == 0:
        return None

    if "name" in meta.columns:
        track_names = meta.loc[mask, "name"].astype(str).tolist()
    else:
        track_names = [f"{assay_type}_track_{i}" for i in range(int(keep_idx.size))]

    if "Assay title" in meta.columns:
        track_assay_titles = meta.loc[mask, "Assay title"].astype(str).tolist()
    else:
        track_assay_titles = [""] * int(keep_idx.size)

    if "strand" in meta.columns:
        track_strands = meta.loc[mask, "strand"].astype(str).tolist()
    else:
        track_strands = ["."] * int(keep_idx.size)

    selected_metadata = meta.loc[mask].reset_index(drop=False).copy()

    return {
        "keep_idx": keep_idx.astype(int),
        "track_names": track_names,
        "track_assay_titles": track_assay_titles,
        "track_strands": track_strands,
        "selected_metadata": selected_metadata,
    }


def _resolve_output_type_enum(dna_model_module: Any, output_type_name: str) -> Any:
    output_type_enum = getattr(dna_model_module, "OutputType", None)
    if output_type_enum is not None:
        return getattr(output_type_enum, output_type_name)
    return getattr(dna_output.OutputType, output_type_name)


def _clean_and_log10(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    clean = np.where(np.isfinite(arr), arr, np.nan)
    log_vals = np.full(clean.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(clean) & (clean > 0)
    log_vals[valid] = np.log10(clean[valid] + 1e-6)
    return clean, log_vals


def _compute_ref_alt_aggregations(
    *,
    ref_track_values: np.ndarray,
    alt_track_values: np.ndarray,
    exon_intervals: Sequence[Tuple[int, int]],
    assay_type: str,
    sequence_length: int,
    log_fc_pseudocount: float,
) -> Tuple[Dict[str, np.ndarray], int]:
    if assay_type == "chip_histone":
        ref_aggs, selected_exon_bp = _compute_chip_histone_nearest_bin_aggregations(
            ref_track_values,
            list(exon_intervals),
            sequence_length,
            np_module=np,
        )
        alt_aggs, _ = _compute_chip_histone_nearest_bin_aggregations(
            alt_track_values,
            list(exon_intervals),
            sequence_length,
            np_module=np,
        )
    else:
        ref_aggs, selected_exon_bp = _compute_exon_aggregations(
            ref_track_values,
            list(exon_intervals),
            np_module=np,
        )
        alt_aggs, _ = _compute_exon_aggregations(
            alt_track_values,
            list(exon_intervals),
            np_module=np,
        )

    if not ref_aggs or not alt_aggs:
        return {}, 0

    combined: Dict[str, np.ndarray] = {}
    for agg_name, ref_values in ref_aggs.items():
        alt_values = alt_aggs[agg_name]
        ref_clean, ref_log10 = _clean_and_log10(ref_values)
        alt_clean, alt_log10 = _clean_and_log10(alt_values)
        combined[f"{agg_name}_ref"] = ref_clean
        combined[f"{agg_name}_alt"] = alt_clean
        combined[f"log10_{agg_name}_ref"] = ref_log10
        combined[f"log10_{agg_name}_alt"] = alt_log10
        combined[f"delta_{agg_name}"] = alt_clean - ref_clean
        combined[f"log_fc_{agg_name}"] = np.log(alt_clean + log_fc_pseudocount) - np.log(
            ref_clean + log_fc_pseudocount
        )

    return combined, int(selected_exon_bp)


def _assign_constant_metadata_column(
    df: pd.DataFrame,
    column: str,
    value: Any,
) -> None:
    """Assigns metadata as a constant column, even for list-like objects."""
    if isinstance(value, np.ndarray):
        stored_value: Any = value.tolist()
    else:
        stored_value = value

    if isinstance(stored_value, (list, tuple, dict, set)):
        df[column] = [repr(stored_value)] * len(df)
        return

    df[column] = stored_value


def score_variant_track_aggregations(
    model: Any,
    variant: genome.Variant,
    interval: genome.Interval,
    *,
    exon_intervals: Any,
    gene_strand: str = "+",
    organism: Any,
    dna_model_module: Any,
    ontology_terms: Optional[Iterable[str]] = None,
    assay_titles: Optional[Sequence[str]] = None,
    requested_output_types: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    log_fc_pseudocount: float = DEFAULT_LOG_FC_PSEUDOCOUNT,
) -> pd.DataFrame:
    """Scores one variant over one custom exon mask.

    The exon intervals must be 0-based half-open and expressed in the interval's
    forward genomic coordinate system.
    """
    if assay_titles is None:
        assay_titles = list(DEFAULT_ASSAY_TITLES)
    requested_output_types = normalize_requested_output_types(requested_output_types)
    sequence_length = int(interval.end - interval.start)
    parsed_exon_intervals = _parse_half_open_intervals(exon_intervals, pd_module=pd)

    requested_outputs = {
        _resolve_output_type_enum(dna_model_module, output_type)
        for output_type in requested_output_types
    }
    variant_output = model.predict_variant(
        interval=interval,
        variant=variant,
        organism=organism,
        requested_outputs=requested_outputs,
        ontology_terms=ontology_terms,
    )

    base_row = {
        "variant_id": variant.name or str(variant),
        "variant_label": str(variant),
        "scored_interval": str(interval),
        "gene_strand": str(gene_strand),
    }
    if extra_metadata:
        base_row.update(dict(extra_metadata))

    rows: List[pd.DataFrame] = []

    for output_type in requested_output_types:
        assay_type, assay_attr = OUTPUT_TYPE_SPECS[output_type]
        ref_track_data = getattr(variant_output.reference, assay_attr)
        alt_track_data = getattr(variant_output.alternate, assay_attr)
        if ref_track_data is None or alt_track_data is None:
            continue

        selection = _prepare_assay_selection_for_strand(
            ref_track_data,
            assay_type,
            assay_titles,
            gene_strand=gene_strand,
        )
        if not selection:
            continue

        ref_values = np.asarray(ref_track_data.values, dtype=np.float32)
        alt_values = np.asarray(alt_track_data.values, dtype=np.float32)
        if ref_values.ndim == 1:
            ref_values = ref_values[:, np.newaxis]
        if alt_values.ndim == 1:
            alt_values = alt_values[:, np.newaxis]

        alt_values = np.asarray(
            research_variant_scoring.align_alternate(alt_values, variant, interval),
            dtype=np.float32,
        )

        keep_idx = selection["keep_idx"]
        ref_selected = ref_values[:, keep_idx]
        alt_selected = alt_values[:, keep_idx]
        if ref_selected.shape[1] == 0 or alt_selected.shape[1] == 0:
            continue

        aggregations, selected_exon_bp = _compute_ref_alt_aggregations(
            ref_track_values=ref_selected,
            alt_track_values=alt_selected,
            exon_intervals=parsed_exon_intervals,
            assay_type=assay_type,
            sequence_length=sequence_length,
            log_fc_pseudocount=log_fc_pseudocount,
        )
        if not aggregations:
            continue

        n_tracks = int(ref_selected.shape[1])
        selected_metadata = selection["selected_metadata"].copy()
        selected_metadata = selected_metadata.rename(
            columns={"index": "track_metadata_idx"}
        )
        assay_df = pd.DataFrame(
            {
                "selected_exon_bp": np.full(n_tracks, selected_exon_bp, dtype=np.int64),
                "track_idx": keep_idx.astype(np.int64),
                "track_name": selection["track_names"],
                "track_strand": selection["track_strands"],
                "assay_type": np.full(n_tracks, assay_type),
                "assay_title": selection["track_assay_titles"],
                "assay_track_idx": np.arange(n_tracks, dtype=np.int64),
                "rna_seq_track_idx": keep_idx.astype(np.int64),
                "rna_seq_track_name": selection["track_names"],
            }
        )
        for metadata_col in selected_metadata.columns:
            if metadata_col in assay_df.columns:
                continue
            assay_df[metadata_col] = selected_metadata[metadata_col].to_numpy()
        for key, value in base_row.items():
            _assign_constant_metadata_column(assay_df, key, value)
        for key, values in aggregations.items():
            assay_df[key] = values
        rows.append(assay_df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def score_variant_track_aggregations_from_orthology_row(
    model: Any,
    row: Mapping[str, Any],
    *,
    organism: Any,
    dna_model_module: Any,
    ontology_terms: Optional[Iterable[str]] = None,
    assay_titles: Optional[Sequence[str]] = None,
    requested_output_types: Optional[Sequence[str]] = None,
    exon_intervals_col: str = "mapped_exon_intervals_in_window_0based_half_open",
) -> pd.DataFrame:
    interval = build_forward_interval_from_orthology_row(row)
    sequence = str(row.get("sequence", ""))
    gene_strand = str(row.get("mapped_strand") if row.get("mapped_strand") is not None else row.get("strand", "+"))
    exon_intervals = forward_oriented_exon_intervals(
        row.get(exon_intervals_col),
        sequence_length=len(sequence) if sequence else int(interval.end - interval.start),
        gene_strand=gene_strand,
    )
    variant = build_variant_from_row(row)

    extra_metadata = {
        key: row.get(key)
        for key in (
            "ortholog_geneid",
            "taxId",
            "taxname",
            "commonName",
            "species",
            "assembly",
            "mapped_contig",
            "mapped_strand",
            "variant_row_idx",
            "variant_id",
            "Chromosome",
            "pos",
            "ref",
            "alt",
            "region",
        )
        if key in row
    }
    extra_metadata["forward_scored_interval"] = str(interval)
    extra_metadata["forward_exon_intervals"] = exon_intervals

    return score_variant_track_aggregations(
        model,
        variant,
        interval,
        exon_intervals=exon_intervals,
        gene_strand=gene_strand,
        organism=organism,
        dna_model_module=dna_model_module,
        ontology_terms=ontology_terms,
        assay_titles=assay_titles,
        requested_output_types=requested_output_types,
        extra_metadata=extra_metadata,
    )


def extract_interval_gene_exon_masks_from_gtf(
    gtf: pd.DataFrame,
    interval: genome.Interval,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extracts the same exon masks used by AlphaGenome gene-mask scorers."""
    gtf = gtf.copy()
    non_gene_gene_ids = set(gtf.loc[gtf["Feature"] != "gene", "gene_id"].dropna().astype(str))
    gene_row_gene_ids = set(gtf.loc[gtf["Feature"] == "gene", "gene_id"].dropna().astype(str))
    missing_gene_rows = non_gene_gene_ids - gene_row_gene_ids
    if missing_gene_rows:
        transcript_like = gtf[
            gtf["gene_id"].astype(str).isin(missing_gene_rows) & gtf["Feature"].ne("gene")
        ].copy()
        synthesized_gene_rows = (
            transcript_like.groupby("gene_id", sort=False)
            .agg(
                {
                    "Chromosome": "first",
                    "Start": "min",
                    "End": "max",
                    "Strand": "first",
                    "gene_name": "first",
                    "gene_type": "first",
                }
            )
            .reset_index()
        )
        synthesized_gene_rows["Feature"] = "gene"
        for column in gtf.columns:
            if column not in synthesized_gene_rows.columns:
                synthesized_gene_rows[column] = pd.NA
        synthesized_gene_rows = synthesized_gene_rows[gtf.columns]
        gtf = pd.concat([gtf, synthesized_gene_rows], ignore_index=True)

    extractor = research_gene_mask_extractor.GeneMaskExtractor(
        gtf=gtf,
        gene_mask_type=research_gene_mask_extractor.GeneMaskType.EXONS,
        gene_query_type=research_gene_mask_extractor.GeneQueryType.INTERVAL_CONTAINED,
    )
    gene_mask_matrix, gene_metadata = extractor.extract(interval)
    gene_metadata = gene_metadata.reset_index(drop=True).copy().rename(
        columns={"strand": "gene_strand"}
    )
    gene_metadata["selected_exon_bp"] = gene_mask_matrix.sum(axis=0).astype(int)
    return np.asarray(gene_mask_matrix, dtype=np.float32), gene_metadata


def binary_mask_to_intervals(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Converts a 1D boolean mask into 0-based half-open intervals."""
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 1:
        raise ValueError(f"Expected a 1D mask, got shape {mask_array.shape}.")

    padded = np.concatenate(
        [np.asarray([False]), mask_array, np.asarray([False])]
    )
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _mask_matrix_to_track_resolution(
    gene_mask_matrix: np.ndarray,
    resolution: int,
) -> np.ndarray:
    resolution = int(resolution)
    if resolution == 1:
        return np.asarray(gene_mask_matrix, dtype=np.float32)

    n_bp, n_genes = gene_mask_matrix.shape
    remainder = n_bp % resolution
    if remainder:
        pad = resolution - remainder
        gene_mask_matrix = np.pad(
            gene_mask_matrix,
            ((0, pad), (0, 0)),
            constant_values=False,
        )
    return gene_mask_matrix.reshape(-1, resolution, n_genes).any(axis=1).astype(np.float32)


def _aggregate_values_over_gene_masks(
    track_values: np.ndarray,
    *,
    gene_mask_matrix: np.ndarray,
    resolution: int,
    aggregation: str = "mean",
) -> np.ndarray:
    track_values = np.asarray(track_values, dtype=np.float32)
    if track_values.ndim == 1:
        track_values = track_values[:, np.newaxis]
    if track_values.ndim != 2:
        raise ValueError(f"Expected 2D track values, got {track_values.shape}.")

    gene_mask_bins = _mask_matrix_to_track_resolution(gene_mask_matrix, resolution)
    if gene_mask_bins.shape[0] != track_values.shape[0]:
        raise ValueError(
            "The gene mask and track values do not share the same number of bins: "
            f"{gene_mask_bins.shape[0]} vs {track_values.shape[0]}."
        )

    masked_sums = np.einsum("lt,lg->gt", track_values, gene_mask_bins)
    if aggregation == "sum":
        return masked_sums
    if aggregation == "mean":
        mask_sizes = gene_mask_bins.sum(axis=0)[:, np.newaxis]
        return masked_sums / mask_sizes
    raise ValueError("aggregation must be one of {'mean', 'sum'}.")


def _format_gene_track_values(
    aggregated_values: np.ndarray,
    *,
    gene_metadata: pd.DataFrame,
    track_metadata: pd.DataFrame,
    allele: str,
    variant: genome.Variant,
    interval: genome.Interval,
    aggregation: str,
    only_matching_strand: bool = True,
) -> pd.DataFrame:
    track_metadata = track_metadata.reset_index(drop=True).copy().rename(
        columns={"name": "rna_seq_track_name", "strand": "rna_seq_track_strand"}
    )
    track_metadata["rna_seq_track_idx"] = np.arange(len(track_metadata), dtype=int)

    repeated_gene_metadata = gene_metadata.loc[
        gene_metadata.index.repeat(len(track_metadata))
    ].reset_index(drop=True)
    tiled_track_metadata = pd.concat(
        [track_metadata] * len(gene_metadata),
        ignore_index=True,
    )
    result = pd.concat([repeated_gene_metadata, tiled_track_metadata], axis=1)
    result["allele"] = allele
    result["aggregation"] = aggregation
    result["variant_id"] = variant.name or str(variant)
    result["variant_label"] = str(variant)
    result["scored_interval"] = str(interval)
    result["signal"] = np.asarray(aggregated_values, dtype=np.float64).reshape(-1)
    result["log10_signal"] = np.log10(result["signal"].to_numpy() + 1e-6)
    result["matches_gene_strand"] = (
        result["rna_seq_track_strand"].eq(".")
        | result["rna_seq_track_strand"].eq(result["gene_strand"])
    )

    if only_matching_strand:
        result = result[result["matches_gene_strand"]].reset_index(drop=True)
    return result


def summarize_variant_rnaseq_over_interval_genes(
    model: Any,
    variant: genome.Variant,
    interval: genome.Interval,
    *,
    gtf: pd.DataFrame,
    organism: Any,
    dna_model_module: Any,
    ontology_terms: Optional[Iterable[str]] = None,
    aggregation: str = "mean",
    only_matching_strand: bool = True,
    variant_output: Optional[Any] = None,
) -> Dict[str, Any]:
    """Manual RNA-seq gene-mask aggregation matching AlphaGenome's scorer."""
    aggregation = aggregation.lower()
    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be one of {'mean', 'sum'}.")

    gene_mask_matrix, gene_metadata = extract_interval_gene_exon_masks_from_gtf(
        gtf,
        interval,
    )
    if variant_output is None:
        variant_output = model.predict_variant(
            interval=interval,
            variant=variant,
            organism=organism,
            requested_outputs=[_resolve_output_type_enum(dna_model_module, "RNA_SEQ")],
            ontology_terms=ontology_terms,
        )

    ref_rna = variant_output.reference.rna_seq
    alt_rna = variant_output.alternate.rna_seq
    if ref_rna is None or alt_rna is None:
        raise ValueError("AlphaGenome did not return RNA-seq predictions.")
    if int(ref_rna.resolution) != int(alt_rna.resolution):
        raise ValueError("REF and ALT RNA-seq predictions do not use the same resolution.")

    ref_values = np.asarray(ref_rna.values, dtype=np.float32)
    alt_values = np.asarray(alt_rna.values, dtype=np.float32)
    alt_values = np.asarray(
        research_variant_scoring.align_alternate(alt_values, variant, interval),
        dtype=np.float32,
    )

    ref_aggregated = _aggregate_values_over_gene_masks(
        ref_values,
        gene_mask_matrix=gene_mask_matrix,
        resolution=int(ref_rna.resolution),
        aggregation=aggregation,
    )
    alt_aggregated = _aggregate_values_over_gene_masks(
        alt_values,
        gene_mask_matrix=gene_mask_matrix,
        resolution=int(alt_rna.resolution),
        aggregation=aggregation,
    )

    ref_gene_track_values = _format_gene_track_values(
        ref_aggregated,
        gene_metadata=gene_metadata,
        track_metadata=ref_rna.metadata,
        allele="ref",
        variant=variant,
        interval=interval,
        aggregation=aggregation,
        only_matching_strand=only_matching_strand,
    )
    alt_gene_track_values = _format_gene_track_values(
        alt_aggregated,
        gene_metadata=gene_metadata,
        track_metadata=alt_rna.metadata,
        allele="alt",
        variant=variant,
        interval=interval,
        aggregation=aggregation,
        only_matching_strand=only_matching_strand,
    )

    merge_keys = [
        col
        for col in ref_gene_track_values.columns
        if col in alt_gene_track_values.columns and col not in {"allele", "signal", "log10_signal"}
    ]
    ref_alt_gene_track_values = ref_gene_track_values.drop(columns=["allele"]).rename(
        columns={"signal": f"ref_{aggregation}_signal", "log10_signal": "log10_ref_signal"}
    ).merge(
        alt_gene_track_values.drop(columns=["allele"]).rename(
            columns={"signal": f"alt_{aggregation}_signal", "log10_signal": "log10_alt_signal"}
        ),
        on=merge_keys,
        how="inner",
    )
    ref_alt_gene_track_values["delta_signal"] = (
        ref_alt_gene_track_values[f"alt_{aggregation}_signal"]
        - ref_alt_gene_track_values[f"ref_{aggregation}_signal"]
    )
    ref_alt_gene_track_values["log_fc"] = np.log(
        ref_alt_gene_track_values[f"alt_{aggregation}_signal"].to_numpy() + DEFAULT_LOG_FC_PSEUDOCOUNT
    ) - np.log(
        ref_alt_gene_track_values[f"ref_{aggregation}_signal"].to_numpy() + DEFAULT_LOG_FC_PSEUDOCOUNT
    )
    ref_alt_gene_track_values["abs_log_fc"] = ref_alt_gene_track_values["log_fc"].abs()
    ref_alt_gene_track_values = ref_alt_gene_track_values.sort_values(
        "abs_log_fc",
        ascending=False,
    ).reset_index(drop=True)

    return {
        "variant_output": variant_output,
        "gene_mask_matrix": gene_mask_matrix,
        "gene_metadata": gene_metadata,
        "ref_gene_track_values": ref_gene_track_values,
        "alt_gene_track_values": alt_gene_track_values,
        "ref_alt_gene_track_values": ref_alt_gene_track_values,
    }
