#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_GENES_CSV = Path("/s/project/benchmark-lm/selection_inference/benchmarklm/traitgym_geneid_nearest_genes.csv")
DEFAULT_GENES_COL = "ncbi_gene_id"
DEFAULT_ORTHOLOGY_DIR = Path("/s/project/benchmark-lm/data/orthology_primates")
DEFAULT_PREDS_DIR = Path("/s/project/benchmark-lm/data/orthology_primates/preds")
DEFAULT_INPUT_TEMPLATE = "orthology_exon_windows_primates_ncbi_id_{gene_id}.parquet"
DEFAULT_OUTPUT_TEMPLATE = "all_tracks_exon_medians_primates_ncbi_id_{gene_id}.parquet"
DEFAULT_MODEL_NAME = "all_folds"
DEFAULT_ASSAY_TITLES = [
    "ATAC-seq",
    "DNase-seq",
    "polyA plus RNA-seq",
    "total RNA-seq",
    "Histone ChIP-seq",
]
OUTPUT_TYPE_SPECS = {
    "ATAC": ("atac", "atac"),
    "DNASE": ("dnase", "dnase"),
    "RNA_SEQ": ("rna_seq", "rna_seq"),
    "CHIP_HISTONE": ("chip_histone", "chip_histone"),
}
OUTPUT_TYPE_ALIASES = {
    "ATAC_SEQ": "ATAC",
    "DNASE_SEQ": "DNASE",
    "RNA": "RNA_SEQ",
    "RNASEQ": "RNA_SEQ",
    "CHIPHISTONE": "CHIP_HISTONE",
    "HISTONE_CHIP": "CHIP_HISTONE",
}
DEFAULT_REQUESTED_OUTPUT_TYPES = tuple(OUTPUT_TYPE_SPECS.keys())

ALL_SELECTED_ONTOLOGY_TERMS = ['CL:0000084',
 'CL:0000100',
 'CL:0000236',
 'CL:0000623',
 'CL:0000624',
 'CL:0000625',
 'CL:0000787',
 'CL:0000788',
 'CL:0000792',
 'CL:0000895',
 'CL:0000899',
 'CL:0000900',
 'CL:0000907',
 'CL:0000913',
 'CL:1001606',
 'CLO:0013950',
 'CLO:0014043',
 'CLO:0014078',
 'CLO:0014080',
 'CLO:0014083',
 'CLO:0014097',
 'CLO:0014163',
 'CLO:0014886',
 'CLO:0014941',
 'CLO:0015884',
 'CLO:0015895',
 'CLO:0015900',
 'CLO:0015908',
 'CLO:0015947',
 'CLO:0024877',
 'CLO:0024886',
 'CLO:0024890',
 'CLO:0024898',
 'CLO:0024919',
 'CLO:0024933',
 'CLO:0024937',
 'CLO:0025816',
 'CLO:0026238',
 'CLO:0026245',
 'CLO:0026249',
 'CLO:0026250',
 'CLO:0026253',
 'CLO:0026256',
 'CLO:0026258',
 'CLO:0026782',
 'CLO:0026790',
 'CLO:0026824',
 'CLO:0026846',
 'CLO:0026863',
 'CLO:0026874',
 'CLO:0026877',
 'CLO:0027429',
 'CLO:0027432',
 'CLO:0027443',
 'CLO:0027445',
 'EFO:0001086',
 'EFO:0001187',
 'EFO:0001196',
 'EFO:0001203',
 'EFO:0001221',
 'EFO:0002067',
 'EFO:0002074',
 'EFO:0002713',
 'EFO:0002784',
 'EFO:0002824',
 'EFO:0005332',
 'EFO:0007074',
 'EFO:0007610',
 'EFO:0007950',
 'EFO:0009747',
 'EFO:0010835',
 'EFO:0010836',
 'EFO:0010837',
 'EFO:0010838',
 'EFO:0010839',
 'EFO:0010840',
 'EFO:0010841',
 'EFO:0010842',
 'EFO:0010843',
 'EFO:0010844',
 'EFO:0010846',
 'EFO:0010847',
 'EFO:0010848',
 'EFO:0010850',
 'EFO:0010851',
 'EFO:0010852',
 'EFO:0010853',
 'EFO:0010855',
 'EFO:0010856',
 'EFO:0010857',
 'EFO:0010858',
 'EFO:0010859',
 'EFO:0010860',
 'EFO:0010861',
 'EFO:0010862',
 'EFO:0010863',
 'EFO:0010864',
 'EFO:0010865',
 'EFO:0010866',
 'EFO:0010867',
 'EFO:0010868',
 'EFO:0010869',
 'EFO:0010870',
 'EFO:0010871',
 'EFO:0010872',
 'EFO:0010873',
 'EFO:0010874',
 'EFO:0010875',
 'EFO:0010876',
 'EFO:0010878',
 'EFO:0010879',
 'EFO:0010880',
 'EFO:0010881',
 'EFO:0010882',
 'EFO:0010883',
 'EFO:0010884',
 'EFO:0010885',
 'EFO:0010886',
 'EFO:0010888',
 'EFO:0010889',
 'NTR:0000491',
 'NTR:0000492',
 'NTR:0000494',
 'UBERON:0000056',
 'UBERON:0000317',
 'UBERON:0000945',
 'UBERON:0000992',
 'UBERON:0000995',
 'UBERON:0001072',
 'UBERON:0001114',
 'UBERON:0001115',
 'UBERON:0001150',
 'UBERON:0001157',
 'UBERON:0001159',
 'UBERON:0001259',
 'UBERON:0001264',
 'UBERON:0001322',
 'UBERON:0002046',
 'UBERON:0002048',
 'UBERON:0002078',
 'UBERON:0002079',
 'UBERON:0002080',
 'UBERON:0002084',
 'UBERON:0002099',
 'UBERON:0002106',
 'UBERON:0002107',
 'UBERON:0002113',
 'UBERON:0002168',
 'UBERON:0002170',
 'UBERON:0002171',
 'UBERON:0002190',
 'UBERON:0002369',
 'UBERON:0002394',
 'UBERON:0002469',
 'UBERON:0003889',
 'UBERON:0004550',
 'UBERON:0004992',
 'UBERON:0005033',
 'UBERON:0007610',
 'UBERON:0008367',
 'UBERON:0008450',
 'UBERON:0008952',
 'UBERON:0008953',
 'UBERON:0008971',
 'UBERON:0010414',
 'UBERON:0011907',
 'UBERON:0015143',
 'CL:0000047',
 'CL:0000115',
 'CL:0000127',
 'CL:0000134',
 'CL:0000136',
 'CL:0000138',
 'CL:0000182',
 'CL:0000187',
 'CL:0000188',
 'CL:0000223',
 'CL:0000312',
 'CL:0000351',
 'CL:0000515',
 'CL:0000545',
 'CL:0000546',
 'CL:0000632',
 'CL:0000653',
 'CL:0000679',
 'CL:0000706',
 'CL:0000746',
 'CL:0000823',
 'CL:0000837',
 'CL:0000862',
 'CL:0000863',
 'CL:0000897',
 'CL:0000904',
 'CL:0000905',
 'CL:0000909',
 'CL:0001042',
 'CL:0001044',
 'CL:0001054',
 'CL:0001059',
 'CL:0002038',
 'CL:0002061',
 'CL:0002231',
 'CL:0002252',
 'CL:0002304',
 'CL:0002306',
 'CL:0002327',
 'CL:0002328',
 'CL:0002351',
 'CL:0002372',
 'CL:0002518',
 'CL:0002536',
 'CL:0002547',
 'CL:0002548',
 'CL:0002550',
 'CL:0002551',
 'CL:0002552',
 'CL:0002553',
 'CL:0002555',
 'CL:0002557',
 'CL:0002558',
 'CL:0002565',
 'CL:0002584',
 'CL:0002586',
 'CL:0002590',
 'CL:0002603',
 'CL:0002604',
 'CL:0002606',
 'CL:0002618',
 'CL:0010001',
 'CL:0011012',
 'CL:0011019',
 'CL:0011020',
 'CL:0011021',
 'CL:0011022',
 'CL:1000507',
 'CL:1000892',
 'CL:1001568',
 'CL:1001608',
 'CL:2000000',
 'CL:2000010',
 'CL:2000013',
 'CL:2000016',
 'CL:2000017',
 'CL:2000041',
 'CL:2000043',
 'CL:2000044',
 'CL:2000045',
 'CLO:0003771',
 'CLO:0007045',
 'CLO:0011092',
 'CLO:0016536',
 'CLO:0018812',
 'CLO:0018813',
 'CLO:0021511',
 'CLO:0021752',
 'CLO:0021754',
 'CLO:0033902',
 'CLO:0033911',
 'CLO:0034837',
 'EFO:0000681',
 'EFO:0001099',
 'EFO:0001193',
 'EFO:0001200',
 'EFO:0001232',
 'EFO:0001247',
 'EFO:0002059',
 'EFO:0002083',
 'EFO:0002101',
 'EFO:0002106',
 'EFO:0002108',
 'EFO:0002140',
 'EFO:0002150',
 'EFO:0002179',
 'EFO:0002184',
 'EFO:0002234',
 'EFO:0002246',
 'EFO:0002285',
 'EFO:0002322',
 'EFO:0002323',
 'EFO:0002330',
 'EFO:0002779',
 'EFO:0002783',
 'EFO:0002785',
 'EFO:0002788',
 'EFO:0002789',
 'EFO:0002790',
 'EFO:0002791',
 'EFO:0002793',
 'EFO:0002798',
 'EFO:0002819',
 'EFO:0002847',
 'EFO:0002860',
 'EFO:0003037',
 'EFO:0003042',
 'EFO:0003044',
 'EFO:0003045',
 'EFO:0003072',
 'EFO:0005236',
 'EFO:0005282',
 'EFO:0005283',
 'EFO:0005337',
 'EFO:0005338',
 'EFO:0005345',
 'EFO:0005352',
 'EFO:0005384',
 'EFO:0005696',
 'EFO:0005697',
 'EFO:0005698',
 'EFO:0005704',
 'EFO:0005707',
 'EFO:0005709',
 'EFO:0005710',
 'EFO:0005712',
 'EFO:0005714',
 'EFO:0005715',
 'EFO:0005719',
 'EFO:0005720',
 'EFO:0005721',
 'EFO:0005722',
 'EFO:0005723',
 'EFO:0005724',
 'EFO:0005725',
 'EFO:0005726',
 'EFO:0005901',
 'EFO:0005903',
 'EFO:0005904',
 'EFO:0005913',
 'EFO:0006270',
 'EFO:0006639',
 'EFO:0006711',
 'EFO:0007098',
 'EFO:0007598',
 'EFO:0007599',
 'EFO:0007600',
 'EFO:0007748',
 'EFO:0007749',
 'EFO:0007750',
 'EFO:0007952',
 'EFO:0009318',
 'EFO:0009500',
 'NTR:0000474',
 'NTR:0000493',
 'NTR:0000512',
 'NTR:0000521',
 'NTR:0000522',
 'NTR:0000523',
 'NTR:0000524',
 'NTR:0000525',
 'NTR:0000856',
 'NTR:0001484',
 'UBERON:0000059',
 'UBERON:0000473',
 'UBERON:0000947',
 'UBERON:0000948',
 'UBERON:0000955',
 'UBERON:0000966',
 'UBERON:0000970',
 'UBERON:0000981',
 'UBERON:0000988',
 'UBERON:0001211',
 'UBERON:0001224',
 'UBERON:0001323',
 'UBERON:0001383',
 'UBERON:0001496',
 'UBERON:0001499',
 'UBERON:0001515',
 'UBERON:0001621',
 'UBERON:0001723',
 'UBERON:0001774',
 'UBERON:0001870',
 'UBERON:0001874',
 'UBERON:0001875',
 'UBERON:0001954',
 'UBERON:0001987',
 'UBERON:0002037',
 'UBERON:0002108',
 'UBERON:0002129',
 'UBERON:0002167',
 'UBERON:0002240',
 'UBERON:0002324',
 'UBERON:0002331',
 'UBERON:0002367',
 'UBERON:0002370',
 'UBERON:0002626',
 'UBERON:0002740',
 'UBERON:0003124',
 'UBERON:0003662',
 'UBERON:0003663',
 'UBERON:0004264',
 'UBERON:0004538',
 'UBERON:0004539',
 'UBERON:0005270',
 'UBERON:0006631',
 'UBERON:0009834',
 'UBERON:0018115',
 'UBERON:0018116',
 'UBERON:0018117',
 'UBERON:0018118',
 'UBERON:0036149',
 'UBERON:8300001',
 'UBERON:8300002',
 'UBERON:8300003',
 'UBERON:8300004',
 'CL:0000062',
 'CL:0000137',
 'CL:0000169',
 'CL:0000192',
 'CL:0000221',
 'CL:0000222',
 'CL:0000307',
 'CL:0000346',
 'CL:0000594',
 'CL:0000650',
 'CL:0000842',
 'CL:0002098',
 'CL:0002188',
 'CL:0002319',
 'CL:0002324',
 'CL:0002326',
 'CL:0002366',
 'CL:0002368',
 'CL:0002451',
 'CL:0002539',
 'CL:0002540',
 'CL:0002543',
 'CL:0002545',
 'CL:0002568',
 'CL:0002570',
 'CL:0002577',
 'CL:0002583',
 'CL:0002591',
 'CL:0002592',
 'CL:0002594',
 'CL:0002597',
 'CL:0002598',
 'CL:0002600',
 'CL:0002601',
 'CL:0010002',
 'CL:0010003',
 'CL:1000458',
 'CL:2000001',
 'CL:2000007',
 'CL:2000011',
 'CL:2000018',
 'CL:2000040',
 'CL:2000066',
 'CL:2000067',
 'CL:2000068',
 'CL:2000071',
 'CL:2000078',
 'CL:2000092',
 'CL:2000093',
 'CL:2000094',
 'EFO:0002103',
 'EFO:0007089',
 'EFO:0030018',
 'NTR:0000427',
 'NTR:0003013',
 'UBERON:0000016',
 'UBERON:0000019',
 'UBERON:0000081',
 'UBERON:0000088',
 'UBERON:0000305',
 'UBERON:0000996',
 'UBERON:0001013',
 'UBERON:0001043',
 'UBERON:0001134',
 'UBERON:0001255',
 'UBERON:0001871',
 'UBERON:0001872',
 'UBERON:0001894',
 'UBERON:0002021',
 'UBERON:0002097',
 'UBERON:0004648',
 'UBERON:0006920',
 'UBERON:0007106',
 'UBERON:1000010',
 'CL:0000121',
 'CL:0001053',
 'EFO:0000572',
 'EFO:0002009',
 'EFO:0002786',
 'EFO:0005234',
 'EFO:0005237',
 'UBERON:0000007',
 'UBERON:0000458',
 'UBERON:0001225',
 'UBERON:0001293',
 'UBERON:0001873',
 'UBERON:0001876',
 'UBERON:0001882',
 'UBERON:0001898',
 'UBERON:0002038',
 'UBERON:0002245',
 'UBERON:0006330',
 'UBERON:0006469',
 'UBERON:0006566',
 'UBERON:0009835',
 'UBERON:0012249',
 'UBERON:0013756',
 'CL:0000540',
 'CL:0000775',
 'EFO:0001162',
 'EFO:0001182',
 'EFO:0001260',
 'EFO:0002095',
 'EFO:0002167',
 'EFO:0002357',
 'EFO:0005333',
 'EFO:0005694',
 'EFO:0005907',
 'EFO:0006283',
 'EFO:0006365',
 'EFO:0006710',
 'EFO:0007070',
 'EFO:0007076',
 'EFO:0007083',
 'EFO:0007086',
 'EFO:0007096',
 'EFO:0007099',
 'EFO:0007102',
 'EFO:0007105',
 'EFO:0007107',
 'EFO:0007108',
 'EFO:0007112',
 'EFO:0007116',
 'EFO:0007752',
 'NTR:0001226',
 'NTR:0001407',
 'NTR:0003079',
 'UBERON:0000320',
 'UBERON:0001003',
 'UBERON:0001199',
 'UBERON:0002305',
 'UBERON:0002686',
 'UBERON:0002967',
 'UBERON:0003346',
 'UBERON:0004222',
 'UBERON:0012488',
 'UBERON:0012489',
 'UBERON:0014455',
 'UBERON:0018112']


def shlex_join_compat(parts: List[str]) -> str:
    try:
        return shlex.join(parts)
    except Exception:
        return " ".join(shlex.quote(p) for p in parts)


def parse_requested_output_type(value: str) -> str:
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    normalized = OUTPUT_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in OUTPUT_TYPE_SPECS:
        valid = ", ".join(DEFAULT_REQUESTED_OUTPUT_TYPES)
        raise argparse.ArgumentTypeError(
            f"Invalid requested output type '{value}'. Expected one of: {valid}."
        )
    return normalized


def normalize_requested_output_types(requested_output_types: Optional[List[str]]) -> List[str]:
    values = list(DEFAULT_REQUESTED_OUTPUT_TYPES) if requested_output_types is None else list(requested_output_types)
    normalized: List[str] = []
    seen = set()
    for value in values:
        output_type = parse_requested_output_type(value)
        if output_type in seen:
            continue
        normalized.append(output_type)
        seen.add(output_type)
    if not normalized:
        raise ValueError("At least one requested output type must be provided.")
    return normalized


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run compute_all_tracks_exon_medians_for_group over orthology parquet files "
            "for many genes in parallel via Slurm arrays."
        )
    )

    ap.add_argument("--genes-csv", type=Path, default=DEFAULT_GENES_CSV)
    ap.add_argument(
        "--genes-col",
        default=DEFAULT_GENES_COL,
        help="Column containing gene IDs. If missing, falls back to ncbi_id/ncbi_gene_id when available.",
    )
    ap.add_argument("--orthology-dir", type=Path, default=DEFAULT_ORTHOLOGY_DIR)
    ap.add_argument("--preds-dir", type=Path, default=DEFAULT_PREDS_DIR)
    ap.add_argument("--input-template", default=DEFAULT_INPUT_TEMPLATE)
    ap.add_argument("--output-template", default=DEFAULT_OUTPUT_TEMPLATE)

    ap.add_argument(
        "--assay-title",
        dest="assay_titles",
        action="append",
        default=None,
        help="Filter tracks by assay title. Can be provided multiple times; defaults to common assay titles.",
    )
    ap.add_argument(
        "--requested-output-type",
        dest="requested_output_types",
        action="append",
        type=parse_requested_output_type,
        default=None,
        help=(
            "Restrict AlphaGenome requested_outputs to specific output types. "
            f"Can be provided multiple times; defaults to {', '.join(DEFAULT_REQUESTED_OUTPUT_TYPES)}."
        ),
    )
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir for snapshot_download.",
    )
    ap.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Load model only from local Hugging Face cache (no network).",
    )
    ap.add_argument(
        "--max-n-ratio",
        type=float,
        default=0.1,
        help="Optional prefilter on orthology rows using n_ratio_total <= value. Use negative value to disable.",
    )

    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--genes-per-task", type=int, default=2)
    ap.add_argument("--num-tasks", type=int, default=None)
    ap.add_argument("--task-id", type=int, default=None, help="Override task id (else uses $SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--show-row-progress", action="store_true", help="Enable per-ortholog-row tqdm in each gene task.")
    ap.add_argument("--summary-dir", type=Path, default=None)

    ap.add_argument("--print-sbatch", action="store_true")
    ap.add_argument("--sbatch-submit", action="store_true", help="Submit the generated sbatch command.")
    ap.add_argument("--sbatch-job-name", default="orth_prim_all_tracks")
    ap.add_argument("--sbatch-max-concurrent", type=int, default=20)
    ap.add_argument("--sbatch-cpus-per-task", type=int, default=8)
    ap.add_argument("--sbatch-mem", default="64G")
    ap.add_argument("--sbatch-time", default="24:00:00")
    ap.add_argument(
        "--sbatch-gres",
        default="gpu:1",
        help="GPU resource request passed to sbatch as --gres=<value>. Default: gpu:1",
    )
    ap.add_argument("--sbatch-nodelist", default="")
    ap.add_argument("--sbatch-log-dir", type=Path, default=None)

    return ap.parse_args()


def _resolve_gene_column(requested: str, columns: List[str]) -> str:
    if requested in columns:
        return requested

    fallbacks: List[str] = []
    if requested != "ncbi_id":
        fallbacks.append("ncbi_id")
    if requested != "ncbi_gene_id":
        fallbacks.append("ncbi_gene_id")

    for col in fallbacks:
        if col in columns:
            return col

    raise RuntimeError(
        f"Column '{requested}' not found in genes CSV. Available columns: {columns}"
    )


def load_genes(
    genes_csv: Path,
    genes_col: str,
    start_index: int = 0,
    max_genes: Optional[int] = None,
) -> Tuple[List[str], str]:
    import csv

    if not genes_csv.exists():
        raise FileNotFoundError(f"genes csv not found: {genes_csv}")

    vals: List[str] = []
    with open(genes_csv, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        resolved_col = _resolve_gene_column(genes_col, columns)
        for row in reader:
            raw = row.get(resolved_col)
            if raw is None:
                continue
            v = str(raw).strip()
            if v.endswith(".0"):
                v = v[:-2]
            if not v or v.lower() == "nan":
                continue
            vals.append(v)

    vals = list(dict.fromkeys(vals))

    if start_index < 0:
        start_index = 0
    vals = vals[start_index:]
    if max_genes is not None:
        vals = vals[: max(0, int(max_genes))]

    return vals, resolved_col


def resolve_task_id(task_id_arg: Optional[int]) -> int:
    if task_id_arg is not None:
        return int(task_id_arg)
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def compute_task_slice(
    total: int,
    task_id: int,
    genes_per_task: int,
    num_tasks: Optional[int],
) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 1

    if num_tasks is not None:
        if num_tasks <= 0:
            raise ValueError("--num-tasks must be > 0")
        base = total // num_tasks
        rem = total % num_tasks
        start = task_id * base + min(task_id, rem)
        size = base + (1 if task_id < rem else 0)
        end = min(total, start + size)
        return start, end, num_tasks

    if genes_per_task <= 0:
        raise ValueError("--genes-per-task must be > 0")

    total_tasks = max(1, math.ceil(total / genes_per_task))
    start = task_id * genes_per_task
    end = min(total, start + genes_per_task)
    return start, end, total_tasks


def _safe_format_template(template: str, gene_id: str) -> str:
    return template.format(gene_id=gene_id, ncbi_id=gene_id)


def build_worker_command(args: argparse.Namespace, genes_col: str) -> List[str]:
    script_abs = Path(__file__).resolve()
    cmd: List[str] = [
        sys.executable,
        str(script_abs),
        "--genes-csv",
        str(args.genes_csv),
        "--genes-col",
        str(genes_col),
        "--orthology-dir",
        str(args.orthology_dir),
        "--preds-dir",
        str(args.preds_dir),
        "--input-template",
        str(args.input_template),
        "--output-template",
        str(args.output_template),
        "--model-name",
        str(args.model_name),
        *(["--hf-cache-dir", str(args.hf_cache_dir)] if args.hf_cache_dir is not None else []),
        *(["--hf-local-files-only"] if args.hf_local_files_only else []),
        "--start-index",
        str(args.start_index),
    ]

    if args.assay_titles:
        for assay_title in args.assay_titles:
            cmd.extend(["--assay-title", str(assay_title)])

    if args.requested_output_types:
        for requested_output_type in normalize_requested_output_types(args.requested_output_types):
            cmd.extend(["--requested-output-type", str(requested_output_type)])

    if args.max_genes is not None:
        cmd.extend(["--max-genes", str(args.max_genes)])

    if args.max_n_ratio is not None:
        cmd.extend(["--max-n-ratio", str(args.max_n_ratio)])

    if args.num_tasks is not None:
        cmd.extend(["--num-tasks", str(args.num_tasks)])
    else:
        cmd.extend(["--genes-per-task", str(args.genes_per_task)])

    if args.summary_dir is not None:
        cmd.extend(["--summary-dir", str(args.summary_dir)])

    if args.skip_existing:
        cmd.append("--skip-existing")
    else:
        cmd.append("--no-skip-existing")

    if args.show_row_progress:
        cmd.append("--show-row-progress")

    return cmd


def print_or_submit_sbatch(args: argparse.Namespace) -> None:
    genes, resolved_col = load_genes(args.genes_csv, args.genes_col, args.start_index, args.max_genes)
    total = len(genes)
    if total == 0:
        print("No genes to process after filtering.", flush=True)
        return

    if args.num_tasks is not None:
        total_tasks = int(args.num_tasks)
    else:
        total_tasks = max(1, math.ceil(total / int(args.genes_per_task)))

    log_dir = args.sbatch_log_dir or (args.preds_dir / "slurm_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_cmd = build_worker_command(args, genes_col=resolved_col)
    max_task_id = max(0, total_tasks - 1)
    array_spec = f"0-{max_task_id}%{max(1, int(args.sbatch_max_concurrent))}"

    sbatch_cmd = [
        "sbatch",
        f"--job-name={args.sbatch_job_name}",
        f"--array={array_spec}",
        f"--cpus-per-task={args.sbatch_cpus_per_task}",
        f"--mem={args.sbatch_mem}",
        f"--time={args.sbatch_time}",
        f"--output={log_dir}/%x_%A_%a.out",
        f"--error={log_dir}/%x_%A_%a.err",
    ]

    if str(args.sbatch_gres).strip():
        sbatch_cmd.append(f"--gres={str(args.sbatch_gres).strip()}")

    if str(args.sbatch_nodelist).strip():
        sbatch_cmd.append(f"--nodelist={str(args.sbatch_nodelist).strip()}")

    sbatch_cmd.extend(["--wrap", shlex_join_compat(worker_cmd)])

    print(shlex_join_compat(sbatch_cmd), flush=True)
    if args.sbatch_submit:
        cp = subprocess.run(sbatch_cmd, text=True, capture_output=True)
        if cp.stdout:
            print(cp.stdout.strip(), flush=True)
        if cp.returncode != 0:
            if cp.stderr:
                print(cp.stderr.strip(), flush=True)
            raise SystemExit(cp.returncode)


def _alphagenome_hf_repo_id(model_name: str) -> str:
    return f"google/alphagenome-{str(model_name).replace('_', '-').lower()}"


def _load_model_from_hf_snapshot(
    *,
    dna_model_module: Any,
    model_name: str,
    hf_cache_dir: Optional[Path],
    hf_local_files_only: bool,
) -> Any:
    import huggingface_hub

    snapshot_kwargs: Dict[str, Any] = {
        "repo_id": _alphagenome_hf_repo_id(model_name),
    }
    if hf_cache_dir is not None:
        snapshot_kwargs["cache_dir"] = str(hf_cache_dir)
    if hf_local_files_only:
        snapshot_kwargs["local_files_only"] = True

    checkpoint_path = huggingface_hub.snapshot_download(**snapshot_kwargs)
    return dna_model_module.create(checkpoint_path)


def _parse_half_open_intervals(interval_value: Any, pd_module: Any) -> List[Tuple[int, int]]:
    if interval_value is None:
        return []

    try:
        if pd_module.isna(interval_value):
            return []
    except Exception:
        pass

    intervals = interval_value
    if isinstance(intervals, str):
        try:
            intervals = ast.literal_eval(intervals)
        except Exception:
            return []

    parsed: List[Tuple[int, int]] = []
    try:
        for interval in intervals:
            if interval is None:
                continue
            start, end = interval
            try:
                if pd_module.isna(start) or pd_module.isna(end):
                    continue
            except Exception:
                pass
            start = int(start)
            end = int(end)
            if (end > start) and (start >= 0):
                parsed.append((start, end))
    except Exception:
        return []

    return parsed


def _compute_exon_aggregations(track_values: Any, exon_intervals: List[Tuple[int, int]], np_module: Any) -> Tuple[Dict[str, Any], int]:
    if track_values.ndim != 2 or track_values.shape[1] == 0:
        return {}, 0

    n_tracks = track_values.shape[1]
    nan_vec = np_module.full(n_tracks, np_module.nan)

    if not exon_intervals:
        return {
            "median_of_exon_medians": nan_vec.copy(),
            "median_of_exon_means": nan_vec.copy(),
            "mean_of_exon_means": nan_vec.copy(),
            "sum_over_all_exons": nan_vec.copy(),
            "mean_over_all_exons": nan_vec.copy(),
        }, 0

    n_bins = track_values.shape[0]
    exon_chunks = []
    exon_medians = []
    exon_means = []

    for exon_start, exon_end in exon_intervals:
        exon_start = int(exon_start)
        exon_end = int(exon_end)
        if (
            (exon_end <= exon_start)
            or (exon_start >= n_bins)
            or (exon_end <= 0)
            or (exon_end > n_bins)
            or (exon_start < 0)
        ):
            continue

        chunk = track_values[exon_start:exon_end, :]
        if chunk.size == 0:
            continue

        exon_chunks.append(chunk)
        exon_medians.append(np_module.median(chunk, axis=0))
        exon_means.append(np_module.mean(chunk, axis=0))

    if not exon_chunks:
        return {
            "median_of_exon_medians": nan_vec.copy(),
            "median_of_exon_means": nan_vec.copy(),
            "mean_of_exon_means": nan_vec.copy(),
            "sum_over_all_exons": nan_vec.copy(),
            "mean_over_all_exons": nan_vec.copy(),
        }, 0

    exon_medians = np_module.stack(exon_medians, axis=0)
    exon_means = np_module.stack(exon_means, axis=0)
    all_exon_values = np_module.concatenate(exon_chunks, axis=0)

    aggregations = {
        "median_of_exon_medians": np_module.median(exon_medians, axis=0),
        "median_of_exon_means": np_module.median(exon_means, axis=0),
        "mean_of_exon_means": np_module.mean(exon_means, axis=0),
        "sum_over_all_exons": np_module.sum(all_exon_values, axis=0),
        "mean_over_all_exons": np_module.mean(all_exon_values, axis=0),
    }
    return aggregations, int(all_exon_values.shape[0])


def _compute_chip_histone_nearest_bin_aggregations(
    track_values: Any,
    exon_intervals: List[Tuple[int, int]],
    sequence_length: int,
    np_module: Any,
) -> Tuple[Dict[str, Any], int]:
    if track_values.ndim != 2 or track_values.shape[1] == 0:
        return {}, 0

    n_tracks = track_values.shape[1]
    nan_vec = np_module.full(n_tracks, np_module.nan)

    if not exon_intervals:
        return {
            "median_of_exon_medians": nan_vec.copy(),
            "median_of_exon_means": nan_vec.copy(),
            "mean_of_exon_means": nan_vec.copy(),
            "sum_over_all_exons": nan_vec.copy(),
            "mean_over_all_exons": nan_vec.copy(),
        }, 0

    n_bins = track_values.shape[0]
    if n_bins <= 0:
        return {
            "median_of_exon_medians": nan_vec.copy(),
            "median_of_exon_means": nan_vec.copy(),
            "mean_of_exon_means": nan_vec.copy(),
            "sum_over_all_exons": nan_vec.copy(),
            "mean_over_all_exons": nan_vec.copy(),
        }, 0

    try:
        seq_len = int(sequence_length)
    except Exception:
        seq_len = n_bins
    if seq_len <= 0:
        seq_len = n_bins

    selected_bin_indices: List[int] = []
    for exon_start, exon_end in exon_intervals:
        exon_start = int(exon_start)
        exon_end = int(exon_end)
        if exon_end <= exon_start:
            continue

        exon_center = (exon_start + exon_end - 1) / 2.0
        if seq_len == 1 or n_bins == 1:
            bin_idx = 0
        else:
            bin_idx = int(round((exon_center / float(seq_len - 1)) * float(n_bins - 1)))
        bin_idx = int(np_module.clip(bin_idx, 0, n_bins - 1))
        selected_bin_indices.append(bin_idx)

    if not selected_bin_indices:
        return {
            "median_of_exon_medians": nan_vec.copy(),
            "median_of_exon_means": nan_vec.copy(),
            "mean_of_exon_means": nan_vec.copy(),
            "sum_over_all_exons": nan_vec.copy(),
            "mean_over_all_exons": nan_vec.copy(),
        }, 0

    selected_values = track_values[selected_bin_indices, :]
    aggregations = {
        "median_of_exon_medians": np_module.median(selected_values, axis=0),
        "median_of_exon_means": np_module.median(selected_values, axis=0),
        "mean_of_exon_means": np_module.mean(selected_values, axis=0),
        "sum_over_all_exons": np_module.sum(selected_values, axis=0),
        "mean_over_all_exons": np_module.mean(selected_values, axis=0),
    }
    return aggregations, int(len(selected_bin_indices))


def _prepare_assay_selection(assay_data: Any, assay_type: str, assay_titles: List[str], np_module: Any) -> Optional[Dict[str, Any]]:
    meta = assay_data.metadata
    if meta is None or meta.empty:
        return None

    mask = np_module.ones(len(meta), dtype=bool)
    if "Assay title" in meta.columns:
        mask &= meta["Assay title"].isin(assay_titles).to_numpy()
    if "strand" in meta.columns:
        mask &= (meta["strand"] != "-").to_numpy()

    keep_idx = np_module.flatnonzero(mask)
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

    return {
        "keep_idx": keep_idx.astype(int),
        "track_names": track_names,
        "track_assay_titles": track_assay_titles,
    }


def compute_all_tracks_exon_medians_for_group(
    orthologs_df: Any,
    *,
    model: Any,
    dna_model_module: Any,
    genome_module: Any,
    ontology_terms: Optional[List[str]] = None,
    assay_titles: Optional[List[str]] = None,
    requested_output_types: Optional[List[str]] = None,
    species_col: str = "species",
    sequence_col: str = "sequence",
    exon_intervals_col: str = "mapped_exon_intervals_in_window_0based_half_open",
    show_row_progress: bool = False,
    progress_desc: Optional[str] = None,
    overlap_predict_and_process: bool = True,
) -> Any:
    import numpy as np
    import pandas as pd

    if ontology_terms is None:
        ontology_terms = ALL_SELECTED_ONTOLOGY_TERMS
    if assay_titles is None:
        assay_titles = list(DEFAULT_ASSAY_TITLES)
    requested_output_types = normalize_requested_output_types(requested_output_types)

    rows = orthologs_df.to_dict("records")

    if not rows:
        return pd.DataFrame()

    summary_frames: List[Any] = []
    assay_defs = [OUTPUT_TYPE_SPECS[output_type] for output_type in requested_output_types]
    requested_outputs = {
        getattr(dna_model_module.OutputType, output_type)
        for output_type in requested_output_types
    }
    assay_selection_cache: Optional[Dict[str, Any]] = None

    if show_row_progress:
        import sys
        from tqdm import tqdm

        pbar = tqdm(
            total=len(rows),
            desc=progress_desc or "Rows",
            unit="row",
            file=sys.stderr,
            leave=True,
            dynamic_ncols=True,
        )
    else:
        pbar = None

    def _clean_and_log(values: Any) -> Tuple[Any, Any]:
        arr = np.asarray(values, dtype=np.float64)
        clean = np.where(np.isfinite(arr), arr, np.nan)
        log_vals = np.full(clean.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(clean) & (clean > 0)
        log_vals[valid] = np.log10(clean[valid] + 1e-6)
        return clean, log_vals

    def _predict_single_row(row: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str, Any]]:
        sequence_raw = row.get(sequence_col)
        if not isinstance(sequence_raw, str):
            return None

        sequence = "".join(sequence_raw.split()).upper()
        if not sequence:
            return None

        species_name = row.get(species_col)
        try:
            if pd.isna(species_name):
                return None
        except Exception:
            pass

        interval = genome_module.Interval(
            str(species_name),
            0,
            len(sequence),
            strand="+",
        )

        preds = model.predict_sequence(
            sequence,
            organism=dna_model_module.Organism.HOMO_SAPIENS,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
            interval=interval,
        )
        return row, sequence, preds

    def _process_predicted_row(row: Dict[str, Any], sequence: str, preds: Any) -> None:
        nonlocal assay_selection_cache

        if assay_selection_cache is None:
            assay_selection_cache = {}
            for assay_type, assay_attr in assay_defs:
                selection = _prepare_assay_selection(
                    getattr(preds, assay_attr),
                    assay_type,
                    assay_titles,
                    np_module=np,
                )
                assay_selection_cache[assay_type] = selection

        exon_intervals = _parse_half_open_intervals(row.get(exon_intervals_col), pd_module=pd)
        seq_len = len(sequence)

        n_count = sequence.count("N")
        n_ratio = n_count / seq_len if seq_len else np.nan

        base_row = {
            "ortholog_geneid": row.get("ortholog_geneid"),
            "taxId": row.get("taxId"),
            "taxname": row.get("taxname"),
            "commonName": row.get("commonName"),
            "species": row.get("species"),
            "assembly": row.get("assembly"),
            "contig": row.get("mapped_contig") if row.get("mapped_contig") is not None else row.get("contig"),
            "strand": row.get("mapped_strand") if row.get("mapped_strand") is not None else row.get("strand"),
            "coding_exon_count": row.get("coding_exon_count"),
            "sequence_length_bp": seq_len,
            "sequence_N_amount": n_count,
            "sequence_N_ratio": n_ratio,
            "all_mapped_exons_fit_in_window": row.get("all_mapped_exons_fit_in_window"),
            "used_tolerance": row.get("used_tolerance"),
        }

        # Preserve per-variant metadata when present so downstream ref/alt merges are stable.
        for passthrough_col in (
            "variant_row_idx",
            "variant_id",
            "Chromosome",
            "pos",
            "ref",
            "alt",
            "region",
        ):
            if passthrough_col in row:
                base_row[passthrough_col] = row.get(passthrough_col)

        global_track_idx = 0

        for assay_type, assay_attr in assay_defs:
            selection = assay_selection_cache.get(assay_type) if assay_selection_cache else None
            if not selection:
                continue

            assay_values = np.asarray(getattr(preds, assay_attr).values)
            if assay_values.ndim == 1:
                assay_values = assay_values[:, np.newaxis]
            if assay_values.ndim != 2 or assay_values.shape[1] == 0:
                continue

            keep_idx = selection["keep_idx"]
            if keep_idx.size == 0:
                continue

            if keep_idx[-1] >= assay_values.shape[1]:
                selection = _prepare_assay_selection(
                    getattr(preds, assay_attr),
                    assay_type,
                    assay_titles,
                    np_module=np,
                )
                if assay_selection_cache is not None:
                    assay_selection_cache[assay_type] = selection
                if not selection:
                    continue
                keep_idx = selection["keep_idx"]

            track_values = assay_values[:, keep_idx]
            n_tracks = int(track_values.shape[1])
            if n_tracks == 0:
                continue


            if assay_type == "chip_histone":
                aggregations, selected_exon_bp = _compute_chip_histone_nearest_bin_aggregations(
                    track_values,
                    exon_intervals,
                    seq_len,
                    np_module=np,
                )
            else:
                aggregations, selected_exon_bp = _compute_exon_aggregations(
                    track_values,
                    exon_intervals,
                    np_module=np,
                )

            if not aggregations:
                continue

            track_names = selection["track_names"]
            track_assay_titles = selection["track_assay_titles"]
            if len(track_names) != n_tracks:
                track_names = [f"{assay_type}_track_{i}" for i in range(n_tracks)]
            if len(track_assay_titles) != n_tracks:
                track_assay_titles = [""] * n_tracks

            median_of_exon_medians, log10_median_of_exon_medians = _clean_and_log(
                aggregations["median_of_exon_medians"]
            )
            median_of_exon_means, log10_median_of_exon_means = _clean_and_log(
                aggregations["median_of_exon_means"]
            )
            mean_of_exon_means, log10_mean_of_exon_means = _clean_and_log(
                aggregations["mean_of_exon_means"]
            )
            sum_over_all_exons, log10_sum_over_all_exons = _clean_and_log(
                aggregations["sum_over_all_exons"]
            )
            mean_over_all_exons, log10_mean_over_all_exons = _clean_and_log(
                aggregations["mean_over_all_exons"]
            )
            track_idx_arr = global_track_idx + np.arange(n_tracks, dtype=np.int64)
            assay_track_idx_arr = np.arange(n_tracks, dtype=np.int64)

            assay_df = pd.DataFrame(
                {
                    "selected_exon_bp": np.full(n_tracks, selected_exon_bp),
                    "track_idx": track_idx_arr,
                    "track_name": track_names,
                    "assay_type": np.full(n_tracks, assay_type),
                    "assay_title": track_assay_titles,
                    "assay_track_idx": assay_track_idx_arr,
                    "rna_seq_track_idx": track_idx_arr,
                    "rna_seq_track_name": track_names,
                    "median_of_exon_medians": median_of_exon_medians,
                    "log10_median_of_exon_medians": log10_median_of_exon_medians,
                    "median_of_exon_means": median_of_exon_means,
                    "log10_median_of_exon_means": log10_median_of_exon_means,
                    "mean_of_exon_means": mean_of_exon_means,
                    "log10_mean_of_exon_means": log10_mean_of_exon_means,
                    "sum_over_all_exons": sum_over_all_exons,
                    "log10_sum_over_all_exons": log10_sum_over_all_exons,
                    "mean_over_all_exons": mean_over_all_exons,
                    "log10_mean_over_all_exons": log10_mean_over_all_exons,
                }
            )

            for key, value in base_row.items():
                assay_df[key] = value

            summary_frames.append(assay_df)
            global_track_idx += n_tracks

    try:
        if overlap_predict_and_process and len(rows) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as predictor_pool:
                prev_future = predictor_pool.submit(_predict_single_row, rows[0])
                for row in rows[1:]:

                    next_future = predictor_pool.submit(_predict_single_row, row)
                    pred_out = prev_future.result()
                    if pred_out is not None:
                        _process_predicted_row(*pred_out)
                    if pbar is not None:
                        pbar.update(1)
                    prev_future = next_future

                pred_out = prev_future.result()
                if pred_out is not None:
                    _process_predicted_row(*pred_out)
                if pbar is not None:
                    pbar.update(1)
        else:
            for row in rows:
                pred_out = _predict_single_row(row)
                if pred_out is not None:
                    _process_predicted_row(*pred_out)
                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if not summary_frames:
        return pd.DataFrame()
    return pd.concat(summary_frames, ignore_index=True)




def main() -> None:
    args = parse_args()

    if args.max_n_ratio is not None and args.max_n_ratio < 0:
        args.max_n_ratio = None

    if args.print_sbatch or args.sbatch_submit:
        print_or_submit_sbatch(args)
        return

    import pandas as pd

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.summary_dir or (args.preds_dir / "summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)

    genes, resolved_col = load_genes(args.genes_csv, args.genes_col, args.start_index, args.max_genes)
    total = len(genes)

    task_id = resolve_task_id(args.task_id)
    start, end, total_tasks = compute_task_slice(
        total=total,
        task_id=task_id,
        genes_per_task=args.genes_per_task,
        num_tasks=args.num_tasks,
    )

    task_genes = genes[start:end] if start < end else []
    print(
        f"[task] task_id={task_id} total_tasks={total_tasks} total_genes={total} "
        f"slice=[{start}:{end}) n_task_genes={len(task_genes)}",
        flush=True,
    )

    if not task_genes:
        print("[task] No genes assigned to this task.", flush=True)
        return

    repo_id = _alphagenome_hf_repo_id(args.model_name)
    print(
        f"[task] creating model: {args.model_name} (repo_id={repo_id}) "
        f"hf_cache_dir={args.hf_cache_dir or '<default>'} "
        f"local_files_only={bool(args.hf_local_files_only)}",
        flush=True,
    )
    try:
        from alphagenome_research.model import dna_model
        from alphagenome.data import genome
    except Exception as exc:
        raise RuntimeError(
            "Failed to import alphagenome packages. Run in an environment with alphagenome installed."
        ) from exc

    model = _load_model_from_hf_snapshot(
        dna_model_module=dna_model,
        model_name=args.model_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_local_files_only=bool(args.hf_local_files_only),
    )

    processed = 0
    skipped_existing = 0
    errors: List[Dict[str, Any]] = []
    gene_status_rows: List[Dict[str, Any]] = []

    for human_gene in task_genes:
        human_gene_str = str(human_gene)

        try:
            input_name = _safe_format_template(args.input_template, human_gene_str)
            output_name = _safe_format_template(args.output_template, human_gene_str)
        except Exception as exc:
            err_msg = f"template_format_error: {exc}"
            errors.append({"human_gene": human_gene_str, "error": err_msg})
            gene_status_rows.append(
                {
                    "task_id": int(task_id),
                    "human_gene": human_gene_str,
                    "status": "error",
                    "input_path": "",
                    "output_path": "",
                    "source_rows": pd.NA,
                    "rows_after_n_filter": pd.NA,
                    "rows_written": pd.NA,
                    "error": err_msg,
                }
            )
            continue

        input_path = args.orthology_dir / input_name
        output_path = args.preds_dir / output_name

        if args.skip_existing and output_path.exists():
            skipped_existing += 1
            gene_status_rows.append(
                {
                    "task_id": int(task_id),
                    "human_gene": human_gene_str,
                    "status": "skipped_existing",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_rows": pd.NA,
                    "rows_after_n_filter": pd.NA,
                    "rows_written": pd.NA,
                    "error": "",
                }
            )
            continue

        if not input_path.exists():
            err_msg = f"missing_input: {input_path}"
            errors.append({"human_gene": human_gene_str, "error": err_msg})
            gene_status_rows.append(
                {
                    "task_id": int(task_id),
                    "human_gene": human_gene_str,
                    "status": "error",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_rows": pd.NA,
                    "rows_after_n_filter": pd.NA,
                    "rows_written": pd.NA,
                    "error": err_msg,
                }
            )
            continue

        try:
            orthologs_df = pd.read_parquet(input_path)
            source_rows = int(len(orthologs_df))

            if args.max_n_ratio is not None and "n_ratio_total" in orthologs_df.columns:
                orthologs_df = orthologs_df[orthologs_df["n_ratio_total"] <= float(args.max_n_ratio)].copy()

            rows_after_n_filter = int(len(orthologs_df))

            preds_df = compute_all_tracks_exon_medians_for_group(
                orthologs_df=orthologs_df,
                model=model,
                dna_model_module=dna_model,
                genome_module=genome,
                assay_titles=args.assay_titles,
                requested_output_types=args.requested_output_types,
                show_row_progress=bool(args.show_row_progress),
            )

            if "human_ncbi_id" not in preds_df.columns:
                preds_df.insert(0, "human_ncbi_id", human_gene_str)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            preds_df.to_parquet(output_path, index=False)
            processed += 1

            gene_status_rows.append(
                {
                    "task_id": int(task_id),
                    "human_gene": human_gene_str,
                    "status": "success",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_rows": source_rows,
                    "rows_after_n_filter": rows_after_n_filter,
                    "rows_written": int(len(preds_df)),
                    "error": "",
                }
            )
        except Exception as exc:
            err_msg = str(exc)
            errors.append({"human_gene": human_gene_str, "error": err_msg})
            gene_status_rows.append(
                {
                    "task_id": int(task_id),
                    "human_gene": human_gene_str,
                    "status": "error",
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "source_rows": pd.NA,
                    "rows_after_n_filter": pd.NA,
                    "rows_written": pd.NA,
                    "error": err_msg,
                }
            )

    summary = {
        "task_id": int(task_id),
        "total_tasks": int(total_tasks),
        "total_genes_after_filter": int(total),
        "genes_col": str(resolved_col),
        "slice_start": int(start),
        "slice_end": int(end),
        "assigned_genes": int(len(task_genes)),
        "processed": int(processed),
        "skipped_existing": int(skipped_existing),
        "errors": int(len(errors)),
        "genes_csv": str(args.genes_csv),
        "orthology_dir": str(args.orthology_dir),
        "preds_dir": str(args.preds_dir),
        "assay_titles": args.assay_titles,
        "requested_output_types": normalize_requested_output_types(args.requested_output_types),
        "model_name": str(args.model_name),
        "max_n_ratio": None if args.max_n_ratio is None else float(args.max_n_ratio),
    }

    summary_path = summary_dir / f"task_{task_id:05d}_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[task] wrote summary: {summary_path}", flush=True)

    status_df = pd.DataFrame(gene_status_rows)
    status_parquet = summary_dir / f"task_{task_id:05d}_gene_status.parquet"
    status_csv = summary_dir / f"task_{task_id:05d}_gene_status.csv"
    status_df.to_parquet(status_parquet, index=False)
    status_df.to_csv(status_csv, index=False)
    print(f"[task] wrote gene status table: {status_parquet}", flush=True)
    print(f"[task] wrote gene status csv: {status_csv}", flush=True)

    if errors:
        err_path = summary_dir / f"task_{task_id:05d}_errors.parquet"
        pd.DataFrame(errors).to_parquet(err_path, index=False)
        print(f"[task] wrote errors: {err_path}", flush=True)


if __name__ == "__main__":
    main()
