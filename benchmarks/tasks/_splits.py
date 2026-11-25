"""Shared identity, count and coordinate checks for prepared samples."""

from itertools import combinations

REGION_BED = "config/hg38_6k_1kb_region.bed"


def proportional_counts(counts, total):
    """Allocate integer quotas while keeping their sum equal to total."""
    population = sum(counts.values())
    shares = {key: divmod(total * count, population) for key, count in counts.items()}
    quotas = {key: whole for key, (whole, _) in shares.items()}
    remaining = total - sum(quotas.values())
    order = sorted(shares, key=lambda key: (-shares[key][1], key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def table_stats(frame):
    result = {"rows": len(frame), "regions": int(frame["build_region_index"].nunique())}
    for key in ("label", "kind"):
        if key in frame and (key == "kind" or frame[key].isin([0, 1]).all()):
            result[key + "_counts"] = {str(value): int(count) for value, count in frame[key].value_counts().items()}
    if "variant_id" in frame:
        result["variants"] = int(frame["variant_id"].nunique())
    return result


def audit_frames(frames, identity, *, allow_overlap=False, binary=False):
    overlaps = {}
    for left, right in combinations(frames, 2):
        overlaps[f"{left}:{right}"] = {
            key: len(set(frames[left][key]) & set(frames[right][key])) for key in identity
        }
    if not allow_overlap and any(n for pair in overlaps.values() for n in pair.values()):
        raise ValueError(f"Splits share held-out identifiers: {overlaps}")
    for name, frame in frames.items():
        if frame.empty or (binary and set(frame["label"]) != {0, 1}):
            raise ValueError(f"{name} is empty or lacks one class.")
    return {"splits": {name: table_stats(frame) for name, frame in frames.items()}, "overlap": overlaps}


def validate_regions(regions_path, tables, *, chain_path=None, lift=None):
    import numpy as np
    import pandas as pd

    columns = ["chrom", "start", "end", "build_region_index"]
    regions = pd.read_csv(regions_path, sep="\t", names=columns).set_index("build_region_index")
    lifted = None
    if chain_path is not None:
        ids = set().union(*(set(table["build_region_index"]) for table in tables))
        if not pd.Index(list(ids)).isin(regions.index).all():
            raise ValueError("A STARR region ID is absent from the pinned hg38 bin table.")
        bins = regions.loc[sorted(ids)]
        midpoints = {(chrom, (start + end) // 2) for chrom, start, end in bins.itertuples(index=False)}
        lifted = lift(chain_path, midpoints)
    for table in tables:
        if not table["build_region_index"].isin(regions.index).all():
            raise ValueError("A region ID is absent from the pinned hg38 bin table.")
        expected = regions.loc[table["build_region_index"], columns[:3]].to_numpy()
        observed = table[columns[:3]].to_numpy()
        if lifted is None:
            mismatch = (expected != observed).any(axis=1)
        else:
            mismatch = np.array([
                not any(c == target_chrom and left <= point < right for c, point in lifted[chrom, (start + end) // 2])
                for (chrom, start, end), (target_chrom, left, right) in zip(expected, observed)
            ])
        if mismatch.any():
            example = int(table.loc[mismatch, "build_region_index"].iloc[0])
            reference = "lifted hg19" if lifted is not None else "hg38"
            raise ValueError(f"{int(mismatch.sum())} rows disagree with the {reference} bin coordinates (example ID {example}).")


def check_regions(regions_path, *tables):
    validate_regions(regions_path, tables)
