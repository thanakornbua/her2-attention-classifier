"""Sampling utilities for mixing normalized and raw patch domains.

This module provides helpers to build controlled mixtures of normalized
and raw image patches for training, with options for:

    - Patch-level or case-level ratio enforcement.
    - Label-stratified sampling to reduce class imbalance drift.
    - Enforcing single-domain per case for MIL-style training.

The main public API is :func:`sample_dual_domain`, which returns both a
combined metadata DataFrame and a :class:`SamplingReport` describing the
result. Docstrings are written to follow PEP 257 conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingReport:
    requested_norm_ratio: float
    ratio_level: Literal["patch", "case"]
    stratify_labels: bool
    enforce_case_domain: bool
    seed: int
    n_norm_available: int
    n_raw_available: int
    n_selected: int
    n_selected_norm: int
    n_selected_raw: int
    n_selected_cases: int
    n_mixed_cases_in_input: int
    n_mixed_cases_in_selected: int
    label_counts_selected: Dict[int, int]


def _rng(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy random number generator.

    Args:
        seed: Integer seed used to initialize the RNG.

    Returns:
        A ``np.random.Generator`` instance seeded with ``seed``.
    """
    return np.random.default_rng(int(seed))


def _label_mode(series: pd.Series) -> int:
    """Return the most frequent label in a Series (deterministic).

    In case of ties between labels with the same maximum count, the
    smallest label value is returned to ensure deterministic behavior.

    Args:
        series: Pandas Series of integer-like labels.

    Returns:
        The mode label as an ``int``; returns ``0`` if the series is empty.
    """
    # deterministic mode for ties: pick smallest label
    vc = series.value_counts()
    if vc.empty:
        return 0
    max_count = vc.max()
    return int(sorted(vc[vc == max_count].index.tolist())[0])


def case_domain_mixing_stats(
    df: pd.DataFrame,
    *,
    case_col: str = "case_name",
    domain_col: str = "source",
) -> Tuple[int, int]:
    """Return (n_cases, n_mixed_cases) where mixed means >1 domain per case."""
    if df.empty:
        return 0, 0
    grp = df.groupby(case_col, dropna=False)[domain_col].nunique(dropna=False)
    n_cases = int(grp.shape[0])
    n_mixed = int((grp > 1).sum())
    return n_cases, n_mixed


def _stratified_sample_patches(
    df: pd.DataFrame,
    *,
    n: int,
    label_col: str,
    label_fracs: Dict[int, float],
    seed: int,
) -> pd.DataFrame:
    """Best-effort stratified sample to match label_fracs; fills remainder from leftovers."""
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()

    n = min(int(n), int(len(df)))
    rng = _rng(seed)

    selected_parts = []
    remaining_idx = set(df.index.tolist())

    # initial per-label allocation
    desired_by_label: Dict[int, int] = {
        int(lbl): int(np.floor(label_fracs.get(int(lbl), 0.0) * n)) for lbl in df[label_col].unique()
    }

    # distribute rounding remainder to labels with largest fractional part
    allocated = sum(desired_by_label.values())
    remainder = n - allocated
    if remainder > 0:
        # compute fractional parts using exact target
        frac_parts = []
        for lbl in desired_by_label:
            exact = label_fracs.get(int(lbl), 0.0) * n
            frac_parts.append((exact - np.floor(exact), int(lbl)))
        for _, lbl in sorted(frac_parts, reverse=True)[:remainder]:
            desired_by_label[lbl] += 1

    # sample each label bucket
    for lbl, k in desired_by_label.items():
        if k <= 0:
            continue
        bucket = df[df[label_col] == lbl]
        if bucket.empty:
            continue
        k = min(int(k), int(len(bucket)))
        take_idx = rng.choice(bucket.index.to_numpy(), size=k, replace=False)
        selected_parts.append(df.loc[take_idx])
        remaining_idx.difference_update(take_idx.tolist())

    selected = pd.concat(selected_parts, axis=0) if selected_parts else df.iloc[0:0].copy()

    # fill to n from remaining
    if len(selected) < n and remaining_idx:
        need = n - len(selected)
        rem = df.loc[list(remaining_idx)]
        take_idx = rng.choice(rem.index.to_numpy(), size=min(need, len(rem)), replace=False)
        selected = pd.concat([selected, df.loc[take_idx]], axis=0)

    # trim if we overshot
    if len(selected) > n:
        selected = selected.sample(n=n, random_state=seed)

    return selected.reset_index(drop=True)


def sample_dual_domain(
    norm_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    norm_ratio: float,
    seed: int = 42,
    ratio_level: Literal["patch", "case"] = "patch",
    stratify_labels: bool = True,
    enforce_case_domain: bool = False,
    label_col: str = "her2_status",
    case_col: str = "case_name",
    domain_col: str = "source",
    total_size: Optional[int] = None,
    prefer_domain: Literal["raw", "normalized"] = "raw",
) -> Tuple[pd.DataFrame, SamplingReport]:
    """Combine normalized + raw pools with a controlled domain ratio.

    - ratio_level="patch": enforces ratio on number of patches.
    - ratio_level="case": assigns whole cases to a domain (MIL-safer), then includes all patches from that domain.

    If stratify_labels=True, sampling is best-effort stratified by label to reduce class-balance drift.

    Returns: (combined_df, report)
    """
    if not (0.0 <= float(norm_ratio) <= 1.0):
        raise ValueError("norm_ratio must be between 0 and 1")

    norm_df = norm_df.copy()
    raw_df = raw_df.copy()

    n_cases_input, n_mixed_input = case_domain_mixing_stats(
        pd.concat([norm_df, raw_df], ignore_index=True),
        case_col=case_col,
        domain_col=domain_col,
    )

    if total_size is None:
        total_size = int(len(norm_df) + len(raw_df))

    if ratio_level == "patch":
        target_norm = int(round(float(norm_ratio) * total_size))
        target_raw = int(total_size - target_norm)

        target_norm = min(target_norm, len(norm_df))
        target_raw = min(target_raw, len(raw_df))

        # If one pool is short, top up from the other
        if target_norm + target_raw < total_size:
            missing = total_size - (target_norm + target_raw)
            # top up from whichever has room
            norm_room = len(norm_df) - target_norm
            raw_room = len(raw_df) - target_raw
            take_norm = min(missing, norm_room)
            target_norm += take_norm
            missing -= take_norm
            target_raw += min(missing, raw_room)

        if stratify_labels:
            combined = pd.concat([norm_df, raw_df], ignore_index=True)
            vc = combined[label_col].value_counts(normalize=True)
            label_fracs = {int(k): float(v) for k, v in vc.to_dict().items()}
            norm_sel = _stratified_sample_patches(norm_df, n=target_norm, label_col=label_col, label_fracs=label_fracs, seed=seed)
            raw_sel = _stratified_sample_patches(raw_df, n=target_raw, label_col=label_col, label_fracs=label_fracs, seed=seed + 1)
        else:
            norm_sel = norm_df.sample(n=target_norm, random_state=seed).reset_index(drop=True) if target_norm > 0 else norm_df.iloc[0:0]
            raw_sel = raw_df.sample(n=target_raw, random_state=seed + 1).reset_index(drop=True) if target_raw > 0 else raw_df.iloc[0:0]

        selected = pd.concat([norm_sel, raw_sel], ignore_index=True)

        if enforce_case_domain:
            selected = enforce_case_domain_consistency(
                selected,
                case_col=case_col,
                domain_col=domain_col,
                seed=seed,
                prefer_domain=prefer_domain,
            )[0]

    elif ratio_level == "case":
        # Build per-case label and availability by domain
        norm_cases = set(norm_df[case_col].dropna().unique().tolist())
        raw_cases = set(raw_df[case_col].dropna().unique().tolist())
        all_cases = sorted(norm_cases | raw_cases)

        if not all_cases:
            selected = pd.concat([norm_df.iloc[0:0], raw_df.iloc[0:0]], ignore_index=True)
        else:
            combined = pd.concat([norm_df, raw_df], ignore_index=True)
            case_to_label = combined.groupby(case_col)[label_col].apply(_label_mode).to_dict()

            rng = _rng(seed)
            prefer_is_raw = prefer_domain == "raw"

            # assign cases to domains (optionally stratified by label)
            assigned_norm: set = set()
            assigned_raw: set = set()

            if stratify_labels:
                labels = sorted(set(case_to_label.values()))
                for lbl in labels:
                    cases_lbl = [c for c in all_cases if case_to_label.get(c) == lbl]
                    if not cases_lbl:
                        continue

                    n_lbl = len(cases_lbl)
                    n_norm_lbl = int(round(float(norm_ratio) * n_lbl))

                    # cases that can be normalized
                    can_norm = [c for c in cases_lbl if c in norm_cases]
                    can_raw = [c for c in cases_lbl if c in raw_cases]

                    # Prefer cases that exist in both when needing flexibility
                    both = [c for c in cases_lbl if c in norm_cases and c in raw_cases]
                    norm_only = [c for c in cases_lbl if c in norm_cases and c not in raw_cases]
                    raw_only = [c for c in cases_lbl if c in raw_cases and c not in norm_cases]

                    # Select normalized-assigned cases
                    pick_norm = []
                    if n_norm_lbl > 0:
                        # must include all norm_only if needed
                        rng.shuffle(both)
                        rng.shuffle(norm_only)
                        pick_norm.extend(norm_only[: min(n_norm_lbl, len(norm_only))])
                        remaining = n_norm_lbl - len(pick_norm)
                        if remaining > 0:
                            pick_norm.extend(both[: min(remaining, len(both))])

                    assigned_norm.update(pick_norm)

                    # Remaining go to raw if possible
                    remaining_cases = [c for c in cases_lbl if c not in assigned_norm]
                    # Some may be raw-only or both; if case isn't available in raw, fall back to norm
                    for c in remaining_cases:
                        if c in raw_cases:
                            assigned_raw.add(c)
                        elif c in norm_cases:
                            assigned_norm.add(c)

                    # If any case still unassigned (shouldn't happen), assign based on availability
                    for c in cases_lbl:
                        if c in assigned_norm or c in assigned_raw:
                            continue
                        if c in raw_cases and (prefer_is_raw or c not in norm_cases):
                            assigned_raw.add(c)
                        elif c in norm_cases:
                            assigned_norm.add(c)
                        elif c in raw_cases:
                            assigned_raw.add(c)
            else:
                # simple random assignment
                rng.shuffle(all_cases)
                n_norm_cases = int(round(float(norm_ratio) * len(all_cases)))
                for c in all_cases[:n_norm_cases]:
                    if c in norm_cases:
                        assigned_norm.add(c)
                    elif c in raw_cases:
                        assigned_raw.add(c)
                for c in all_cases[n_norm_cases:]:
                    if c in raw_cases:
                        assigned_raw.add(c)
                    elif c in norm_cases:
                        assigned_norm.add(c)

            norm_sel = norm_df[norm_df[case_col].isin(assigned_norm)].copy()
            raw_sel = raw_df[raw_df[case_col].isin(assigned_raw)].copy()
            selected = pd.concat([norm_sel, raw_sel], ignore_index=True)

            if enforce_case_domain:
                selected, _ = enforce_case_domain_consistency(
                    selected,
                    case_col=case_col,
                    domain_col=domain_col,
                    seed=seed,
                    prefer_domain=prefer_domain,
                )

    else:
        raise ValueError("ratio_level must be 'patch' or 'case'")

    # Final report
    n_cases_selected, n_mixed_selected = case_domain_mixing_stats(selected, case_col=case_col, domain_col=domain_col)
    label_counts = selected[label_col].value_counts().to_dict() if not selected.empty else {}
    label_counts = {int(k): int(v) for k, v in label_counts.items()}

    rep = SamplingReport(
        requested_norm_ratio=float(norm_ratio),
        ratio_level=ratio_level,
        stratify_labels=bool(stratify_labels),
        enforce_case_domain=bool(enforce_case_domain),
        seed=int(seed),
        n_norm_available=int(len(norm_df)),
        n_raw_available=int(len(raw_df)),
        n_selected=int(len(selected)),
        n_selected_norm=int((selected[domain_col] == "normalized").sum()) if domain_col in selected.columns else 0,
        n_selected_raw=int((selected[domain_col] == "raw").sum()) if domain_col in selected.columns else 0,
        n_selected_cases=int(n_cases_selected),
        n_mixed_cases_in_input=int(n_mixed_input),
        n_mixed_cases_in_selected=int(n_mixed_selected),
        label_counts_selected=label_counts,
    )

    return selected, rep


def enforce_case_domain_consistency(
    df: pd.DataFrame,
    *,
    case_col: str = "case_name",
    domain_col: str = "source",
    seed: int = 42,
    prefer_domain: Literal["raw", "normalized"] = "raw",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Ensure each case appears in only one domain by dropping the other domain's rows.

    For cases that have both domains, we keep the preferred domain when available; otherwise we
    keep the domain with more patches; remaining ties are broken deterministically by seed.
    """
    if df.empty:
        return df.copy(), {"n_cases_mixed": 0, "n_rows_dropped": 0}

    rng = _rng(seed)
    keep_rows = []
    dropped = 0
    mixed_cases = 0

    for case, sub in df.groupby(case_col, dropna=False):
        domains = sub[domain_col].dropna().unique().tolist()
        if len(domains) <= 1:
            keep_rows.append(sub)
            continue

        mixed_cases += 1
        # decide keep domain
        if prefer_domain in domains:
            keep = prefer_domain
        else:
            counts = sub[domain_col].value_counts().to_dict()
            best = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
            if len(best) >= 2 and best[0][1] == best[1][1]:
                keep = rng.choice([best[0][0], best[1][0]])
            else:
                keep = best[0][0]

        kept_sub = sub[sub[domain_col] == keep]
        dropped += int(len(sub) - len(kept_sub))
        keep_rows.append(kept_sub)

    out = pd.concat(keep_rows, ignore_index=True)
    return out, {"n_cases_mixed": int(mixed_cases), "n_rows_dropped": int(dropped)}
