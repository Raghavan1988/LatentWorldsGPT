"""Aggregate per-layer transplant logs into a summary table.

For each (domain × condition × layer), reports mean ± std of the
transplant lift over unpatched across 5 seeds.
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "checkpoints" / "multiseed_w2_perlayer"

LIFT_UNP_PATTERNS = [
    r"TRANSPLANT lift on P\(B's nbrs\) over unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(B's legal moves\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"Δ P\(near B's RSVP\)\s+transplant − unpatched\s*:\s*([+-]?\d*\.?\d+)",
    r"^\s*P\(B-phase tokens\) gain\s+([+-]?\d*\.?\d+)\s+[+-]?\d*\.?\d+\s+[+-]?\d*\.?\d+",
]

DOMAINS = {
    "Cities": (
        [("cities_london_real", "London real"),
         ("cities_london_within", "London within-shuf"),
         ("cities_london_global", "London global-shuf"),
         ("cities_manhattan", "Manhattan"),
         ("cities_boston", "Boston")],
        6,
    ),
    "Othello": (
        [("othello", "Othello (50k)")],
        4,
    ),
    "Flight": (
        [("flight_real", "Flight real"),
         ("flight_within", "Flight within"),
         ("flight_global", "Flight global")],
        2,
    ),
    "Music VL": (
        [("music_real", "Music real"),
         ("music_within", "Music within"),
         ("music_global", "Music global")],
        3,
    ),
}


def find_first(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.MULTILINE)
        if m:
            return float(m.group(1))
    return None


def parse(tag, layer, seeds=range(5)):
    vals = []
    for seed in seeds:
        path = LOG_DIR / f"{tag}_layer{layer}_seed{seed}.log"
        if not path.exists():
            continue
        v = find_first(LIFT_UNP_PATTERNS, path.read_text())
        if v is not None:
            vals.append(v)
    return np.array(vals)


def main():
    print("# Per-layer transplant ablation — multi-seed summary\n")
    print("Lift over unpatched at each layer (mean ± std, 5 seeds).\n")
    for domain, (conditions, n_layer) in DOMAINS.items():
        print(f"## {domain}\n")
        header = "| Condition | " + " | ".join(f"L{L}" for L in range(n_layer)) + " | peak L | peak lift |"
        sep    = "|---|" + "|".join("---" for _ in range(n_layer)) + "|---|---|"
        print(header)
        print(sep)
        for tag, label in conditions:
            row_vals = []
            row_strs = []
            for L in range(n_layer):
                vals = parse(tag, L)
                if len(vals) == 0:
                    row_strs.append("—")
                    row_vals.append(None)
                else:
                    mean = vals.mean()
                    std = vals.std(ddof=1) if len(vals) > 1 else 0
                    row_strs.append(f"{mean:+.3f}±{std:.3f}")
                    row_vals.append(mean)
            valid = [(L, v) for L, v in enumerate(row_vals) if v is not None]
            if valid:
                peak_L, peak_v = max(valid, key=lambda x: x[1])
                peak_str = f"L{peak_L}"
                peak_v_str = f"{peak_v:+.3f}"
            else:
                peak_str = "—"
                peak_v_str = "—"
            print(f"| {label} | " + " | ".join(row_strs) + f" | {peak_str} | {peak_v_str} |")
        print()


if __name__ == "__main__":
    main()
