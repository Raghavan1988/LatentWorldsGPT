"""Aggregate the 15 symgroup multi-seed probe logs into a single
mean ± std table across the 3 variants (real / within-shuffled /
global-shuffled).

Each per-run log (e.g. `checkpoints/multiseed_w2_symgroup/symgroup_sa_seed0.log`)
contains a HEADLINE block that records best-layer mean accuracy for:
- position-level linear / MLP, trained / untrained
- word-level linear / MLP, trained / untrained

We parse these, group by (variant, condition, probe-family, split),
and report mean ± std across the 5 seeds for each cell.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "checkpoints" / "multiseed_w2_symgroup"

VARIANTS = ("symgroup_sa", "symgroup_sa_within", "symgroup_sa_global")
VARIANT_LABEL = {
    "symgroup_sa": "sa (real)",
    "symgroup_sa_within": "sa_within-shuffled",
    "symgroup_sa_global": "sa_global-shuffled",
}

# HEADLINE line format:
#   trained linear (mean)               L6  mean_acc=0.3479  best_element=0.4123
HEADLINE_RE = re.compile(
    r"^\s*(trained|untrained)\s+(linear|MLP)\s+\(mean\)\s+"
    r"(embed|L\d+)\s+mean_acc=([\d.]+)\s+best_element=([\d.]+)"
)


def parse_log(path):
    """Return {(probe_type, split): {'lin' or 'mlp': {'best_layer': str, 'acc': float}}}."""
    text = path.read_text()
    sections = {}
    cur_split = None
    for line in text.splitlines():
        if "POSITION-LEVEL:" in line:
            cur_split = "position"
            sections[cur_split] = {}
        elif "WORD-LEVEL" in line and "(held-out" in line:
            cur_split = "word"
            sections[cur_split] = {}
        m = HEADLINE_RE.match(line)
        if m and cur_split is not None:
            cond, family, layer, acc, best = m.groups()
            key = f"{cond}_{family.lower()}"  # e.g., "trained_linear", "untrained_mlp"
            sections[cur_split][key] = {
                "best_layer": layer,
                "acc": float(acc),
                "best_element": float(best),
            }
    return sections


def main():
    # results[variant][split][key] = list of accs across 5 seeds
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for variant in VARIANTS:
        for seed in range(5):
            path = LOG_DIR / f"{variant}_seed{seed}.log"
            if not path.exists():
                continue
            sections = parse_log(path)
            for split, entries in sections.items():
                for key, info in entries.items():
                    results[variant][split][key].append(info["acc"])

    print("=" * 78)
    print("Symgroup multi-seed at parity — best-layer mean across 5 seeds")
    print("=" * 78)
    for split in ("position", "word"):
        label = "POSITION-LEVEL" if split == "position" else "WORD-LEVEL (held-out)"
        print(f"\n## {label}\n")
        header = "| Variant              | trained linear   | trained MLP      | untrained linear | untrained MLP    | T-U linear | T-U MLP |"
        sep    = "|---|---|---|---|---|---|---|"
        print(header)
        print(sep)
        for variant in VARIANTS:
            data = results[variant][split]
            def fmt(vals):
                if not vals:
                    return "—"
                a = np.array(vals)
                if len(a) > 1:
                    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"
                else:
                    return f"{a[0]:.4f}"
            tl = fmt(data.get("trained_linear", []))
            tm = fmt(data.get("trained_mlp", []))
            ul = fmt(data.get("untrained_linear", []))
            um = fmt(data.get("untrained_mlp", []))
            tl_mean = np.mean(data.get("trained_linear", [0]))
            tm_mean = np.mean(data.get("trained_mlp", [0]))
            ul_mean = np.mean(data.get("untrained_linear", [0]))
            um_mean = np.mean(data.get("untrained_mlp", [0]))
            gap_lin = tl_mean - ul_mean
            gap_mlp = tm_mean - um_mean
            print(f"| {VARIANT_LABEL[variant]:<20} | {tl} | {tm} | {ul} | {um} | {gap_lin:+.4f} | {gap_mlp:+.4f} |")


if __name__ == "__main__":
    main()
