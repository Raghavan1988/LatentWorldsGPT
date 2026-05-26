"""
LatentWorldsGPT — Milestone 2 (music) valid voice-step evaluator.

Music-domain analogue of eval/valid_edge.py (cities). Asks the same harder
question: beyond cross-entropy, does the model's next-token distribution
correspond to MUSICALLY-LEGAL moves?

The "edge" in the music vocab is voice-leading: in 4-voice fixed-order
tokenization, the previous same-voice pitch is exactly 4 positions earlier.
Bach chorale style has tight voice-leading constraints — most consecutive
same-voice intervals are ≤ a perfect fifth (7 semitones), almost all are
≤ an octave (12 semitones). A model that learned chorale style should
predict next-pitches that respect these constraints.

  ┌──────────────────────────────────────────────────────────────────┐
  │  MODE A — next-step voice-leading validity                       │
  └──────────────────────────────────────────────────────────────────┘

      stream from gen.bin (held-out pieces):
        ... BOS  S1 A1 T1 B1  S2 A2 T2 B2  ...  EOS  ...

      For each interior position q (5 ≤ q < len(piece)):
          let prior = piece[q - 4]    (= previous same-voice pitch)
          let pred  = argmax model(piece[:q]).logits[-1]
          if both `prior` and `pred` are real pitches:
              interval = |midi(pred) - midi(prior)|
              count as "≤7 valid" iff interval ≤ 7
              count as "≤12 valid" iff interval ≤ 12

  ┌──────────────────────────────────────────────────────────────────┐
  │  MODE B — full-piece continuation                                │
  └──────────────────────────────────────────────────────────────────┘

      For each held-out piece, prompt with the first PROMPT_BEATS beats
      (= 4 * PROMPT_BEATS tokens after BOS). Free-generate the next
      GEN_BEATS beats. Walk the generated portion and apply the same
      voice-leading validity check at every position whose prior
      same-voice slot exists (which is always, given the prompt covers
      ≥ 1 beat).

      Stricter than next-step: per-step errors compound over generated
      sequences. A 90 %/step rate produces ≈ (0.9)^48 ≈ 0.6 % fully
      valid 12-beat continuations.

WHAT WE'RE PROVING (and what we are NOT)
========================================
This metric proves the model learned Bach voice-leading. It does NOT prove
anything about the model's representation of beat / chord / key — those are
the probe tasks (eval/probe_music.py). Voice-leading is the necessary
prerequisite ("earn the right to ask" about emergent musical structure).

Prediction pattern under the three-condition gradient (pivot.md M2):
    real London          → near-100% voice-leading rate (Bach is smooth)
    within-piece-shuffled → collapses; same set of pitches but order destroyed
    global-shuffled       → collapses harder; predictions revert to unigram

THE ONE RULE
============
This script never reads key.csv / chord.csv / beat.csv. It uses only
token IDs and meta.pkl's itos (token-id → MIDI pitch). No probe target
ever enters the model or its scoring loop.

USAGE
=====
    python eval/valid_voice_step.py --ckpt checkpoints/music_bach/best.pt \
        --data_dir data/music_bach --split gen

    # Just next-step mode
    python eval/valid_voice_step.py --ckpt checkpoints/music_bach/best.pt \
        --data_dir data/music_bach --mode next

    # Stricter tolerance band (semitones)
    python eval/valid_voice_step.py --ckpt checkpoints/music_bach/best.pt \
        --data_dir data/music_bach --strict_band 5
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import GPT, GPTConfig, BOS, EOS, PAD  # noqa: E402

REST = 3
N_RESERVED = 4
VOICE_NAMES = ("S", "A", "T", "B")


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_artifacts(data_dir: Path):
    with open(data_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return meta


def split_into_pieces(stream: np.ndarray) -> list[np.ndarray]:
    """Walk the stream and emit one ndarray per [BOS ... EOS) piece. The BOS
    is included (it's position 0 in the piece-local frame); the EOS is NOT
    included (it acts only as a piece terminator). Pieces with fewer than
    9 tokens (BOS + at least 2 beats of 4 voices) are skipped — there's no
    valid same-voice prior to score.
    """
    pieces = []
    i = 0
    while i < len(stream):
        if stream[i] != BOS:
            i += 1
            continue
        j = i + 1
        while j < len(stream) and stream[j] != EOS:
            j += 1
        piece = stream[i:j]  # [BOS, S1, A1, T1, B1, S2, ...]  (no EOS)
        if len(piece) >= 9:
            pieces.append(np.asarray(piece))
        i = j + 1
    return pieces


# ─────────────────────────────────────────────────────────────────────────────
# Mode A: next-step voice-leading
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def next_step_voice_leading(model: GPT, pieces: list[np.ndarray], itos: dict,
                            block_size: int, strict_band: int, loose_band: int,
                            device: str, vocab_size: int) -> dict:
    """For each piece, run the model in teacher-forced mode and score the
    greedy next-pitch prediction at every q ≥ 5 against the same-voice prior
    at position q-4.

    A position is "scored" iff the prior token (piece[q-4]) is a real pitch
    (token id ≥ N_RESERVED). Predictions are categorized:

      - real_pitch         : prediction is a real-pitch token (not control / REST)
      - within_strict_band : real_pitch AND |interval| ≤ strict_band
      - within_loose_band  : real_pitch AND |interval| ≤ loose_band

    Returns aggregated rates, plus a per-voice breakdown for diagnostics.
    """
    n_scored = 0
    n_pred_real = 0
    n_within_strict = 0
    n_within_loose = 0
    intervals = []
    per_voice = {v: {"scored": 0, "within_strict": 0, "within_loose": 0,
                     "intervals": []} for v in VOICE_NAMES}

    for piece in pieces:
        # truncate to block_size; longer chorales are rare in this corpus
        # but be defensive
        if len(piece) > block_size:
            piece = piece[:block_size]
        x = torch.from_numpy(piece.astype(np.int64)).unsqueeze(0).to(device)
        # ONE-RULE check
        assert int(x.min()) >= 0 and int(x.max()) < vocab_size

        logits, _ = model(x)               # (1, T, V)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # (T,)

        # For each piece-local position q (1 = first S note, 2 = first A, ...)
        # the model's prediction for position q is preds[q - 1] (logits at
        # q-1 predict q). We need q - 4 ≥ 1 (so q ≥ 5).
        L = len(piece)
        for q in range(5, L):
            prior_tok = int(piece[q - 4])
            if prior_tok < N_RESERVED:
                continue          # prior is PAD/BOS/EOS/REST → no melodic interval
            pred_tok = int(preds[q - 1])
            n_scored += 1
            voice = VOICE_NAMES[(q - 1) % 4]
            per_voice[voice]["scored"] += 1

            if pred_tok < N_RESERVED:
                continue          # prediction is control / REST → not a real pitch

            n_pred_real += 1
            interval = abs(itos[pred_tok] - itos[prior_tok])
            intervals.append(interval)
            per_voice[voice]["intervals"].append(interval)
            if interval <= strict_band:
                n_within_strict += 1
                per_voice[voice]["within_strict"] += 1
            if interval <= loose_band:
                n_within_loose += 1
                per_voice[voice]["within_loose"] += 1

    voice_summary = {}
    for v, d in per_voice.items():
        if d["scored"] == 0:
            voice_summary[v] = None
            continue
        median_iv = float(np.median(d["intervals"])) if d["intervals"] else float("nan")
        voice_summary[v] = {
            "scored":        d["scored"],
            "rate_strict":   d["within_strict"] / d["scored"],
            "rate_loose":    d["within_loose"]  / d["scored"],
            "median_iv":     median_iv,
        }

    return {
        "n_scored":           n_scored,
        "n_pred_real":        n_pred_real,
        "real_pitch_rate":    n_pred_real / max(1, n_scored),
        "rate_strict":        n_within_strict / max(1, n_scored),
        "rate_loose":         n_within_loose  / max(1, n_scored),
        "median_iv":          float(np.median(intervals)) if intervals else float("nan"),
        "per_voice":          voice_summary,
        "strict_band":        strict_band,
        "loose_band":         loose_band,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode B: full-piece continuation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_piece_continuation(model: GPT, pieces: list[np.ndarray], itos: dict,
                            prompt_beats: int, gen_beats: int,
                            strict_band: int, loose_band: int,
                            device: str, temperature: float,
                            vocab_size: int) -> dict:
    """For each piece, prompt with [BOS] + first (4 * prompt_beats) tokens,
    free-generate (4 * gen_beats) more tokens, score voice-leading on the
    generated portion only (using either earlier generated tokens or prompt
    tokens as the same-voice prior).
    """
    n_generated_scored = 0
    n_within_strict = 0
    n_within_loose = 0
    n_pred_real = 0
    n_fully_valid_pieces = 0

    for piece in pieces:
        prompt_len = 1 + 4 * prompt_beats  # BOS + N beats of 4 voices
        if len(piece) <= prompt_len:
            continue
        prompt = torch.from_numpy(piece[:prompt_len].astype(np.int64)).unsqueeze(0).to(device)
        out = model.generate(prompt, max_new_tokens=4 * gen_beats,
                             temperature=temperature)
        seq = out.squeeze(0).cpu().numpy()
        # seq has BOS at position 0, prompt + generated continuation following.
        gen_start = prompt_len
        gen_end = min(len(seq), prompt_len + 4 * gen_beats)

        piece_valid = True
        n_in_piece = 0
        n_strict_in_piece = 0
        for q in range(gen_start, gen_end):
            if q < 5:
                continue
            prior_tok = int(seq[q - 4])
            if prior_tok < N_RESERVED:
                # generation produced a control/REST in the prior slot — skip
                continue
            pred_tok = int(seq[q])
            n_generated_scored += 1
            n_in_piece += 1

            if pred_tok < N_RESERVED:
                piece_valid = False
                continue
            n_pred_real += 1
            interval = abs(itos[pred_tok] - itos[prior_tok])
            if interval <= strict_band:
                n_within_strict += 1
                n_strict_in_piece += 1
            else:
                piece_valid = False
            if interval <= loose_band:
                n_within_loose += 1

        # "Fully valid" = every scored position passes the strict band.
        if n_in_piece > 0 and piece_valid:
            n_fully_valid_pieces += 1

    return {
        "n_generated_scored":     n_generated_scored,
        "n_pred_real":            n_pred_real,
        "real_pitch_rate":        n_pred_real / max(1, n_generated_scored),
        "rate_strict":            n_within_strict / max(1, n_generated_scored),
        "rate_loose":             n_within_loose  / max(1, n_generated_scored),
        "fully_valid_piece_rate": n_fully_valid_pieces / max(1, len(pieces)),
        "n_pieces":               len(pieces),
        "strict_band":            strict_band,
        "loose_band":             loose_band,
        "prompt_beats":           prompt_beats,
        "gen_beats":              gen_beats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", choices=("val", "gen"), default="gen",
                   help="which .bin to use (default: gen = held-out pieces)")
    p.add_argument("--mode", choices=("next", "full", "both"), default="both")
    p.add_argument("--strict_band", type=int, default=7,
                   help="strict voice-leading tolerance in semitones (default 7 = perfect fifth)")
    p.add_argument("--loose_band", type=int, default=12,
                   help="loose voice-leading tolerance in semitones (default 12 = octave)")
    p.add_argument("--prompt_beats", type=int, default=4)
    p.add_argument("--gen_beats", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() \
             else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    data_dir = Path(args.data_dir)
    model, ckpt = load_model(args.ckpt, device)
    meta = load_artifacts(data_dir)
    itos = meta["itos"]
    vocab_size = ckpt["vocab_size"]
    block_size = model.config.block_size

    print(f"loaded ckpt: iter={ckpt.get('iter','?')}  "
          f"val_ppl={ckpt.get('val_perplexity',float('nan')):.4f}")

    dtype = np.dtype(meta["dtype"])
    stream = np.asarray(np.memmap(data_dir / f"{args.split}.bin",
                                  dtype=dtype, mode="r"))
    pieces = split_into_pieces(stream)
    print(f"{args.split}.bin: {len(stream):,} tokens / {len(pieces)} pieces")

    # ── Mode A: next-step ──
    if args.mode in ("next", "both"):
        result = next_step_voice_leading(
            model, pieces, itos, block_size,
            args.strict_band, args.loose_band, device, vocab_size,
        )
        print(f"\n[next-step / {args.split}.bin]"
              f"  (strict ≤{args.strict_band}st, loose ≤{args.loose_band}st)")
        print(f"  positions scored        : {result['n_scored']:,}")
        print(f"  real-pitch prediction   : {result['real_pitch_rate']*100:6.2f}%")
        print(f"  voice-leading ≤{args.strict_band}st (strict) : {result['rate_strict']*100:6.2f}%")
        print(f"  voice-leading ≤{args.loose_band}st (loose)  : {result['rate_loose']*100:6.2f}%")
        print(f"  median |interval| (st)  : {result['median_iv']:6.2f}")
        print(f"  per-voice rates (strict):")
        for v, d in result["per_voice"].items():
            if d is None:
                print(f"    {v}: no positions scored"); continue
            print(f"    {v}: scored={d['scored']:>5,d}  "
                  f"rate={d['rate_strict']*100:6.2f}%  "
                  f"median_iv={d['median_iv']:5.2f}")

    # ── Mode B: full-piece continuation ──
    if args.mode in ("full", "both"):
        result = full_piece_continuation(
            model, pieces, itos, args.prompt_beats, args.gen_beats,
            args.strict_band, args.loose_band, device, args.temperature, vocab_size,
        )
        print(f"\n[full-piece continuation, temperature={args.temperature}]"
              f"  prompt={args.prompt_beats} beats, gen={args.gen_beats} beats")
        print(f"  positions scored        : {result['n_generated_scored']:,} "
              f"(over {result['n_pieces']} pieces)")
        print(f"  real-pitch prediction   : {result['real_pitch_rate']*100:6.2f}%")
        print(f"  voice-leading ≤{args.strict_band}st (strict) : {result['rate_strict']*100:6.2f}%")
        print(f"  voice-leading ≤{args.loose_band}st (loose)  : {result['rate_loose']*100:6.2f}%")
        print(f"  fully-valid piece rate  : {result['fully_valid_piece_rate']*100:6.2f}%  "
              f"(every generated position passes the strict band)")


if __name__ == "__main__":
    main()
