"""
LatentWorldsGPT — Milestone 2 (music) dataset preparation.

Turns the Bach chorale corpus (music21) into a token corpus for next-note
language modeling, plus three probe-target tables the model never sees during
training.

Outputs (all in --out_dir):
    train.bin    uint16 token stream — 80% of pieces (piece-level split)
    val.bin      uint16 token stream — 10% (in-distribution perplexity)
    gen.bin      uint16 token stream — 10% (held-out test: pieces never seen
                 by the model, used for the probe eval)
    meta.pkl     {vocab_size, stoi, itos, dtype, pad, bos, eos, rest, ...}
    key.csv      split,piece_idx,token_pos,mode     (1=major / 0=minor)
    chord.csv    split,piece_idx,token_pos,roman    (Roman-numeral figure string)
    beat.csv     split,piece_idx,token_pos,beat     (1..4 for 4/4 — the
                 load-bearing Othello-positive probe target)

THE ONE RULE (music edition): no probe-target value (key/mode, chord Roman
numeral, beat-in-measure) ever enters the token stream. Tokens are pitch
identifiers only. Probe targets live in the CSV side tables and are read only
by the probe / eval code.

Token convention: 0=PAD, 1=BOS, 2=EOS, 3=REST; real pitches start at index 4.
Sequence layout per piece: [BOS, S_1, A_1, T_1, B_1, S_2, A_2, T_2, B_2, ...,
EOS] — at each integer-quarter-note beat we emit the 4 voices' pitches in
fixed order (Soprano, Alto, Tenor, Bass). Voice can be recovered from
(position % 4); beat-in-measure cannot (next-token windows are randomly
offset by the trainer, so BOS is not at a fixed phase to the bar).

All pieces are transposed so majors -> C major and minors -> A minor. The key
probe therefore degenerates to a binary major-vs-minor (mode) classification —
this is intentional: it isolates the mode signal from the pitch-class signal.
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np

PAD, BOS, EOS, REST = 0, 1, 2, 3
N_RESERVED = 4


# ---------------------------------------------------------------------------
# 1. Load + filter the Bach chorale corpus
# ---------------------------------------------------------------------------
def load_chorales(limit=None, composers=("bach",)):
    """Yield (label, Score) for pieces by the given composers.

    'bach' uses corpus.chorales.Iterator (~371 Bach chorales).
    Any other composer name goes through corpus.search(composer, 'composer')
    and yields the parsed Score for each metadata bundle entry.

    Downstream code filters by meter / voice count.
    """
    from music21 import corpus
    yielded = 0
    for composer in composers:
        if composer == "bach":
            it = corpus.chorales.Iterator(returnType="stream")
            for i, score in enumerate(it):
                if limit is not None and yielded >= limit:
                    return
                label = getattr(score.metadata, "title", None) or f"bach_chorale_{i}"
                yielded += 1
                yield label, score
        else:
            results = corpus.search(composer, "composer")
            for j, md in enumerate(results):
                if limit is not None and yielded >= limit:
                    return
                try:
                    score = md.parse()
                except Exception:
                    continue
                label = getattr(score.metadata, "title", None) or f"{composer}_{j}"
                yielded += 1
                yield label, score


def is_44_satb(score, accept_4_2: bool = True) -> bool:
    """Filter: keep pieces with 4 parts (SATB) and 4-beats-per-measure meter.

    Accepts 4/4 by default; with accept_4_2=True also accepts 4/2 (same
    beat structure, just longer note-length convention common in renaissance
    polyphony). Rejects pieces with changing time signatures.
    """
    from music21 import meter
    parts = list(score.parts)
    if len(parts) != 4:
        return False
    tss = list(score.recurse().getElementsByClass(meter.TimeSignature))
    if not tss:
        return False
    ratios = {ts.ratioString for ts in tss}
    allowed = {"4/4"} | ({"4/2"} if accept_4_2 else set())
    if not ratios.issubset(allowed):
        return False
    if len(ratios) != 1:
        return False  # changing meter
    return True


# ---------------------------------------------------------------------------
# 2. Normalize: transpose major -> C, minor -> a
# ---------------------------------------------------------------------------
def normalize_key(score):
    """Transpose so all major pieces sit in C major and all minor in A minor.

    Returns (transposed_score, mode_label) where mode_label is 1 for major,
    0 for minor. The transposition collapses the 24 keys to a 2-class mode
    probe target; that's intentional (see module docstring).
    """
    from music21 import interval, pitch
    k = score.analyze("key")
    target_tonic = "C" if k.mode == "major" else "A"
    iv = interval.Interval(k.tonic, pitch.Pitch(target_tonic))
    transposed = score.transpose(iv)
    return transposed, (1 if k.mode == "major" else 0)


# ---------------------------------------------------------------------------
# 3. Per-piece encoding: 4-voice quarter-note grid -> token stream + probe labels
# ---------------------------------------------------------------------------
def _beat_step_quarters(score) -> float:
    """How many quarter-notes equal one BEAT for the score's time signature?

    4/4 → 1.0  (quarter-note beat)
    4/2 → 2.0  (half-note beat)

    Returns 1.0 for unknown meters (treat as 4/4-equivalent).
    """
    from music21 import meter
    tss = list(score.recurse().getElementsByClass(meter.TimeSignature))
    if not tss:
        return 1.0
    ratio = tss[0].ratioString
    if ratio == "4/2":
        return 2.0
    return 1.0


def piece_to_beat_grid(score):
    """Return per-beat 4-tuple of (S, A, T, B) MIDI pitches.

    For each integer-BEAT offset in [0, n_beats), we look up each part's
    currently-sounding note. The 'beat' length depends on the time
    signature: 1 quarter-note for 4/4, 2 quarter-notes for 4/2. This keeps
    the encoding meter-agnostic — every emitted token-tuple corresponds to
    one beat regardless of notation.

    Returns list of tuples; entries are MIDI ints, or None for rest / no
    note sounding.
    """
    step = _beat_step_quarters(score)
    total_quarters = int(round(score.highestTime))
    n_beats = int(total_quarters / step)
    parts = list(score.parts)[:4]  # S, A, T, B
    voice_streams = [part.flatten().notes for part in parts]

    grid = []
    for b in range(n_beats):
        q = b * step  # quarter-note offset of this beat
        voices = []
        for vs in voice_streams:
            # find note whose [offset, offset+duration) contains q
            current = None
            for n in vs:
                start = n.offset
                end = start + n.duration.quarterLength
                if start <= q < end:
                    if n.isChord:
                        current = max(p.midi for p in n.pitches)
                    else:
                        current = n.pitch.midi
                    break
            voices.append(current)
        grid.append(tuple(voices))
    return grid


def beat_in_measure(beat_offset_quarters, time_sig_numerator=4):
    """1-indexed beat position within the measure (1..N for N/4)."""
    return (beat_offset_quarters % time_sig_numerator) + 1


def roman_at_beat(voice_pitches, key_obj):
    """Roman-numeral figure for the chord formed by the 4 voices, in `key_obj`.

    Returns a short string ('I', 'V7', 'ii6', '?', etc.). Returns '?' for
    rests-only or chords music21 can't analyze.
    """
    from music21 import chord, roman
    midis = [p for p in voice_pitches if p is not None]
    if not midis:
        return "?"
    try:
        c = chord.Chord(midis)
        rn = roman.romanNumeralFromChord(c, key_obj)
        return rn.figure
    except Exception:
        return "?"


def encode_piece(score, stoi, piece_idx, mode_label, key_obj):
    """Encode one normalized piece into (tokens, key_labels, chord_labels,
    beat_labels). Lengths all equal.

    BOS / EOS / PAD positions are labeled with sentinel `-1` for every probe
    target so the probe-side code can skip them trivially.
    """
    grid = piece_to_beat_grid(score)

    tokens = [BOS]
    key_labels = [-1]
    chord_labels = ["?"]
    beat_labels = [-1]

    for q, voices in enumerate(grid):
        bim = beat_in_measure(q, 4)
        rn = roman_at_beat(voices, key_obj)
        for p in voices:
            tok = stoi[p] if p is not None else REST
            tokens.append(tok)
            key_labels.append(mode_label)
            chord_labels.append(rn)
            beat_labels.append(bim)

    tokens.append(EOS)
    key_labels.append(-1)
    chord_labels.append("?")
    beat_labels.append(-1)
    return tokens, key_labels, chord_labels, beat_labels


# ---------------------------------------------------------------------------
# 4. Tokenizer: pitch-MIDI -> contiguous index
# ---------------------------------------------------------------------------
def build_tokenizer(all_pitches):
    """Bijection MIDI-pitch <-> token index. PAD/BOS/EOS/REST reserved as 0-3."""
    sorted_pitches = sorted(set(all_pitches))
    stoi = {p: i + N_RESERVED for i, p in enumerate(sorted_pitches)}
    itos = {i: p for p, i in stoi.items()}
    vocab_size = len(stoi) + N_RESERVED
    return stoi, itos, vocab_size


# ---------------------------------------------------------------------------
# 5. Destroyed-structure controls
# ---------------------------------------------------------------------------
def shuffle_within_piece(tokens, rng):
    """Weak control: shuffle the real-pitch tokens within each [BOS..EOS]
    span, leaving BOS/EOS markers in place.

    Preserves per-piece pitch-set membership (so mode/key probe should still
    survive; cities-analogue prediction). Destroys voice-position ordering
    and beat structure within each piece (so beat probe should collapse).
    """
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == BOS:
            j = i + 1
            while j < len(tokens) and tokens[j] != EOS:
                j += 1
            interior = list(tokens[i + 1 : j])
            rng.shuffle(interior)
            out.append(BOS)
            out.extend(interior)
            if j < len(tokens):
                out.append(EOS)
            i = j + 1
        else:
            out.append(tokens[i])
            i += 1
    return out


def shuffle_globally(tokens, rng):
    """Strict control: shuffle every real-pitch token across the entire
    stream. BOS/EOS positions stay in place. Destroys both ordering and
    piece-level set membership.
    """
    real_positions = [i for i, t in enumerate(tokens) if t >= N_RESERVED]
    real_values = [tokens[i] for i in real_positions]
    rng.shuffle(real_values)
    out = list(tokens)
    for pos, val in zip(real_positions, real_values):
        out[pos] = val
    return out


# ---------------------------------------------------------------------------
# 6. Piece-level split
# ---------------------------------------------------------------------------
def split_pieces(piece_indices, val_frac, gen_frac, rng):
    """Piece-level disjoint split. gen pieces are the test holdout."""
    indices = list(piece_indices)
    rng.shuffle(indices)
    n = len(indices)
    n_gen = int(n * gen_frac)
    n_val = int(n * val_frac)
    gen = set(indices[:n_gen])
    val = set(indices[n_gen : n_gen + n_val])
    train = set(indices[n_gen + n_val :])
    return train, val, gen


# ---------------------------------------------------------------------------
# 7. Dump to disk in nanoGPT's expected format + probe target CSVs
# ---------------------------------------------------------------------------
def pick_dtype(vocab_size):
    return np.uint16 if vocab_size < 2**16 else np.uint32


def dump(out_dir, splits, stoi, itos, vocab_size):
    """splits: dict[name -> {"tokens": [...], "key": [...], "chord": [...],
    "beat": [...], "piece_token_starts": [(piece_idx, start, end), ...]}]

    Each piece's probe-target rows are emitted as
    (split, piece_idx, token_pos_in_split, value).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = pick_dtype(vocab_size)

    for name, payload in splits.items():
        toks = payload["tokens"]
        arr = np.array(toks, dtype=dtype)
        if arr.size:
            assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size, (
                f"{name}.bin contains out-of-vocab values "
                f"(min={int(arr.min())}, max={int(arr.max())}, "
                f"vocab_size={vocab_size})"
            )
        arr.tofile(out / f"{name}.bin")

    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(
            {
                "vocab_size": vocab_size,
                "stoi": stoi,
                "itos": itos,
                "dtype": np.dtype(dtype).name,
                "pad": PAD,
                "bos": BOS,
                "eos": EOS,
                "rest": REST,
                "voices_per_beat": 4,
                "voice_order": ["S", "A", "T", "B"],
                "time_signature": "4/4",
            },
            f,
        )

    # Probe target CSVs: one row per (split, piece_idx, token_pos_in_split)
    # for non-sentinel positions. The probe code joins these against
    # activations from train/val/gen.bin.
    with open(out / "key.csv", "w", newline="") as fk, \
         open(out / "chord.csv", "w", newline="") as fc, \
         open(out / "beat.csv", "w", newline="") as fb:
        kw = csv.writer(fk); kw.writerow(["split", "piece_idx", "token_pos", "mode"])
        cw = csv.writer(fc); cw.writerow(["split", "piece_idx", "token_pos", "roman"])
        bw = csv.writer(fb); bw.writerow(["split", "piece_idx", "token_pos", "beat"])

        for name, payload in splits.items():
            key_lbl = payload["key"]
            chord_lbl = payload["chord"]
            beat_lbl = payload["beat"]
            # which piece each token belongs to: we wrote piece_token_starts
            # as a sequence of (piece_idx, start, end) ranges.
            for piece_idx, start, end in payload["piece_token_starts"]:
                for pos in range(start, end):
                    if key_lbl[pos] == -1:
                        continue  # BOS/EOS sentinel
                    kw.writerow([name, piece_idx, pos, key_lbl[pos]])
                    cw.writerow([name, piece_idx, pos, chord_lbl[pos]])
                    bw.writerow([name, piece_idx, pos, beat_lbl[pos]])

    return dtype


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------
def build_corpus(args, rng):
    """Pull, filter, normalize, encode the chorales. Returns the splits dict
    in dump()'s expected shape, plus stoi/itos/vocab_size.
    """
    from music21 import key as key_module

    print(f"[1/5] loading pieces from music21 corpus (composers={args.composers}) ...")
    pieces = []  # list of (mode_label, transposed_score, key_obj)
    for label, score in load_chorales(limit=args.limit, composers=tuple(args.composers)):
        if not is_44_satb(score):
            continue
        try:
            transposed, mode = normalize_key(score)
        except Exception as e:
            continue
        # After normalization, key is C major or a minor — explicit objects:
        k_obj = key_module.Key("C") if mode == 1 else key_module.Key("a")
        pieces.append((mode, transposed, k_obj))
    print(f"      {len(pieces):,} chorales pass 4/4 + SATB filter")
    if not pieces:
        raise SystemExit("no chorales survived filtering; nothing to do")

    print("[2/5] collecting pitch vocabulary ...")
    all_pitches = set()
    grids = []  # cache per-piece beat grids
    for mode, score, k_obj in pieces:
        g = piece_to_beat_grid(score)
        grids.append(g)
        for voices in g:
            for p in voices:
                if p is not None:
                    all_pitches.add(p)
    stoi, itos, vocab_size = build_tokenizer(all_pitches)
    print(f"      vocab_size = {vocab_size} (pitches + 4 reserved)")

    print("[3/5] splitting pieces 80/10/10 (train / val / gen) ...")
    train_set, val_set, gen_set = split_pieces(
        range(len(pieces)), args.val_frac, args.gen_frac, rng
    )
    print(
        f"      train={len(train_set):,}  val={len(val_set):,}  "
        f"gen={len(gen_set):,}"
    )

    print("[4/5] encoding pieces ...")
    splits = {
        "train": {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
        "val":   {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
        "gen":   {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
    }

    for piece_idx, ((mode, score, k_obj), grid) in enumerate(zip(pieces, grids)):
        if piece_idx in train_set:
            name = "train"
        elif piece_idx in val_set:
            name = "val"
        else:
            name = "gen"
        toks, kl, cl, bl = encode_piece(score, stoi, piece_idx, mode, k_obj)
        start = len(splits[name]["tokens"])
        splits[name]["tokens"].extend(toks)
        splits[name]["key"].extend(kl)
        splits[name]["chord"].extend(cl)
        splits[name]["beat"].extend(bl)
        end = len(splits[name]["tokens"])
        splits[name]["piece_token_starts"].append((piece_idx, start, end))

    # destroyed-structure controls applied AFTER encoding
    if args.shuffle_within_piece:
        print("[4b/5] applying within-piece shuffle (weak destroyed-structure) ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_within_piece(
                splits[name]["tokens"], rng
            )
    if args.shuffle_globally:
        print("[4c/5] applying GLOBAL shuffle (strict destroyed-structure) ...")
        for name in splits:
            splits[name]["tokens"] = shuffle_globally(
                splits[name]["tokens"], rng
            )

    return splits, stoi, itos, vocab_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/music_bach")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.10,
                   help="fraction of pieces for in-distribution validation")
    p.add_argument("--gen_frac", type=float, default=0.10,
                   help="fraction of pieces for held-out test (probe eval)")
    p.add_argument("--limit", type=int, default=None,
                   help="optional cap on number of pieces (for smoke runs)")
    p.add_argument("--composers", nargs="+", default=["bach"],
                   help="composer(s) to pull from the music21 corpus. "
                        "'bach' uses the Bach chorale iterator; other names "
                        "(palestrina, monteverdi, josquin, ...) use "
                        "corpus.search(name, 'composer'). Multiple allowed.")
    p.add_argument("--shuffle_within_piece", action="store_true",
                   help="weak destroyed-structure control: shuffle pitch "
                        "tokens within each piece. Predicted to leave the "
                        "mode/chord probes intact (cities-analogue) while "
                        "collapsing the beat probe (Othello-positive).")
    p.add_argument("--shuffle_globally", action="store_true",
                   help="strict destroyed-structure control: shuffle every "
                        "real-pitch token across the entire stream. Should "
                        "collapse all three probes to chance.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    splits, stoi, itos, vocab_size = build_corpus(args, rng)

    print(f"[5/5] writing to {args.out_dir} ...")
    dtype = dump(args.out_dir, splits, stoi, itos, vocab_size)

    print("\ndone.")
    for name in ("train", "val", "gen"):
        print(f"  {name}.bin : {len(splits[name]['tokens']):>10,} tokens")
    print(f"  dtype     : {np.dtype(dtype).name}")


if __name__ == "__main__":
    main()
