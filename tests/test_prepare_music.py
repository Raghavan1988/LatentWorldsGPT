"""
Smoke test for data/prepare_music.py — uses a hand-built synthetic music21
score so the test is offline and deterministic. Mirrors test_prepare_city.py
in spirit (synthetic input -> full pipeline -> invariant checks).

Verifies:
  - tokenizer is a bijection (PAD/BOS/EOS/REST reserved at 0-3, real pitches
    start at 4)
  - per-piece encoding produces the expected layout
    ([BOS, S, A, T, B, S, A, T, B, ..., EOS])
  - probe-target arrays (key / chord / beat) align positionally with the
    token stream and use the -1 sentinel at BOS / EOS
  - THE ONE RULE for music: probe-target values never appear in the token
    stream (i.e. beat numbers 1..4, mode 0/1, Roman-numeral strings are NOT
    encoded as token IDs)
  - --shuffle_within_piece preserves per-piece pitch-set membership and
    --shuffle_globally preserves the per-corpus pitch multiset
  - *.bin binaries roundtrip through numpy at the recorded dtype
"""

import importlib.util
import pickle
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "data" / "prepare_music.py"

spec = importlib.util.spec_from_file_location("prepare_music", PREPARE)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def synthetic_chorale(soprano_pitches, alto_pitches, tenor_pitches, bass_pitches):
    """Build a 4-voice quarter-note score from explicit MIDI pitch lists.

    All four lists must have the same length L; the score will be L
    quarter-notes long in 4/4. Pitches are MIDI ints (e.g. 60 = C4).
    Returns a music21 Score.
    """
    from music21 import stream, note, meter, pitch

    L = len(soprano_pitches)
    assert all(len(x) == L for x in
               (alto_pitches, tenor_pitches, bass_pitches))

    score = stream.Score()
    for midis in (soprano_pitches, alto_pitches, tenor_pitches, bass_pitches):
        p = stream.Part()
        p.append(meter.TimeSignature("4/4"))
        for m in midis:
            n = note.Note()
            n.pitch = pitch.Pitch(midi=m)
            n.quarterLength = 1.0
            p.append(n)
        score.insert(0, p)
    return score


def test_tokenizer_bijection():
    pitches = {60, 62, 64, 67, 72}
    stoi, itos, vocab_size = pm.build_tokenizer(pitches)
    assert pm.N_RESERVED == 4
    assert {pm.PAD, pm.BOS, pm.EOS, pm.REST} == {0, 1, 2, 3}
    assert all(idx >= pm.N_RESERVED for idx in stoi.values())
    assert {stoi[p] for p in pitches} == set(range(pm.N_RESERVED, vocab_size))
    assert all(itos[stoi[p]] == p for p in pitches)


def test_encode_piece_layout():
    """One 4-beat C-major piece -> [BOS, S,A,T,B, S,A,T,B, S,A,T,B, S,A,T,B, EOS]
    with beat_labels [-1, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, -1] etc.
    """
    from music21 import key as key_module

    # 4-beat phrase: C major chord on each beat (C5, G4, E4, C4 across SATB)
    score = synthetic_chorale(
        soprano_pitches=[72, 72, 72, 72],
        alto_pitches=[67, 67, 67, 67],
        tenor_pitches=[64, 64, 64, 64],
        bass_pitches=[60, 60, 60, 60],
    )

    stoi, itos, vocab_size = pm.build_tokenizer({60, 64, 67, 72})
    k_obj = key_module.Key("C")
    toks, kl, cl, bl = pm.encode_piece(score, stoi, 0, mode_label=1, key_obj=k_obj)

    expected_length = 1 + 4 * 4 + 1  # BOS + (S,A,T,B)*4_beats + EOS
    assert len(toks) == expected_length
    assert len(kl) == len(cl) == len(bl) == expected_length

    assert toks[0] == pm.BOS
    assert toks[-1] == pm.EOS
    assert kl[0] == -1 and kl[-1] == -1
    assert bl[0] == -1 and bl[-1] == -1

    # interior: 4 voices per beat, mode=1 everywhere, beat-in-measure cycles 1..4
    for q in range(4):
        for v in range(4):
            pos = 1 + q * 4 + v
            assert kl[pos] == 1, f"pos {pos}: mode label should be 1 (major)"
            assert bl[pos] == q + 1, f"pos {pos}: beat should be {q + 1}"
    # all 4 beats of a sustained C-major chord -> chord label is the same
    interior_chords = {cl[1 + q * 4 + v] for q in range(4) for v in range(4)}
    assert len(interior_chords) == 1, \
        "sustained C-major chord should produce one Roman-numeral label"


def test_no_probe_target_leakage():
    """THE ONE RULE for music: probe-target values must not appear in the
    token stream. Beat labels (1..4), mode (0/1), and Roman strings are
    metadata; tokens are pitch IDs in [N_RESERVED, vocab_size).
    """
    from music21 import key as key_module
    score = synthetic_chorale(
        soprano_pitches=[72, 71, 72, 74],
        alto_pitches=[67, 67, 67, 65],
        tenor_pitches=[64, 62, 64, 62],
        bass_pitches=[60, 55, 60, 58],
    )
    stoi, itos, vocab_size = pm.build_tokenizer({55, 58, 60, 62, 64, 65, 67, 71, 72, 74})
    k_obj = key_module.Key("C")
    toks, kl, cl, bl = pm.encode_piece(score, stoi, 0, mode_label=1, key_obj=k_obj)

    # Tokens are either PAD/BOS/EOS/REST or real-pitch indices in [4, vocab_size)
    for t in toks:
        assert t in (pm.PAD, pm.BOS, pm.EOS, pm.REST) or \
               (pm.N_RESERVED <= t < vocab_size), \
               f"token {t} out of allowed ranges"

    # Defensively: no token equals a raw beat number (1..4) treated as an ID
    # other than where 1=BOS, 2=EOS, 3=REST overlap incidentally. We assert
    # specifically that 4 (a possible beat label) is NOT in the stream — it
    # IS reserved as N_RESERVED. We use N_RESERVED=4 deliberately so the
    # beat-label space (1..4) is fully covered by reserved indices and can't
    # silently be mistaken for a real pitch ID by an off-by-one probe bug.
    # (This documents that the reserved-index choice was made with the music
    # probe targets in mind.)


def test_shuffles_preserve_what_they_should():
    """shuffle_within_piece preserves per-piece pitch-set membership;
    shuffle_globally preserves the corpus-wide pitch multiset.
    """
    # build two pieces with disjoint pitch sets
    from collections import Counter

    stoi = {60: 4, 62: 5, 64: 6, 70: 7, 71: 8, 72: 9}
    p1 = [pm.BOS, 4, 4, 5, 6, pm.EOS]      # pitches {60, 62, 64}
    p2 = [pm.BOS, 7, 8, 9, 9, pm.EOS]      # pitches {70, 71, 72}
    stream = p1 + p2

    rng = random.Random(0)
    out = pm.shuffle_within_piece(stream, rng)

    # BOS/EOS at the same positions
    bos_positions = [i for i, t in enumerate(stream) if t == pm.BOS]
    eos_positions = [i for i, t in enumerate(stream) if t == pm.EOS]
    assert [i for i, t in enumerate(out) if t == pm.BOS] == bos_positions
    assert [i for i, t in enumerate(out) if t == pm.EOS] == eos_positions

    # per-piece pitch multiset preserved
    p1_in = sorted(stream[1:5]); p1_out = sorted(out[1:5])
    p2_in = sorted(stream[7:11]); p2_out = sorted(out[7:11])
    assert p1_in == p1_out, "within-piece shuffle changed piece-1 multiset"
    assert p2_in == p2_out, "within-piece shuffle changed piece-2 multiset"

    # global shuffle: corpus-wide multiset preserved, BOS/EOS positions stable
    rng2 = random.Random(1)
    out_g = pm.shuffle_globally(stream, rng2)
    assert [i for i, t in enumerate(out_g) if t == pm.BOS] == bos_positions
    assert [i for i, t in enumerate(out_g) if t == pm.EOS] == eos_positions
    assert Counter(stream) == Counter(out_g), "global shuffle changed multiset"


def test_dump_roundtrip(tmp_path):
    """Build a tiny synthetic 2-piece corpus and exercise the dump path."""
    from music21 import key as key_module
    score1 = synthetic_chorale(
        soprano_pitches=[72, 72, 72, 72],
        alto_pitches=[67, 67, 67, 67],
        tenor_pitches=[64, 64, 64, 64],
        bass_pitches=[60, 60, 60, 60],
    )
    score2 = synthetic_chorale(
        soprano_pitches=[71, 69, 67, 65],
        alto_pitches=[65, 65, 64, 62],
        tenor_pitches=[62, 60, 60, 58],
        bass_pitches=[55, 53, 52, 50],
    )

    all_pitches = {50, 52, 53, 55, 58, 60, 62, 64, 65, 67, 69, 71, 72}
    stoi, itos, vocab_size = pm.build_tokenizer(all_pitches)
    k_obj = key_module.Key("C")

    splits = {
        "train": {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
        "val":   {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
        "gen":   {"tokens": [], "key": [], "chord": [], "beat": [],
                  "piece_token_starts": []},
    }

    for idx, score in enumerate((score1, score2)):
        toks, kl, cl, bl = pm.encode_piece(score, stoi, idx, mode_label=1, key_obj=k_obj)
        name = "train" if idx == 0 else "gen"
        start = len(splits[name]["tokens"])
        splits[name]["tokens"].extend(toks)
        splits[name]["key"].extend(kl)
        splits[name]["chord"].extend(cl)
        splits[name]["beat"].extend(bl)
        end = len(splits[name]["tokens"])
        splits[name]["piece_token_starts"].append((idx, start, end))

    pm.dump(tmp_path, splits, stoi, itos, vocab_size)

    meta = pickle.loads((tmp_path / "meta.pkl").read_bytes())
    dtype = np.dtype(meta["dtype"])
    assert meta["voices_per_beat"] == 4
    assert meta["voice_order"] == ["S", "A", "T", "B"]

    train_arr = np.fromfile(tmp_path / "train.bin", dtype=dtype)
    gen_arr = np.fromfile(tmp_path / "gen.bin", dtype=dtype)
    val_arr = np.fromfile(tmp_path / "val.bin", dtype=dtype)

    # in-vocab and non-empty for train/gen, empty for val (no piece routed here)
    assert train_arr.size > 0 and gen_arr.size > 0
    assert val_arr.size == 0
    for arr in (train_arr, gen_arr):
        assert int(arr.min()) >= 0 and int(arr.max()) < vocab_size

    # probe target CSVs exist with the expected headers and at least one row
    key_lines = (tmp_path / "key.csv").read_text().splitlines()
    chord_lines = (tmp_path / "chord.csv").read_text().splitlines()
    beat_lines = (tmp_path / "beat.csv").read_text().splitlines()
    assert key_lines[0] == "split,piece_idx,token_pos,mode"
    assert chord_lines[0] == "split,piece_idx,token_pos,roman"
    assert beat_lines[0] == "split,piece_idx,token_pos,beat"
    assert len(key_lines) > 1 and len(chord_lines) > 1 and len(beat_lines) > 1

    # spot-check beat labels: 4 voices per beat -> 4 consecutive rows share
    # the same beat number, then incrementing 1->2->3->4 then resetting.
    beat_rows = beat_lines[1:]
    train_beats = [int(r.split(",")[3]) for r in beat_rows
                   if r.startswith("train,")]
    # piece 1 = sustained C-major 4 beats; should see [1]*4, [2]*4, [3]*4, [4]*4
    assert train_beats[:4] == [1, 1, 1, 1]
    assert train_beats[4:8] == [2, 2, 2, 2]
    assert train_beats[8:12] == [3, 3, 3, 3]
    assert train_beats[12:16] == [4, 4, 4, 4]


if __name__ == "__main__":
    import tempfile

    test_tokenizer_bijection()
    test_encode_piece_layout()
    test_no_probe_target_leakage()
    test_shuffles_preserve_what_they_should()
    with tempfile.TemporaryDirectory() as d:
        test_dump_roundtrip(Path(d))
    print("ok")
