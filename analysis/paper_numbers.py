"""Recompute every number the paper reports from results/, and check it.

Each check prints the recomputed value next to the value printed in the paper
(at the paper's precision) and whether they agree. Exit status is non-zero if
any check disagrees, so this doubles as a regression test of the artifacts.

Also writes the appendix tables (A2 fits, A3 probe, A4 retrieval) as plain
text to results/paper_tables.txt.

Conventions (paper Sec. 2):
  * every cell is read at its own held-out retrieval optimum (``_epretr``),
    test release R6 only; 30 cells per arm = 10 rungs x 3 nested draws;
  * a rung's value is the mean over its 3 draws;
  * probes are mean Pearson r over the 12 features; retrieval is time-pool
    e->v top-1 (101-way within-task, 85-way cross-task);
  * "floor" is the untrained encoder of the same depth.

Where the paper derived a number from the PRINTED tables rather than from
the raw values, this script does the same and says so: the "gain" ratios and
the 701->1863 retrieval gains use the rung means as printed in Tables A3/A4
(4 decimals for probes, 3 for retrieval) over the printed floor/chance, and
the cross-task zero-shot retrieval row of Table A2 is fit to the 3-decimal
values of Table A4. That row sits within ~0.005 of chance, so rounding moves
its statistics; results/paper_tables.txt prints the unrounded fit beside it.

Usage:
    python analysis/paper_numbers.py
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results"
S_AXIS = [10, 20, 50, 100, 200, 400, 701, 1000, 1400, 1863]
LOW, HIGH = [10, 20, 50], [50, 100, 200, 400, 701, 1000, 1400, 1863]
CHANCE = {"within": 1 / 101, "cross": 1 / 85}

ARMS = {  # key -> (label, retrieval selection, probe selection, depth)
    "warm": ("pretrained, depth 22", "addval_selection_retr.json",
             "addval_selection_probe.json", 22),
    "scratch22": ("random-init, depth 22", "fromscratch_selection_retr.json",
                  "fromscratch_optimum_selection.json", 22),
    "scratch12": ("random-init, depth 12", "d12av_selection_retr.json",
                  "d12av_selection_probe.json", 12),
}
CURVES = {"warm": ["addval_epoch_curve.json"],
          "scratch22": ["fromscratch_epoch_curve.json", "fromscratch800_epoch_curve.json"],
          "scratch12": ["d12av_epoch_curve.json"]}
READOUTS = {  # key -> (file prefix, kind, task)
    "within probe": ("e04_tt_test", "probe", "within"),
    "within retrieval": ("e04_retr_test", "retr", "within"),
    "cross probe": ("xtask_tt_DMtest", "probe", "cross"),
    "cross retrieval": ("xtask_retr_DMtest", "retr", "cross"),
}
FLOOR_FILE = {
    ("within probe", 22): "e04_tt_test_random.json",
    ("within probe", 12): "e03_tt_test_random.json",
    ("within retrieval", 22): "e04_retr_test_random.json",
    ("within retrieval", 12): "e03_retr_test_random.json",
    ("cross probe", 22): "xtask_tt_DMtest_random_e04.json",
    ("cross probe", 12): "xtask_tt_DMtest_random.json",
    ("cross retrieval", 22): "xtask_retr_DMtest_random_e04.json",
    ("cross retrieval", 12): "xtask_retr_DMtest_random.json",
}


# ---------------------------------------------------------------- loading ---
def load(name):
    return json.loads((RAW / name).read_text())


def read(path: Path, kind: str) -> float:
    d = json.loads(path.read_text())
    if kind == "probe":
        f = d["features"]
        return sum(v["pearson_r"] for v in f.values()) / len(f)
    return d["levels"]["time"]["e2v_top_k"]["1"]


def s_of(cell: str) -> int:
    return int(cell.split("_s", 1)[1].split("_", 1)[0])


def draw_of(cell: str) -> int:
    return int(cell.rsplit("_d", 1)[1])


def cells(arm: str, which: str = "retr") -> dict[str, dict]:
    return load(ARMS[arm][1] if which == "retr" else ARMS[arm][2])


def values(arm: str, readout: str, suffix: str = "_epretr", which: str = "retr"):
    """{S: [value per draw]} at the selected checkpoints."""
    prefix, kind, _ = READOUTS[readout]
    out: dict[int, list[float]] = {}
    for c in cells(arm, which):
        out.setdefault(s_of(c), []).append(read(RAW / f"{prefix}_{c}{suffix}.json", kind))
    return out


def rung_means(arm, readout, **kw):
    v = values(arm, readout, **kw)
    return {S: st.mean(v[S]) for S in S_AXIS}


def printed(arm, readout):
    """Rung means at the precision Tables A3/A4 print them."""
    nd = 4 if READOUTS[readout][1] == "probe" else 3
    return {S: round(v, nd) for S, v in rung_means(arm, readout).items()}


def floor(readout: str, depth: int) -> float:
    return read(RAW / FLOOR_FILE[(readout, depth)], READOUTS[readout][1])


def gain(arm: str, readout: str) -> float:
    """App. D 'gain' at S=1863: probes over the untrained floor of the same
    depth, retrieval over chance -- from the printed values (see header)."""
    _, kind, task = READOUTS[readout]
    den = floor(readout, ARMS[arm][3]) if kind == "probe" else CHANCE[task]
    return printed(arm, readout)[1863] / round(den, 4)


def load_curves(arm: str) -> dict[str, dict]:
    """Per-checkpoint held-out curves; depth-22 random-init is the composite."""
    if arm != "scratch22":
        return load(CURVES[arm][0])["cells"]
    orig, ep800 = (load(n)["cells"] for n in CURVES[arm])
    out = {c: v for c, v in orig.items() if s_of(c) < 701}
    out.update({c: v for c, v in ep800.items() if s_of(c) >= 701})
    return out


# -------------------------------------------------------------- statistics --
def ols(xs, ys):
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return b, a, 1 - ss_res / ss_tot


def loglin(means: dict[int, float], s_values):
    return ols([math.log2(s) for s in s_values], [means[s] for s in s_values])


def sign_p(k: int, n: int = 9) -> float:
    """One-sided P(X >= k), X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


# ---------------------------------------------------------------- checking --
RESULTS: list[tuple[str, str, str, bool]] = []


def check(label: str, value, paper, fmt: str = "{:.4f}", ok: bool | None = None):
    """Compare at the paper's printed precision unless `ok` is given."""
    shown = fmt.format(value) if not isinstance(value, str) else value
    if ok is None:
        ok = shown == (fmt.format(paper) if not isinstance(paper, str) else paper)
    RESULTS.append((label, shown, paper if isinstance(paper, str) else fmt.format(paper), ok))


def fit_table(out):
    out.append("Table A2 -- log-linear fits over S = 10..1863 (rung means, test split)")
    hdr = f"{'readout':<34}{'slope':>9}{'R2':>7}{'gain':>8}{'up/9':>6}{'p':>8}{'10->50':>9}{'50->1863':>10}{'R2':>7}"
    fits = {}

    def row(m):
        b, _, r2 = loglin(m, S_AXIS)
        up = sum(m[S_AXIS[i + 1]] > m[S_AXIS[i]] for i in range(9))
        b_hi, _, r2_hi = loglin(m, HIGH)
        return dict(slope=b, r2=r2, up=up, p=sign_p(up), lo=loglin(m, LOW)[0],
                    hi=b_hi, r2_hi=r2_hi)

    def fmt(name, f, g):
        return (f"  {name:<34}{f['slope']:>+9.4f}{f['r2']:>7.3f}{g:>7.2f}x{f['up']:>4}/9"
                f"{f['p']:>8.4f}{f['lo']:>+9.4f}{f['hi']:>+10.4f}{f['r2_hi']:>7.3f}")

    for arm, (label, *_rest) in ARMS.items():
        out.append(f"\n  {label}\n  {hdr}")
        for ro in READOUTS:
            g = gain(arm, ro)
            if ro == "cross retrieval":
                f = row(printed(arm, ro))       # as printed in the paper
                raw = row(rung_means(arm, ro))
                out.append(fmt("cross zero-shot retrieval", f, g))
                out.append(fmt("  (same, unrounded rung means)", raw, g))
            else:
                f = row(rung_means(arm, ro))
                out.append(fmt(ro, f, g))
            fits[(arm, ro)] = dict(f, gain=g)
    return fits


def depth_tables(out):
    tabs = {}
    for ro in READOUTS:
        tabs[ro] = {a: rung_means(a, ro) for a in ARMS}
    for title, ro_w, ro_c, fmt in (
            ("Table A3 -- probe mean(12) Pearson r", "within probe", "cross probe", "{:.4f}"),
            ("Table A4 -- time-pool e->v top-1", "within retrieval", "cross retrieval", "{:.3f}")):
        out.append(f"\n{title}  (d12r | d22r | d22p | Dd=d22r-d12r | Di=d22p-d22r)")
        out.append(f"  {'S':>5}   {'TP test':<44}{'DM test'}")
        for S in S_AXIS:
            row = []
            for ro in (ro_w, ro_c):
                d12, d22, p22 = (tabs[ro][a][S] for a in ("scratch12", "scratch22", "warm"))
                row.append(" ".join(fmt.format(v) for v in (d12, d22, p22))
                           + f" {d22 - d12:+.4f} {p22 - d22:+.4f}")
            out.append(f"  {S:>5}   {row[0]:<44}{row[1]}")
    return tabs


def main() -> int:
    out: list[str] = []
    fits = fit_table(out)
    tabs = depth_tables(out)
    (RAW / "paper_tables.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    # ---- Table A2, every cell, against the paper -------------------------
    paper_a2 = {
        ("warm", "within probe"): (0.0244, 0.935, 2.91, 8, 0.0195, 0.0027, 0.0321, 0.995),
        ("warm", "within retrieval"): (0.0093, 0.899, 7.98, 9, 0.0020, 0.0017, 0.0126, 0.970),
        ("warm", "cross probe"): (0.0214, 0.962, 2.92, 8, 0.0195, 0.0042, 0.0265, 0.997),
        ("warm", "cross retrieval"): (0.0006, 0.853, 1.36, 5, 0.5000, -0.0000, 0.0008, 0.853),
        ("scratch22", "within probe"): (0.0146, 0.923, 2.04, 8, 0.0195, 0.0017, 0.0191, 0.983),
        ("scratch22", "within retrieval"): (0.0055, 0.866, 5.05, 8, 0.0195, 0.0015, 0.0074, 0.913),
        ("scratch22", "cross probe"): (0.0138, 0.967, 2.07, 9, 0.0020, 0.0063, 0.0167, 0.992),
        ("scratch22", "cross retrieval"): (0.0002, 0.265, 1.10, 3, 0.9102, -0.0001, 0.0003, 0.409),
        ("scratch12", "within probe"): (0.0162, 0.946, 2.17, 9, 0.0020, 0.0049, 0.0207, 0.991),
        ("scratch12", "within retrieval"): (0.0078, 0.891, 6.46, 8, 0.0195, 0.0016, 0.0105, 0.940),
        ("scratch12", "cross probe"): (0.0120, 0.979, 1.82, 9, 0.0020, 0.0068, 0.0139, 0.989),
        ("scratch12", "cross retrieval"): (0.0003, 0.426, 1.27, 5, 0.5000, -0.0000, 0.0005, 0.589),
    }
    for (arm, ro), p in paper_a2.items():
        f = fits[(arm, ro)]
        tag = f"A2 {arm}/{ro}"
        check(f"{tag} slope", f["slope"], p[0], "{:+.4f}")
        check(f"{tag} R2", f["r2"], p[1], "{:.3f}")
        check(f"{tag} gain", f["gain"], p[2], "{:.2f}")
        check(f"{tag} up/9", f["up"], p[3], "{:d}")
        check(f"{tag} sign p", f["p"], p[4], "{:.4f}")
        check(f"{tag} slope 10->50", f["lo"], p[5], "{:+.4f}")
        check(f"{tag} slope 50->1863", f["hi"], p[6], "{:+.4f}")
        check(f"{tag} R2 50->1863", f["r2_hi"], p[7], "{:.3f}")

    # ---- Tables A3 / A4 ---------------------------------------------------
    paper_a3 = {  # S: (TP d12r, d22r, d22p, DM d12r, d22r, d22p)
        10: (0.1102, 0.1189, 0.1341, 0.0929, 0.0895, 0.1134),
        20: (0.1150, 0.1146, 0.1421, 0.1026, 0.0964, 0.1289),
        50: (0.1215, 0.1224, 0.1408, 0.1091, 0.1042, 0.1240),
        100: (0.1317, 0.1316, 0.1691, 0.1212, 0.1163, 0.1581),
        200: (0.1540, 0.1477, 0.1956, 0.1361, 0.1280, 0.1839),
        400: (0.1742, 0.1730, 0.2282, 0.1507, 0.1529, 0.2114),
        701: (0.1933, 0.1954, 0.2539, 0.1553, 0.1635, 0.2301),
        1000: (0.2068, 0.2021, 0.2736, 0.1693, 0.1740, 0.2425),
        1400: (0.2191, 0.2075, 0.2976, 0.1785, 0.1823, 0.2567),
        1863: (0.2220, 0.2136, 0.3057, 0.1798, 0.1875, 0.2646),
    }
    paper_a4 = {
        10: (0.014, 0.014, 0.014, 0.012, 0.012, 0.012),
        20: (0.014, 0.014, 0.014, 0.013, 0.014, 0.013),
        50: (0.018, 0.017, 0.018, 0.012, 0.012, 0.012),
        100: (0.023, 0.018, 0.021, 0.013, 0.013, 0.013),
        200: (0.028, 0.021, 0.030, 0.011, 0.013, 0.015),
        400: (0.036, 0.032, 0.044, 0.013, 0.013, 0.015),
        701: (0.055, 0.046, 0.058, 0.013, 0.015, 0.016),
        1000: (0.063, 0.048, 0.066, 0.014, 0.014, 0.017),
        1400: (0.067, 0.046, 0.074, 0.014, 0.014, 0.016),
        1863: (0.064, 0.050, 0.079, 0.015, 0.013, 0.016),
    }
    order = ("scratch12", "scratch22", "warm")
    for paper, (ro_w, ro_c), fmt, tab in (
            (paper_a3, ("within probe", "cross probe"), "{:.4f}", "A3"),
            (paper_a4, ("within retrieval", "cross retrieval"), "{:.3f}", "A4")):
        for S, row in paper.items():
            got = [tabs[ro_w][a][S] for a in order] + [tabs[ro_c][a][S] for a in order]
            for name, g, p in zip(("TP d12r", "TP d22r", "TP d22p", "DM d12r", "DM d22r", "DM d22p"), got, row):
                check(f"{tab} S={S} {name}", g, p, fmt)
    fl = {(ro, d): floor(ro, d) for ro in ("within probe", "cross probe") for d in (12, 22)}
    check("A3 untrained floor TP d12", fl[("within probe", 12)], 0.1022)
    check("A3 untrained floor TP d22", fl[("within probe", 22)], 0.1049)
    check("A3 untrained floor DM d12", fl[("cross probe", 12)], 0.0989)
    check("A3 untrained floor DM d22", fl[("cross probe", 22)], 0.0905)
    check("A4 cross-task random encoder", floor("cross retrieval", 22), 0.012, "{:.3f}")

    # ---- Section 3.1 ------------------------------------------------------
    wp, wr = rung_means("warm", "within probe"), rung_means("warm", "within retrieval")
    check("3.1 pretrained top-rung probe r", wp[1863], 0.306, "{:.3f}")
    check("3.1 pretrained top-rung retrieval x chance", wr[1863] / CHANCE["within"], 8.0, "{:.1f}")
    for name, m in (("probe", wp), ("retrieval", wr)):
        b_all = loglin(m, S_AXIS)[0]
        b9, a9, _ = loglin(m, S_AXIS[:-1])
        check(f"3.1 drop S=1863: |slope change| {name} <= 0.001",
              abs(b_all - b9), "<=0.0010", "{:.4f}", ok=abs(b_all - b9) <= 0.001)
        pred = a9 + b9 * math.log2(1863)
        check(f"3.1 S=1863 above 9-rung trend ({name})", m[1863] - pred, "> 0",
              "{:+.4f}", ok=m[1863] > pred)
    check("3.1 untrained within-task probe floor", fl[("within probe", 22)], 0.105, "{:.3f}")
    lo_probe = max(rung_means(a, "within probe")[S] / floor("within probe", ARMS[a][3])
                   for a in ARMS for S in LOW)
    lo_retr = max(rung_means(a, "within retrieval")[S] / CHANCE["within"] for a in ARMS for S in LOW)
    check("3.1 S<=50 max probe / floor < 1.4x", lo_probe, "<1.4", "{:.2f}", ok=lo_probe < 1.4)
    check("3.1 S<=50 max retrieval / chance < 2x", lo_retr, "<2", "{:.2f}", ok=lo_retr < 2)
    tp = [fits[(a, "within probe")]["gain"] for a in ARMS]
    tr = [fits[(a, "within retrieval")]["gain"] for a in ARMS]
    # The text quotes these ranges to one decimal (2.04-2.91 -> "2.0-2.9",
    # 5.05-7.98 -> "5.0-8.0"); compare the endpoints to within 0.06.
    for name, g, lo, hi in (("probe", tp, 2.0, 2.9), ("retrieval", tr, 5.0, 8.0)):
        check(f"3.1 S=1863 {name} gain range", f"{min(g):.2f}-{max(g):.2f}", f"{lo:.1f}-{hi:.1f}",
              ok=abs(min(g) - lo) < 0.06 and abs(max(g) - hi) < 0.06)
    zs = max(abs(v - round(floor("cross retrieval", ARMS[a][3]), 3))
             for a in ARMS for v in printed(a, "cross retrieval").values())
    check("3.1 zero-shot cross retrieval within 0.005 of floor", zs, "<=0.005", "{:.4f}",
          ok=zs <= 0.005 + 1e-9)
    check("3.1 cross-task probe gain (pretrained)", fits[("warm", "cross probe")]["gain"], 2.92, "{:.2f}")
    # Typical (median) between-draw sd over every within-task cell of every arm.
    sds_p = [st.stdev(v) for a in ARMS for v in values(a, "within probe").values()]
    sds_r = [st.stdev(v) for a in ARMS for v in values(a, "within retrieval").values()]
    check("Fig.1 between-draw sd, probe (~0.003)", st.median(sds_p), 0.003, "{:.3f}")
    check("Fig.1 between-draw sd, retrieval (~0.002)", st.median(sds_r), 0.002, "{:.3f}")

    # ---- Section 3.2 ------------------------------------------------------
    ratios = [fits[("warm", "within probe")]["slope"] / fits[(a, "within probe")]["slope"]
              for a in ("scratch22", "scratch12")]
    check("3.2 pretrained/random probe slope ratio", f"{min(ratios):.1f}-{max(ratios):.1f}", "1.5-1.7")
    d701 = {a: printed(a, "within retrieval")[1863] - printed(a, "within retrieval")[701]
            for a in ARMS}
    check("3.2 retrieval gain 701->1863, random d22", d701["scratch22"], 0.004, "{:+.3f}")
    check("3.2 retrieval gain 701->1863, pretrained", d701["warm"], 0.021, "{:+.3f}")
    m12 = rung_means("scratch12", "within retrieval")
    check("3.2 depth-12 retrieval peaks before the top rung", max(m12, key=m12.get), "1400", "{}")
    for ro, paper_mean, paper_wins, label in (
            ("within probe", 0.008, 9, "within probe"),
            ("within retrieval", 0.017, 9, "within retrieval"),
            ("cross probe", -0.005, 1, "cross probe")):
        c12 = {(s_of(c), draw_of(c)): c for c in cells("scratch12")}
        c22 = {(s_of(c), draw_of(c)): c for c in cells("scratch22")}
        diffs = []
        prefix, kind, _ = READOUTS[ro]
        for key in sorted(c12):
            if key[0] >= 1000:
                a = read(RAW / f"{prefix}_{c12[key]}_epretr.json", kind)
                b = read(RAW / f"{prefix}_{c22[key]}_epretr.json", kind)
                diffs.append(a - b)
        check(f"3.2 paired d12-d22, S>=1000, {label}: mean", st.mean(diffs), paper_mean, "{:+.3f}")
        check(f"3.2 paired d12-d22, S>=1000, {label}: d12 wins", sum(d > 0 for d in diffs),
              paper_wins, "{:d}")
    gap = [tabs["within probe"]["warm"][S] - tabs["within probe"]["scratch22"][S] for S in S_AXIS]
    check("C.2 initialization gap range (TP probe)", f"{min(gap):.3f}-{max(gap):.3f}", "0.015-0.092")
    wins = sum(tabs[ro]["warm"][S] > tabs[ro][a][S]
               for ro in ("within probe", "cross probe") for a in ("scratch22", "scratch12")
               for S in S_AXIS)
    check("E pretrained wins probe cells (of 40)", wins, 40, "{:d}")
    di_r = {S: tabs["within retrieval"]["warm"][S] - tabs["within retrieval"]["scratch22"][S]
            for S in S_AXIS}
    check("E retrieval init advantage at S=1000", di_r[1000], 0.018, "{:+.3f}")
    check("E retrieval init advantage at S=1863", di_r[1863], 0.029, "{:+.3f}")
    zs_adv = max(tabs["cross retrieval"]["warm"][S] - tabs["cross retrieval"]["scratch22"][S]
                 for S in S_AXIS)
    check("E zero-shot init advantage max", zs_adv, "<=+0.003", "{:+.3f}", ok=zs_adv <= 0.0035)
    di_c = [tabs["cross probe"]["warm"][S] - tabs["cross probe"]["scratch22"][S] for S in S_AXIS]
    check("E cross probe init advantage range", f"{min(di_c):+.4f} to {max(di_c):+.4f}",
          "+0.0199 to +0.0771")

    # ---- Section 3.3 / App. C ----------------------------------------------
    flops = load("train_flops.json")
    fpe = {22: flops["depth22"]["flops_per_epoch"], 12: flops["depth12"]["flops_per_epoch"]}
    check("C.3 FLOPs/step depth 22 (T)", flops["depth22"]["flops_per_step"] / 1e12, 7.43, "{:.2f}")
    check("C.3 FLOPs/step depth 12 (T)", flops["depth12"]["flops_per_step"] / 1e12, 1.96, "{:.2f}")
    check("C.3 FLOPs/epoch depth 22 (T)", fpe[22] / 1e12, 81.7, "{:.1f}")
    check("C.3 FLOPs/epoch depth 12 (T)", fpe[12] / 1e12, 21.5, "{:.1f}")
    check("C.3 depth-22 / depth-12 step cost", flops["depth22"]["flops_per_step"]
          / flops["depth12"]["flops_per_step"], 3.80, "{:.2f}")
    check("C.3 tokens per window d22", flops["depth22"]["tokens_per_sample"], 258, "{:d}")
    check("C.3 tokens per window d12", flops["depth12"]["tokens_per_sample"], 129, "{:d}")

    def med_epoch(arm, S):
        return st.median(v["selected_epoch"] for c, v in cells(arm).items() if s_of(c) == S)
    check("3.3 random-init median optimum, S=10", med_epoch("scratch22", 10), 125, "{:.0f}")
    check("3.3 random-init median optimum, S=1863", med_epoch("scratch22", 1863), 775, "{:.0f}")
    check("3.3 pretrained median optimum, S=1863", med_epoch("warm", 1863), 325, "{:.0f}")
    top22 = [v["selected_epoch"] for c, v in cells("scratch22").items() if s_of(c) == 1863]
    check("3.3 random d22 S=1863 draws at final checkpoint", sum(e == 775 for e in top22), 3, "{:d}")
    check("3.3 PFLOPs to optimum, random d22", med_epoch("scratch22", 1863) * fpe[22] / 1e15, 63.3, "{:.1f}")
    check("3.3 PFLOPs to optimum, pretrained", med_epoch("warm", 1863) * fpe[22] / 1e15, 26.6, "{:.1f}")
    check("3.3 PFLOPs to optimum, random d12", med_epoch("scratch12", 1863) * fpe[12] / 1e15, 15.6, "{:.1f}")

    def last_ckpt(arm):
        n = 0
        for c, v in cells(arm).items():
            last = 775 if "_ep800" in c else 375
            n += v["selected_epoch"] == last
        return n
    for arm, paper in (("warm", 6), ("scratch22", 5), ("scratch12", 6)):
        check(f"C.1 cells selecting their last checkpoint, {arm} (of 30)", last_ckpt(arm), paper, "{:d}")

    # Early stopping with patience = 10 % of the checkpoint count.
    def early_stop(curve, patience):
        eps = sorted(int(e) for e in curve)
        best_v, best_e, bad = float("-inf"), eps[0], 0
        for e in eps:
            v = curve[str(e)]["e2v1_time"]
            if v > best_v:
                best_v, best_e, bad = v, e, 0
            else:
                bad += 1
                if bad >= patience:
                    return best_e, best_v, e
        return best_e, best_v, eps[-1]

    for arm, patience, paper in (("warm", 2, 12), ("scratch22", 3, 8), ("scratch12", 3, 8)):
        curves = {c: v for c, v in load_curves(arm).items() if s_of(c) >= 400}
        agree, costs, bad_exits = 0, [], []
        for c, cv in curves.items():
            eps = sorted(int(e) for e in cv)
            am = max(eps, key=lambda e: cv[str(e)]["e2v1_time"])
            dep, val, _ = early_stop(cv, patience)
            if dep == am:
                agree += 1
            else:
                costs.append(cv[str(am)]["e2v1_time"] - val)
                bad_exits.append((dep, am, cv[str(am)]["e2v1_time"] - val))
        check(f"C.3 early stopping agrees with argmax, {arm} (of {len(curves)})", agree, paper, "{:d}")
        if arm == "warm":
            check("C.3 pretrained early-stop misses cost", f"{min(costs):.4f}-{max(costs):.4f}",
                  "0.0028-0.0249")
        if arm == "scratch22":
            e200 = [(am, cost) for dep, am, cost in bad_exits if dep == 200]
            check("C.3 random d22 exits at epoch 200: optima",
                  "/".join(str(a) for a in sorted(a for a, _ in e200)), "625/675")
            check("C.3 random d22 exits at epoch 200: costs",
                  "/".join(f"{c:.4f}" for c in sorted(c for _, c in e200)), "0.0401/0.0410")

    # Second selector (probe-selected epochs, `_epprb`) vs the retrieval selector.
    for ro_group, paper_mean, paper_max in (
            (("within probe", "cross probe"), "0.0016-0.0035", 0.0125),
            (("within retrieval",), "0.0016-0.0034", 0.0087)):
        means, mx = [], 0.0
        for arm in ARMS:
            for ro in ro_group:
                a = rung_means(arm, ro, suffix="_epprb", which="probe")
                b = rung_means(arm, ro)
                d = [abs(a[S] - b[S]) for S in S_AXIS]
                means.append(st.mean(d))
                mx = max(mx, max(d))
        name = "probe" if "probe" in ro_group[0] else "retrieval"
        check(f"C.2 selector |delta| mean range ({name})", f"{min(means):.4f}-{max(means):.4f}", paper_mean)
        check(f"C.2 selector |delta| max ({name})", mx, paper_max, "{:.4f}")
    # A rung counts when the probe selector picks an earlier epoch on a
    # majority (>= 2) of its three draws.
    pr, rt = cells("warm", "probe"), cells("warm")
    earlier = sum(sum(pr[c]["selected_epoch"] < rt[c]["selected_epoch"]
                      for c in pr if s_of(c) == S) >= 2 for S in S_AXIS)
    check("C.2 probe selects earlier (pretrained rungs, of 10)", earlier, 7, "{:d}")

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'check':<66}{'recomputed':>16}{'paper':>16}")
    for label, got, paper, ok in RESULTS:
        print(f"{'  ' if ok else '! '}{label:<64}{got:>16}{paper:>16}")
    bad = [r for r in RESULTS if not r[3]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks agree with the paper.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
