#!/usr/bin/env python3
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figures")
RECALL_LEVELS = (0.8, 0.9, 0.95)

STYLE = {
    "noop": ("#8a8984", "x", ":", "Unmodified index"),
    "oodbridge": ("#2a78d6", "o", "-", "Incremental modification (oodbridge)"),
    "comember-score": ("#2a78d6", "o", "-", "Workload-adaptive modification (comember)"),
    "comember-none": ("#1baf7a", "^", "-", "comember without edge removal"),
    "comember-score-B": ("#2a78d6", "o", "--", "comember, overlay edge budget"),
    "comember-none-B": ("#1baf7a", "^", "--", "comember without removal, overlay edge budget"),
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def aggregate(rows, rkey, qkey):
    by_l = defaultdict(list)
    for r in rows:
        if rkey in r and qkey in r:
            by_l[r["L"]].append((r[rkey], r[qkey]))
    return {L: (float(np.mean([x[0] for x in v])), float(np.exp(np.mean(np.log([x[1] for x in v])))), len(v))
            for L, v in sorted(by_l.items())}


def qps_at(agg, recall):
    pts = sorted((v[0], np.log(v[1])) for v in agg.values())
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if r0 <= recall <= r1 and r1 > r0:
            return float(np.exp(q0 + (q1 - q0) * (recall - r0) / (r1 - r0)))
    return None


def draw(ax, agg, label):
    c, mk, ls, name = STYLE[label]
    ax.plot([v[0] for v in agg.values()], [v[1] for v in agg.values()], color=c, marker=mk, linestyle=ls,
            linewidth=1.6, markersize=4.5, label=name)


def axes(ax, title):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (1 thread)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", color="#e4e2dc", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def summarize(agg):
    return {"n_seeds": min(v[2] for v in agg.values()),
            "qps_at_recall": {str(r): qps_at(agg, r) for r in RECALL_LEVELS},
            "curve": {str(L): {"recall": v[0], "qps": v[1]} for L, v in agg.items()}}


def incremental(summary):
    runs = defaultdict(lambda: defaultdict(list))
    for d in glob.glob(f"{RES}/curves/*/mm-clustered*-s*/*"):
        method = os.path.basename(d)
        if method not in STYLE:
            continue
        rbs = os.path.basename(os.path.dirname(d))
        ds = os.path.basename(os.path.dirname(os.path.dirname(d)))
        steps = [int(m.group(1)) for f in os.listdir(d) if (m := re.fullmatch(r"step(\d+)\.jsonl", f))]
        if steps:
            runs[(ds, rbs.rsplit("-s", 1)[0])][method].append(f"{d}/step{max(steps)}.jsonl")
    for (ds, rb), methods in sorted(runs.items()):
        groups = [("image", "existing-modality queries"), ("text", "inserted-modality queries")]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (g, title) in zip(axs, groups):
            for method, files in sorted(methods.items()):
                rows = [r for f in files for r in read_jsonl(f)]
                agg = aggregate(rows, f"recall_{g}", f"qps_{g}") or aggregate(rows, "recall", "qps")
                if not agg:
                    continue
                draw(ax, agg, method)
                summary.setdefault("incremental", {}).setdefault(f"{ds}/{rb}", {}).setdefault(g, {})[method] = summarize(agg)
            axes(ax, f"{ds} {rb}: {title}")
        axs[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{OUT}/incremental_{ds}_{rb}.png", dpi=200)
        plt.close(fig)


def adaptive(summary):
    runs = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f"{RES}/curves/*/*-s*/*.jsonl"):
        label = os.path.basename(f)[:-6]
        wls = os.path.basename(os.path.dirname(f))
        if label not in STYLE or label == "oodbridge" or wls.startswith("mm-clustered"):
            continue
        ds = os.path.basename(os.path.dirname(os.path.dirname(f)))
        runs[(ds, wls.rsplit("-s", 1)[0])][label].append(f)
    for (ds, wl), labels in sorted(runs.items()):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for label in STYLE:
            if label not in labels:
                continue
            agg = aggregate([r for f in labels[label] for r in read_jsonl(f)], "recall", "qps")
            draw(ax, agg, label)
            summary.setdefault("adaptive", {}).setdefault(f"{ds}/{wl}", {})[label] = summarize(agg)
        axes(ax, f"{ds} {wl}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/adaptive_{ds}_{wl}.png", dpi=200)
        plt.close(fig)


def crossmodal(summary):
    for dsdir in sorted(glob.glob(f"{RES}/crossmodal/*")):
        ds = os.path.basename(dsdir)
        table = {}
        for label in ("noop", "comember-score", "comember-none"):
            per_l = defaultdict(lambda: defaultdict(list))
            for f in glob.glob(f"{dsdir}/*/{label}.jsonl"):
                rounds = [r for r in read_jsonl(f) if r.get("type") == "round"]
                if not rounds:
                    continue
                last = max(rounds, key=lambda r: r["round"])
                for k, v in last.items():
                    m = re.fullmatch(r"(drop|recall)_L(\d+)(_single|_multi)?", k)
                    if m:
                        per_l[int(m.group(2))][m.group(1) + (m.group(3) or "")].append(v)
            if per_l:
                table[label] = {str(L): {k: float(np.mean(v)) for k, v in d.items()} | {"n_seeds": len(d["drop"])}
                                for L, d in sorted(per_l.items())}
        if table:
            summary.setdefault("crossmodal", {})[ds] = table
            print(f"\n{ds}: recall@10 drop (image queries - text queries), mean over seeds")
            print(f"{'L':>5} {'image recall':>13} " + " ".join(f"{x:>15}" for x in table))
            for L in next(iter(table.values())):
                single = table.get("noop", {}).get(L, {}).get("recall_single", float("nan"))
                print(f"{L:>5} {single:13.3f} " + " ".join(f"{t.get(L, {}).get('drop', float('nan')):15.3f}"
                                                         for t in table.values()))


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    incremental(summary)
    adaptive(summary)
    crossmodal(summary)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nfigures and summary.json in {OUT}")


if __name__ == "__main__":
    main()
