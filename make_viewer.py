#!/usr/bin/env python
"""Interactive 3D pages (three.js, self-contained) of traced single muon decays.

One HTML page per run plus index.html. Each page embeds a subsample of the FLUKA tracking
steps (with times, for the time slider), the entry points into the detector region and the
accelerator wireframe. Open index.html in a browser; no server or network needed.

    python make_viewer.py --traces <dir> [--out viewer] [run ...]

three.min.js and OrbitControls.js (r128, MIT licence) are copied from vendor/.
"""
import argparse
import base64
import json
import shutil
from pathlib import Path

import numpy as np

import plot_traces as pt
import mdi_geometry as geo
from read_det_ip import read

HERE = Path(__file__).resolve().parent
COLORS = {"e±": "#3FD28A", "Photons": "#5BA8FF", "Other": "#FFB52E", "Neutrons": "#FF4D9B"}
WIRE = {"dipole": ("Dipoles", "#B3A6FF", 0.9), "quadrupole": ("Quadrupoles", "#FF8A3D", 0.9), "nozzle": ("Nozzle", "#ECECF2", 0.8),
        "boron": ("Boron layer", "#3FD2F2", 0.7), "chamber": ("Beam chamber", "#C77DD8", 0.9), "tunnel": ("Tunnel wall", "#59607A", 0.6)}
T_END = 1e5      # ns, end of the time slider


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def scene(p):
    """(x, y, z) in cm -> scene (X = z, Y = y, Z = -x)."""
    return np.column_stack([p[:, 2], p[:, 1], -p[:, 0]])


def slider_u(t_ns, t_front):
    t = np.maximum(t_ns, 0.0)
    late = 0.6 + 0.4 * np.log(np.maximum(t, t_front) / t_front) / np.log(T_END / t_front)
    return np.where(t < t_front, 0.6 * t / t_front, np.minimum(late, 0.998)).astype(np.float32)


def page(run, prev, nxt, cap, out):
    m = np.memmap(pt.ROOT / run / "trace001_TRACKS.dat", dtype=pt.REC, mode="r")
    ent = read(str(pt.ROOT / run / "trace001_DET_IP.dat"))
    zd = float(m[0]["p0"][2])
    zlo, zhi, rmax = zd - 300, 1600, 130
    t_front = (abs(zd) + 800) / pt.C_CM_NS
    g, t, p0, p1 = pt.select3d(m, zlo, zhi, rmax, cap=cap)
    u = slider_u(t, t_front)
    groups = []
    for k in pt.LEGEND_ORDER:
        name = pt.GROUPS[k][0]
        sel = g == k
        pos = np.stack([scene(p0[sel]), scene(p1[sel])], axis=1).reshape(-1, 3)
        groups.append(dict(name=name, color=COLORS[name], n=int(sel.sum()), opacity=0.35 if name in ("e±", "Photons") else 0.6,
                           pos=b64(pos), u=b64(np.repeat(u[sel], 2))))
    wire = {}
    for kind, z, x, y in geo.wire3d(zlo, zhi, tunnel_step=500 if zhi - zlo < 6000 else 1000):
        pts = scene(np.column_stack([x, y, z]))
        wire.setdefault(kind, []).append(np.stack([pts[:-1], pts[1:]], axis=1).reshape(-1, 3))
    wires = [dict(name=WIRE[k][0], color=WIRE[k][1], opacity=WIRE[k][2], pos=b64(np.concatenate(v))) for k, v in wire.items()]
    epos = scene(np.column_stack([ent["x"], ent["y"], ent["z"]])) if len(ent) else np.zeros((0, 3))
    data = dict(experiment="MuColl FLUKA simulation, 10 TeV", zd=zd / 100, nsteps=int(len(m)), nshown=int(len(g)), nent=int(len(ent)),
                zlo=zlo, zhi=zhi, rmax=rmax, tfront=t_front, tend=T_END, groups=groups, wire=wires,
                entries=dict(n=int(len(ent)), pos=b64(epos)), prev=prev, next=nxt)
    html = (HERE / "viewer_template.html").read_text().replace("/*DATA*/", json.dumps(data))
    (out / f"{run}.html").write_text(html)
    return zd / 100, len(m), len(ent), len(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--traces", default=pt.ROOT, type=Path, help="directory with one folder per traced decay")
    ap.add_argument("--out", default=HERE / "viewer", type=Path)
    ap.add_argument("--three", default=HERE / "vendor", type=Path, help="directory holding three.min.js and OrbitControls.js")
    ap.add_argument("--retemplate", action="store_true", help="only re-apply viewer_template.html to the existing pages")
    ap.add_argument("--cap", type=int, default=50_000, help="steps drawn per species")
    a = ap.parse_args(); pt.ROOT, OUT = a.traces, a.out
    if a.retemplate:
        template = (HERE / "viewer_template.html").read_text()
        for f in sorted(OUT.glob("z*.html")):
            old = f.read_text()
            start = old.index("const D=") + 8
            data = old[start:old.index(";\nfunction f32", start)]
            f.write_text(template.replace("/*DATA*/", data))
            print(f"re-templated {f.name}")
        raise SystemExit
    (OUT / "lib").mkdir(parents=True, exist_ok=True)
    for f in ("three.min.js", "OrbitControls.js"):
        shutil.copy(a.three / f, OUT / "lib" / f)
    runs = a.runs or sorted(p.name for p in pt.ROOT.iterdir() if (p / "trace001_TRACKS.dat").exists())
    rows = []
    for i, run in enumerate(runs):
        zd, n, nent, size = page(run, f"{runs[i - 1]}.html" if i else None, f"{runs[i + 1]}.html" if i + 1 < len(runs) else None, a.cap, OUT)
        rows.append((zd, run, n, nent))
        print(f"{run}: decay at {zd:.1f} m, {n:,} steps, {nent} entering, page {size / 1e6:.1f} MB", flush=True)
    items = "".join(f'<li><a href="{run}.html">Decay at z = {zd:.1f} m</a> · {n:,} FLUKA steps · {nent} particles enter the detector region</li>'
                    for zd, run, n, nent in sorted(rows, reverse=True))
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Muon decay displays</title><style>'
        'body{background:#07080C;color:#ECECF2;font:15px/1.6 system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}'
        'a{color:#B3A6FF}li{margin:8px 0}p{color:#80869A}</style></head><body><h1>Single muon decays, MuColl 10 TeV</h1>'
        '<p>FLUKA tracking steps of one forced muon decay each, with the magnets, nozzle and tunnel of the MDI model.</p>'
        f'<ul>{items}</ul></body></html>')
