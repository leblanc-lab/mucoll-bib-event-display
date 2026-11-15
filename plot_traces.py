#!/usr/bin/env python
"""Displays of every FLUKA tracking step of single muon decays, with the accelerator outline.

Input: one folder per decay under the traces directory, holding
  trace001_TRACKS.dat   one 36-byte record per track segment: FLUKA particle id (int32),
                        kinetic energy [GeV], time since the decay at the end of the step [s],
                        start x, y, z and end x, y, z [cm] (float32)
  trace001_DET_IP.dat   the particles entering the detector region (see read_det_ip.py)
See fluka/ for how these are produced.

Per run, in the output directory:
  trace_<run>_tunnel, trace_<run>_ip   all steps projected onto x-z (path length per pixel)
  trace_<run>_3d                        perspective view of a subsample of the steps
  trace_<run>_tunnel, _3d as .webm and .gif       the same, building up in time (needs ffmpeg)

    python plot_traces.py --traces <dir> [--out output] [--no-movies] [run ...]

The traces directory can also be given as the environment variable BIB_TRACES.
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import house_style as hs
from read_det_ip import read
import mdi_geometry as geo

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BIB_TRACES", HERE / "traces"))
OUT = HERE / "output"
EXPT = "MuColl FLUKA simulation, 10 TeV"
REC = np.dtype([("j", "<i4"), ("e", "<f4"), ("t", "<f4"), ("p0", "<f4", 3), ("p1", "<f4", 3)])
# drawn in this order, so the sparse species end up on top
GROUPS = [("e±", [3, 4], hs.CAT[7]), ("Photons", [7], hs.CAT[3]),
          ("Other", None, hs.CAT[2]), ("Neutrons", [8], hs.CAT[1])]
LEGEND_ORDER = [3, 1, 0, 2]
WIRE = {"dipole": hs.CAT[0], "quadrupole": hs.CAT[4], "nozzle": hs.INK, "boron": hs.CAT[6], "chamber": hs.CAT[5], "tunnel": "#B0B0B0"}
NZ, NX, CHUNK, MAXSUB = 900, 280, 4_000_000, 3000
C_CM_NS = 29.9792458


def group_index(j):
    g = np.full(len(j), 2)
    for k, (_, ids, _) in enumerate(GROUPS):
        if ids:
            g[np.isin(j, ids)] = k
    return g


def frame_times(zd):
    """Frame edges [ns]: linear while the shower front travels to the IP, then logarithmic."""
    t_front = (abs(zd) + 800) / C_CM_NS
    return np.r_[np.linspace(0, t_front, 61)[1:], np.geomspace(t_front, 1e5, 41)[1:]]


def rasterise(m, view, tedges=None):
    """Path length per pixel, shape (frames, groups, NX, NZ); one frame if tedges is None."""
    zlo, zhi, xlo, xhi = view
    nt, ng = (len(tedges) + 1 if tedges is not None else 1), len(GROUPS)
    img = np.zeros(nt * ng * NX * NZ, np.float32)
    pz, px = (zhi - zlo) / NZ, (xhi - xlo) / NX
    for s in range(0, len(m), CHUNK):
        c = m[s:s + CHUNK]
        p0, p1 = c["p0"].astype(np.float64), c["p1"].astype(np.float64)
        length = np.linalg.norm(p1 - p0, axis=1)
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        keep = (hi[:, 2] > zlo) & (lo[:, 2] < zhi) & (hi[:, 0] > xlo) & (lo[:, 0] < xhi) & (length > 0)
        if not keep.any():
            continue
        a, b, w, g = p0[keep], p1[keep], length[keep], group_index(c["j"][keep])
        tb = np.searchsorted(tedges, c["t"][keep] * 1e9) if tedges is not None else np.zeros(len(w), int)
        n = np.clip(np.ceil(np.maximum(np.abs(b[:, 2] - a[:, 2]) / pz, np.abs(b[:, 0] - a[:, 0]) / px)), 1, MAXSUB).astype(int)
        rep = np.repeat(np.arange(len(n)), n)
        f = (np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[rep]
        z = a[rep, 2] + f * (b[rep, 2] - a[rep, 2]); x = a[rep, 0] + f * (b[rep, 0] - a[rep, 0])
        iz, ix = np.floor((z - zlo) / pz).astype(int), np.floor((x - xlo) / px).astype(int)
        ok = (iz >= 0) & (iz < NZ) & (ix >= 0) & (ix < NX)
        flat = (((tb[rep] * ng + g[rep]) * NX + ix) * NZ + iz)[ok]
        u, inv = np.unique(flat, return_inverse=True)
        img[u] += np.bincount(inv, weights=(w[rep] / n[rep])[ok]).astype(np.float32)
    return img.reshape(nt, ng, NX, NZ)


def scale(img):
    """Per group (low, high) path length used for the colour depth."""
    return [(np.percentile(h[h > 0], 5), h.max()) if (h > 0).any() else None for h in img]


def compose(img, scales):
    rgb = np.ones((NX, NZ, 3))
    for h, sc, (_, _, col) in zip(img, scales, GROUPS):
        if sc is None:
            continue
        lo, hi = sc
        alpha = np.where(h > 0, 0.5 + 0.5 * np.clip(np.log(np.maximum(h, lo) / lo) / np.log(max(hi / lo, 1.0001)), 0, 1), 0.0)
        rgb = rgb * (1 - alpha[..., None]) + np.array(to_rgb(col)) * alpha[..., None]
    return rgb


def outline2d(ax, view):
    """Magnets, nozzle and tunnel wall in the x-z plane (z in m, x in cm)."""
    zlo, zhi, xlo, xhi = view
    for m in geo.magnets():
        z0, z1 = (m["z"] - m["half"]) / 100, (m["z"] + m["half"]) / 100
        if z1 < zlo / 100 or z0 > zhi / 100:
            continue
        col = WIRE[m["kind"]]
        ax.plot([z0, z1, z1, z0, z0], np.array([-1, -1, 1, 1, -1]) * m["r_yoke"] + m["x"], color=col, lw=0.8)
        for sgn in (-1, 1):
            ax.plot([z0, z1], [m["x"] + sgn * m["r_ap"]] * 2, color=col, lw=0.5, ls=":")
    noz = geo.nozzle_profile()
    for sz in (1, -1):
        for sx in (-1, 1):
            ax.plot(sz * noz["outer"][0] / 100, sx * noz["outer"][1], color=WIRE["nozzle"], lw=0.8)
            ax.plot(sz * noz["bore"][0] / 100, sx * noz["bore"][1], color=WIRE["nozzle"], lw=0.5)
            ax.plot(sz * noz["boron"][0] / 100, sx * noz["boron"][1], color=WIRE["boron"], lw=0.7)
            ax.plot([sz * noz["tip_z"] / 100] * 2, [sx * noz["bore"][1][-1], sx * noz["tip_r"]], color=WIRE["nozzle"], lw=0.8)
    cz = abs(noz["cham_z"]) / 100
    for sx in (-1, 1):
        ax.plot([-cz, cz], [sx * noz["chamber"]] * 2, color=WIRE["chamber"], lw=0.8)
    t = geo.TUNNEL
    for x in (t["x0"] - t["r"], t["x0"] + t["r"]):
        if xlo < x < xhi:
            ax.plot([max(zlo, t["z0"]) / 100, t["z1"] / 100], [x, x], color=WIRE["tunnel"], lw=1.0)


def legend_handles():
    h = [Patch(fc=GROUPS[i][2], ec="none", label=GROUPS[i][0]) for i in LEGEND_ORDER]
    return h + [Line2D([], [], color=WIRE["dipole"], lw=1.0, label="Dipoles"), Line2D([], [], color=WIRE["quadrupole"], lw=1.0, label="Quadrupoles")]


def panel2d(view, lines):
    zlo, zhi, xlo, xhi = view
    fig, (ax,) = hs._fig([hs.PANEL * 0.8], width=hs.PANEL * 1.5)
    im = ax.imshow(np.ones((NX, NZ, 3)), origin="lower", extent=[zlo / 100, zhi / 100, xlo, xhi], aspect="auto", interpolation="nearest")
    outline2d(ax, view)
    ax.set_xlim(zlo / 100, zhi / 100); ax.set_ylim(xlo, xhi + 0.95 * (xhi - xlo))
    ax.set_xlabel("z [m]"); ax.set_ylabel("x [cm]")
    hs.labels(ax, EXPT, lines)
    h = legend_handles(); hs.legend(ax, h, [x.get_label() for x in h])
    return fig, ax, im


def still2d(img, view, lines, name):
    fig, ax, im = panel2d(view, lines)
    im.set_data(compose(img, scale(img)))
    hs.save(fig, str(OUT / name))


def movie2d(frames, tedges, view, lines, name):
    cum = np.cumsum(frames, axis=0)
    scales = scale(cum[-1])
    fig, ax, im = panel2d(view, lines)
    clock = ax.annotate("", xy=(0, 1), xycoords="axes fraction", xytext=(10, -10 - hs.LABEL_SIZE * 1.45 - hs.SUBLABEL_SIZE * 1.4 * len(lines)),
                        textcoords="offset points", ha="left", va="top", fontsize=hs.SUBLABEL_SIZE, color=hs.INK)
    writer = FFMpegWriter(fps=12, codec="libvpx-vp9", bitrate=3000)
    with writer.saving(fig, str(OUT / f"{name}.webm"), dpi=150):
        for k in range(len(cum)):
            im.set_data(compose(cum[k], scales))
            clock.set_text(time_label(tedges[k]) if k < len(tedges) else "All steps")
            writer.grab_frame()
        for _ in range(18):
            writer.grab_frame()
    plt.close(fig)
    to_gif(name)


def to_gif(name):
    """GIF copy of a movie, for viewers that cannot play WebM."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(OUT / f"{name}.webm"), "-filter_complex",
                    "[0:v]fps=12,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(OUT / f"{name}.gif")], check=True)


def time_label(t_ns):
    return f"{t_ns:.0f} ns after the decay" if t_ns < 1000 else f"{t_ns / 1000:.3g} µs after the decay"


def select3d(m, zlo, zhi, rmax, cap=70_000, seed=1):
    """Subsample of the steps for line drawing: (group, t [ns], p0, p1) sorted by time."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(0, len(m), CHUNK):
        c = m[s:s + CHUNK]
        p0, p1 = c["p0"], c["p1"]
        length = np.linalg.norm(p1 - p0, axis=1)
        g = group_index(c["j"])
        inside = ((p0[:, 2] > zlo) & (p1[:, 2] > zlo) & (p0[:, 2] < zhi) & (p1[:, 2] < zhi)
                  & (np.abs(p0[:, :2]).max(1) < rmax) & (np.abs(p1[:, :2]).max(1) < rmax))
        keep = inside & (length > np.where(g >= 2, 0.3, 1.0))        # e+- and photons: steps over 1 cm only
        parts.append((g[keep], c["t"][keep] * 1e9, p0[keep].copy(), p1[keep].copy()))
    g, t, p0, p1 = (np.concatenate(x) for x in zip(*parts))
    sel = np.concatenate([rng.choice(i, min(len(i), cap), replace=False) for i in (np.flatnonzero(g == k) for k in range(len(GROUPS)))])
    sel = sel[np.argsort(t[sel])]
    return g[sel], t[sel], p0[sel], p1[sel]


def scene3d(zlo, zhi, rmax, lines):
    fig = plt.figure(figsize=(10.5, 5.2))
    ax = fig.add_axes([-0.02, -0.12, 1.04, 1.12], projection="3d")
    ax.set_box_aspect((4.2, 1, 1), zoom=1.22)
    ax.view_init(elev=16, azim=-66)
    for kind, z, x, y in geo.wire3d(zlo, zhi, tunnel_step=500 if zhi - zlo < 6000 else 2000):
        ok = (np.abs(x) <= rmax * 1.6) & (np.abs(y) <= rmax * 1.6)
        ax.plot(z / 100, np.where(ok, x, np.nan), np.where(ok, y, np.nan), color=WIRE[kind], lw=0.5 if kind != "tunnel" else 0.4,
                alpha=0.9 if kind != "tunnel" else 0.6)
    ax.plot([zlo / 100, zhi / 100], [0, 0], [0, 0], color=hs.INK, lw=0.4, ls="--")
    ax.set_xlim(zlo / 100, zhi / 100); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
    ax.set_xlabel("z [m]", labelpad=18); ax.set_ylabel("x [cm]"); ax.set_zlabel("y [cm]")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white"); axis.pane.set_edgecolor("#DDDDDD")
    ax.grid(False)
    ax.set_yticks([-100, 0, 100]); ax.set_zticks([-100, 0, 100])
    ax.text2D(0.03, 0.95, EXPT, transform=fig.transFigure, fontsize=hs.LABEL_SIZE, va="top")
    for k, line in enumerate(lines):
        ax.text2D(0.03, 0.95 - 0.045 * (k + 1.2), line, transform=fig.transFigure, fontsize=hs.SUBLABEL_SIZE, va="top")
    h = legend_handles()
    fig.legend(h, [x.get_label() for x in h], loc="upper right", bbox_to_anchor=(0.985, 0.97), frameon=False, fontsize=hs.LEGEND_SIZE, ncol=1)
    colls = []
    for _, _, col in GROUPS:
        lc = Line3DCollection(np.zeros((1, 2, 3)), colors=col, linewidths=0.35, alpha=0.55)
        ax.add_collection3d(lc); colls.append(lc)
    clock = ax.text2D(0.03, 0.95 - 0.045 * (len(lines) + 1.2), "", transform=fig.transFigure, fontsize=hs.SUBLABEL_SIZE, va="top")
    return fig, ax, colls, clock


def segs(p0, p1):
    a = np.column_stack([p0[:, 2] / 100, p0[:, 0], p0[:, 1]]); b = np.column_stack([p1[:, 2] / 100, p1[:, 0], p1[:, 1]])
    return np.stack([a, b], axis=1)


def display3d(m, zd, lines, name, movie, tedges):
    zlo, zhi, rmax = zd - 300, 1600, 130
    g, t, p0, p1 = select3d(m, zlo, zhi, rmax)
    s = segs(p0, p1)
    fig, ax, colls, clock = scene3d(zlo, zhi, rmax, lines)
    for k, lc in enumerate(colls):
        lc.set_segments(s[g == k] if (g == k).any() else np.zeros((1, 2, 3)))
    fig.savefig(str(OUT / f"{name}.png"), dpi=250); fig.savefig(str(OUT / f"{name}.pdf"))
    if movie:
        writer = FFMpegWriter(fps=12, codec="libvpx-vp9", bitrate=4000)
        with writer.saving(fig, str(OUT / f"{name}.webm"), dpi=120):
            for te in np.r_[tedges, np.inf]:
                upto = np.searchsorted(t, te)
                for k, lc in enumerate(colls):
                    part = s[:upto][g[:upto] == k]
                    lc.set_segments(part if len(part) else np.zeros((1, 2, 3)))
                clock.set_text(time_label(te) if np.isfinite(te) else "All steps")
                writer.grab_frame()
            for _ in range(18):
                writer.grab_frame()
        to_gif(name)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--traces", default=ROOT, type=Path, help="directory with one folder per traced decay")
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--no-movies", action="store_true")
    a = ap.parse_args(); ROOT, OUT = a.traces, a.out
    hs.apply(); OUT.mkdir(parents=True, exist_ok=True)
    runs = a.runs or sorted(p.name for p in ROOT.iterdir() if (p / "trace001_TRACKS.dat").exists())
    for run in runs:
        m = np.memmap(ROOT / run / "trace001_TRACKS.dat", dtype=REC, mode="r")
        ent = read(str(ROOT / run / "trace001_DET_IP.dat"))
        zd = float(m[0]["p0"][2])
        print(f"{run}: {len(m):,} steps, decay at z = {zd / 100:.2f} m, {len(ent)} particles enter the detector region", flush=True)
        head = [f"One muon decay at z = {zd / 100:.1f} m"]
        tedges = frame_times(zd)
        tunnel, ip = (zd - 300, 1600, -125, 125), (-650, 650, -70, 70)
        lines_t = head
        if a.no_movies:
            still2d(rasterise(m, tunnel)[0], tunnel, lines_t, f"trace_{run}_tunnel")
        else:
            frames = rasterise(m, tunnel, tedges)
            still2d(frames.sum(0), tunnel, lines_t, f"trace_{run}_tunnel")
            movie2d(frames, tedges, tunnel, lines_t, f"trace_{run}_tunnel")
            del frames
        still2d(rasterise(m, ip)[0], ip, head, f"trace_{run}_ip")
        display3d(m, zd, head, f"trace_{run}_3d", not a.no_movies, tedges)
