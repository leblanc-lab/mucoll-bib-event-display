"""Outline of the accelerator elements of the MuColl MDI model, for overlaying on displays.

The outline is stored in geometry/mdi_v0p8.json, so the displays do not need the FLUKA
input. It was extracted from the lattice v0.8 input (lattice_v_0p8_EU_BIB_.inp): element
centres from the ROT-DEFI cards, lengths from the lattice container bodies, aperture and
yoke radii from the magnet prototypes, the nozzle from its cone bodies. To extract it
again from an input:

    python mdi_geometry.py path/to/lattice_v_0p8_EU_BIB_.inp [output.json]

Set BIB_GEOMETRY to use a different JSON file. Lengths in cm, in the IP frame (beam
along +z, IP at the origin).
"""
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

GEOMETRY = Path(os.environ.get("BIB_GEOMETRY", Path(__file__).resolve().parent / "geometry/mdi_v0p8.json"))
# lattice element -> magnet prototype (matched by name and container length)
PROTO = {"IB201": "IB2", "IB102": "IB1", "IB203": "IB2", "IQF204": "IF2", "IQF2A05": "F2a", "IQD107": "ID1",
         "IQD1A09": "D1a", "IQF1B11": "IFB", "IQF1A13": "IFA", "IQF115": "IF1", "IQF1d17": "IF1",
         "IQF1Ad19": "IFA", "IQF1Bd21": "IFB"}
TUNNEL = dict(x0=-51.0, y0=5.0, r=150.0, z0=-24000.0, z1=-1300.0)        # inner wall, upstream of the IP hall
HALL = dict(x=1000.0, y=1000.0, z=1300.0)                                 # half sizes of the IP hall


def magnets_from_input(path):
    text = Path(path).read_text()
    define = {m.group(1): float(m.group(2)) for m in re.finditer(r"^#define +(\w+) +(\S+) *$", text, re.M)
              if re.match(r"[-+0-9.eE]+$", m.group(2))}
    out = []
    for name, proto in PROTO.items():
        rot = re.search(rf"^ROT-DEFI *,[^,]*,[^,]*,[^,]*,([^,]*),([^,]*),([^,]*), *{name} *$", text, re.M)
        rcc = re.search(rf"^RCC +{name} +(.*)$", text, re.M).group(1).split()
        yoke = float(re.search(rf"^ZCC +{proto}v8yk +\S+ +\S+ +(\S+)", text, re.M).group(1))
        out.append(dict(name=name, kind="dipole" if name.startswith("IB") else "quadrupole",
                        x=-float(rot.group(1)), z=-float(rot.group(3)), half=float(rcc[5]) / 2,
                        r_ap=define[f"{proto}v8_v"], r_yoke=yoke))
    return sorted(out, key=lambda m: m["z"])


def nozzle_from_input(path):
    """Upstream nozzle in (z, r), from the nozzle bodies: outer surface, beam bore, boron layer
    (closed polygon) and the beryllium chamber radius between the two tips."""
    text = Path(path).read_text()

    def cone(name):
        v = [float(x) for x in re.search(rf"^TRC +{name} +(.*)$", text, re.M).group(1).split()]
        return lambda z: v[6] + (z - v[2]) / v[5] * (v[7] - v[6])

    zcc = lambda name: float(re.search(rf"^ZCC +{name} +\S+ +\S+ +(\S+)", text, re.M).group(1))
    plane = lambda name: float(re.search(rf"^XYP +{name} +(\S+)", text, re.M).group(1))
    wo, wt, bpb, bp1, bp2 = (cone(n) for n in ("LWo1", "LWto", "LBPb", "LBPt1", "LBPt2"))
    bo, bi, floor = cone("LBoc1o"), cone("LBoc1i"), zcc("bor_n")
    z_n, z_t, z_far, z_tip = plane("pLn"), plane("pLt"), -595.0, -6.0
    zz = np.linspace(z_far, z_n, 400)
    has = bo(zz) > floor                                     # boron exists only outside r = bor_n
    inner = np.maximum(bi(zz[has]), floor)
    return dict(
        outer=(np.array([z_far, z_n, z_tip]), np.array([wo(z_far), wo(z_n), wt(z_tip)])),
        bore=(np.array([z_far, z_n, z_t, z_tip]), np.array([bpb(z_far), bpb(z_n), bp1(z_t), bp2(z_tip)])),
        boron=(np.r_[zz[has], zz[has][::-1], zz[has][0]], np.r_[bo(zz[has]), inner[::-1], bo(zz[has][0])]),
        chamber=zcc("cham_i"), tip_r=wt(z_tip), tip_z=z_tip, cham_z=z_tip - (zcc("cham_i") - wt(z_tip)) / ((wt(z_n) - wt(z_tip)) / (z_tip - z_n)))


@lru_cache(maxsize=None)
def _stored():
    return json.loads(GEOMETRY.read_text())


def magnets():
    """Lattice magnets, upstream first: name, kind, centre x and z, half length, aperture and yoke radius."""
    return _stored()["magnets"]


def nozzle_profile():
    """Upstream nozzle in (z, r); the downstream one is its mirror image."""
    n = _stored()["nozzle"]
    return {k: tuple(np.array(a) for a in v) if isinstance(v, list) else v for k, v in n.items()}


def extract(path, out):
    noz = nozzle_from_input(path)
    noz = {k: [a.tolist() for a in v] if isinstance(v, tuple) else float(v) for k, v in noz.items()}
    Path(out).write_text(json.dumps(dict(source=Path(path).name, units="cm", magnets=magnets_from_input(path), nozzle=noz), indent=1))


def ring(z, r, x0=0.0, y0=0.0, n=49):
    a = np.linspace(0, 2 * np.pi, n)
    return np.full(n, z), x0 + r * np.cos(a), y0 + r * np.sin(a)


def wire3d(zlo, zhi, tunnel_step=500.0):
    """Polylines (z, x, y) for a 3D wireframe between zlo and zhi: list of (kind, z, x, y)."""
    lines = []
    for m in magnets():
        z0, z1 = m["z"] - m["half"], m["z"] + m["half"]
        if z1 < zlo or z0 > zhi:
            continue
        for r in (m["r_yoke"], m["r_ap"]):
            for z in (z0, z1):
                lines.append((m["kind"], *ring(z, r, m["x"])))
        for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
            x, y = m["x"] + m["r_yoke"] * np.cos(a), m["r_yoke"] * np.sin(a)
            lines.append((m["kind"], np.array([z0, z1]), np.array([x, x]), np.array([y, y])))
    noz = nozzle_profile()
    for sgn in (1, -1):                                      # upstream, downstream
        if (sgn > 0 and zlo > -6) or (sgn < 0 and zhi < 6):
            continue
        for kind, key, nphi in (("nozzle", "outer", 12), ("nozzle", "bore", 4), ("boron", "boron", 6)):
            zz, rr = noz[key]
            for z, r in zip(zz[[0, -1]] if key != "boron" else zz[[0, len(zz) // 2 - 1]], rr[[0, -1]] if key != "boron" else rr[[0, len(zz) // 2 - 1]]):
                lines.append((kind, *ring(sgn * z, r)))
            if key == "boron":
                lines.append((kind, *ring(sgn * zz[len(zz) // 2], rr[len(zz) // 2])))
            if key == "outer":
                lines.append((kind, *ring(sgn * zz[1], rr[1])))
            for a in np.linspace(0, 2 * np.pi, nphi + 1)[:-1]:
                lines.append((kind, sgn * zz, rr * np.cos(a), rr * np.sin(a)))
    cz = abs(noz["cham_z"])
    for z in (-cz, cz):
        lines.append(("chamber", *ring(z, noz["chamber"])))
    for a in np.linspace(0, 2 * np.pi, 5)[:-1]:
        lines.append(("chamber", np.array([-cz, cz]), np.full(2, noz["chamber"] * np.cos(a)), np.full(2, noz["chamber"] * np.sin(a))))
    t = TUNNEL
    for z in np.arange(np.ceil(max(zlo, t["z0"]) / tunnel_step) * tunnel_step, min(zhi, t["z1"]) + 1, tunnel_step):
        lines.append(("tunnel", *ring(z, t["r"], t["x0"], t["y0"])))
    return lines


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else GEOMETRY)
