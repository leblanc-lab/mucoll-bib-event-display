#!/usr/bin/env python3
"""Read a DET_IP.dat BIB particle dump written by mgdraw_in_noz_area.f.

Record layout (Fortran stream access, no record markers):
  int32   JTRACK   FLUKA particle id
  int32   JMOTHE   mother particle id
  float64 Ekin     kinetic energy [GeV]
  float64 x, y, z  entry point into DET_LAT [cm]
  float64 cx, cy, cz  direction cosines
  float64 t        time wrt bunch crossing [s]
  float64 xd, yd, zd  muon decay point [cm]

Usage: read_det_ip.py DET_IP.dat [more.dat ...]
"""
import sys

import numpy as np

RECORD = np.dtype(
    [("jtrack", "<i4"), ("jmothe", "<i4")]
    + [(n, "<f8") for n in ("ekin", "x", "y", "z", "cx", "cy", "cz", "t", "xd", "yd", "zd")]
)


def read(path):
    return np.fromfile(path, dtype=RECORD)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        d = read(path)
        print(f"{path}: {len(d)} particles")
        if len(d):
            ids, counts = np.unique(d["jtrack"], return_counts=True)
            for i, c in sorted(zip(ids, counts), key=lambda t: -t[1])[:10]:
                print(f"  particle id {i:4d}: {c}")
            print(f"  Ekin: min {d['ekin'].min():.3e}  max {d['ekin'].max():.3e} GeV")
