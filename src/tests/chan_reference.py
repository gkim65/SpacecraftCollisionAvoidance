"""
chan_reference.py

Thin CLI wrapper around the original Python Chan (1997) implementation in the
sibling ProbofCollision repo, used by test_chan_crossvalidation.jl to compute
reference Pc values on identical inputs at run time (no hard-coded numbers).

Reads one JSON object from stdin:
    {"sc1": [6], "sc2": [6], "cov1": [6][6], "cov2": [6][6], "hbr": float}
Writes one JSON object to stdout:
    {"pc": float}

Locate the ProbofCollision source via PROBOFCOLLISION_SRC env var, falling back
to ../../ProbofCollision/src relative to this file.
"""

import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_default_src = os.path.normpath(
    os.path.join(_here, "..", "..", "..", "ProbofCollision", "src")
)
_src = os.environ.get("PROBOFCOLLISION_SRC", _default_src)
sys.path.insert(0, _src)

from collision.chan1997 import chan_pc  # noqa: E402


def main():
    req = json.load(sys.stdin)
    sc1 = np.asarray(req["sc1"], dtype=float)
    sc2 = np.asarray(req["sc2"], dtype=float)
    cov1 = np.asarray(req["cov1"], dtype=float)
    cov2 = np.asarray(req["cov2"], dtype=float)
    hbr = float(req["hbr"])
    pc = chan_pc(sc1, sc2, cov1, cov2, hbr)
    json.dump({"pc": float(pc)}, sys.stdout)


if __name__ == "__main__":
    main()
