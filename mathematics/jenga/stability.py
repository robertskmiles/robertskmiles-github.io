"""Idealized Jenga statics.

Model: rigid, uniform, perfectly manufactured blocks; frictionless contacts
transmitting only non-negative vertical pressure; blocks sit in discrete slots.

Geometry (unit = one slot width): tower footprint is the 3x3 square [0,3]^2.
Layer k (0-indexed from the bottom):
  even k: blocks run along x, block in slot s occupies [0,3] x [s,s+1]
  odd  k: blocks run along y, block in slot s occupies [s,s+1] x [0,3]
All blocks have mass 1. Block height is irrelevant to statics (only the
horizontal centre of mass matters under vertical gravity).

A tower is a list of layers, bottom to top; a layer is a set of slots
drawn from {0,1,2} (must be non-empty: an empty layer with anything above
it has no contact at all and trivially collapses).

Two stability tests:

1. cut_test: for every horizontal interface (table/layer0 and each pair of
   consecutive layers), the centre of mass of everything above the cut must
   lie in the convex hull of the contact region at that cut. Because
   consecutive layers are perpendicular and blocks span the full footprint,
   every (lower slot, upper slot) pair produces a contact patch, so the hull
   is simply a rectangle: x-extent from the y-running layer's occupied slots,
   y-extent from the x-running layer's occupied slots. All coordinates are
   half-integers, so we do the test in doubled integer units -> exact,
   and marginal (centre of mass exactly on the hull boundary) is detected
   exactly, never a float question.

2. lp_test: ground truth for the frictionless rigid-body model. Stability =
   existence of an assignment of non-negative vertical point forces at
   contact-patch corners such that every individual block is in force and
   torque balance. (Allowing point forces at the corners of each rectangular
   patch is equivalent to allowing arbitrary non-negative pressure
   distributions over the patch.)

The cut test is a known necessary condition. It is NOT sufficient for
general block-stacking (cf. Paterson & Zwick's maximum-overhang analysis),
but for Jenga's constrained geometry the two tests agreed on every
configuration ever tested (exhaustive to 7 layers ~960k towers; 60k+
random/grown towers to 30 layers; zero disagreements, including all
marginal cases) - so the game engine uses the cut test alone.

Game conventions (decided 2026-08-28):
  - Knife-edge (MARGINAL) counts as stable: in the idealized frictionless
    world a balanced tower stays balanced. Use stable(), which is the
    closed test. Empirically MARGINAL always has a (boundary) equilibrium
    in the LP, so the closed cut test still matches the LP exactly.
  - Moves are non-atomic: a move is (removal, placement) and the
    intermediate state after removal must itself be stable, as must the
    final state after placement. A removal whose every placement collapses
    is never part of a legal move (choosing it would just be an immediate
    loss; a player with no legal move loses).
"""

from itertools import product, combinations
import numpy as np
from scipy.optimize import linprog

NONEMPTY = [frozenset(c) for r in (1, 2, 3) for c in combinations((0, 1, 2), r)]

FAIL, MARGINAL, STABLE = 0, 1, 2


def cut_test(layers):
    """Exact per-cut COM test.

    Returns (status, min_margin) where status is FAIL / MARGINAL / STABLE and
    min_margin is the smallest signed distance (in slot widths) from a free
    body's COM to any contact hull boundary (positive = inside).
    """
    n = len(layers)
    sx = sy = m = 0  # running suffix sums: doubled coords, block count
    status = STABLE
    min_margin = None
    for k in range(n - 1, -1, -1):
        for s in layers[k]:
            if k % 2 == 0:
                sx += 3
                sy += 2 * s + 1
            else:
                sx += 2 * s + 1
                sy += 3
            m += 1
        # Evaluate the cut directly below layer k (free body = layers k..top).
        if k > 0:
            lo, hi = layers[k - 1], layers[k]
            ey = lo if (k - 1) % 2 == 0 else hi  # x-running layer bounds y
            ex = hi if (k - 1) % 2 == 0 else lo  # y-running layer bounds x
            xlo, xhi = 2 * min(ex), 2 * (max(ex) + 1)
        else:
            # Table cut: contact = layer 0 footprint (x-running, spans all x).
            ey = layers[0]
            xlo, xhi = 0, 6
        ylo, yhi = 2 * min(ey), 2 * (max(ey) + 1)
        c = min(sy - ylo * m, yhi * m - sy, sx - xlo * m, xhi * m - sx)
        margin = c / (2 * m)
        if min_margin is None or margin < min_margin:
            min_margin = margin
        if c < 0:
            status = FAIL
        elif c == 0 and status == STABLE:
            status = MARGINAL
    return status, min_margin


def _blocks(layers):
    out = []
    for k, occ in enumerate(layers):
        for s in sorted(occ):
            if k % 2 == 0:
                rect = (0.0, 3.0, float(s), float(s + 1))
            else:
                rect = (float(s), float(s + 1), 0.0, 3.0)
            out.append((k, rect))
    return out


def lp_test(layers):
    """Rigid-body equilibrium feasibility (frictionless). True = stable."""
    blocks = _blocks(layers)
    by_layer = {}
    for i, (k, _) in enumerate(blocks):
        by_layer.setdefault(k, []).append(i)

    contacts = []  # (lower_block or -1 for table, upper_block, 4 corner points)
    for j in by_layer.get(0, []):
        x0, x1, y0, y1 = blocks[j][1]
        contacts.append((-1, j, [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]))
    for k in range(len(layers) - 1):
        for i in by_layer[k]:
            for j in by_layer[k + 1]:
                a, b = blocks[i][1], blocks[j][1]
                x0, x1 = max(a[0], b[0]), min(a[1], b[1])
                y0, y1 = max(a[2], b[2]), min(a[3], b[3])
                if x1 > x0 and y1 > y0:
                    contacts.append((i, j, [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]))

    nb, nv = len(blocks), 4 * len(contacts)
    A = np.zeros((3 * nb, nv))
    b_eq = np.zeros(3 * nb)
    for i, (_, (x0, x1, y0, y1)) in enumerate(blocks):
        b_eq[3 * i] = 1.0
        b_eq[3 * i + 1] = (x0 + x1) / 2
        b_eq[3 * i + 2] = (y0 + y1) / 2
    for c, (lo, up, corners) in enumerate(contacts):
        for t, (x, y) in enumerate(corners):
            v = 4 * c + t
            A[3 * up, v] += 1.0
            A[3 * up + 1, v] += x
            A[3 * up + 2, v] += y
            if lo >= 0:
                A[3 * lo, v] -= 1.0
                A[3 * lo + 1, v] -= x
                A[3 * lo + 2, v] -= y
    res = linprog(np.zeros(nv), A_eq=A, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.status == 0


def stable(layers):
    """Game stability test: closed convention (knife-edge counts as stable)."""
    return cut_test(layers)[0] >= MARGINAL


def all_towers(n):
    return product(NONEMPTY, repeat=n)


def fmt(layers):
    names = "LMR"
    return " | ".join("".join(names[s] for s in sorted(occ)) for occ in layers)
