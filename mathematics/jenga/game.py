"""The idealized Jenga game, on top of the model validated in stability.py.

State: tuple of layer bitmasks (bit s = slot s occupied), bottom to top;
layer k orientation = k % 2 (even layers run along x, occupying y-slots).

Rules:
  - A move = remove one block, then place it on top. The intermediate state
    (block in hand) and the final state must both be stable (closed
    convention: knife-edge counts as stable).
  - Removal allowed only from levels strictly below the highest completed
    (3-block) level: if the top layer is complete, from any level below it;
    otherwise from any level below top-1 (official Jenga rule).
  - Removing a layer's last block is never legal (empty layer = no contact).
  - Placement: fill any empty slot of the incomplete top layer; if the top
    is complete, start a new layer in any of the 3 slots.
  - No passing. A player with no legal move loses (they must topple the
    tower). Normal play: last player to move wins.

The game is finite: every move raises the sum of block heights, which is
bounded because layers stay non-empty. The move graph is acyclic, so
win/loss and Grundy values are well-defined by memoized DFS.

fast_stable() is a bitmask reimplementation of the exact integer cut test
from stability.py (closed convention); verify.py checks them against each
other exhaustively.
"""

import sys

sys.setrecursionlimit(1_000_000)

FULL = 7
CNT = [bin(m).count("1") for m in range(8)]
SY = [sum(2 * s + 1 for s in (0, 1, 2) if m >> s & 1) for m in range(8)]  # doubled
LO = [0] + [2 * min(s for s in (0, 1, 2) if m >> s & 1) for m in range(1, 8)]
HI = [0] + [2 * (max(s for s in (0, 1, 2) if m >> s & 1) + 1) for m in range(1, 8)]
FLIP = [0, 4, 2, 6, 1, 5, 3, 7]  # slot s -> 2-s
NAMES = ["", "L", "M", "LM", "R", "LR", "MR", "LMR"]


def full_tower(n):
    return (FULL,) * n


def fmt(state):
    return " | ".join(NAMES[m] for m in state)


def fast_stable(state):
    """Exact closed-convention cut test on a mask tuple. True = stable."""
    n = len(state)
    sx = sy = m = 0
    for k in range(n - 1, -1, -1):
        lay = state[k]
        c = CNT[lay]
        m += c
        if k % 2 == 0:
            sy += SY[lay]
            sx += 3 * c
        else:
            sx += SY[lay]
            sy += 3 * c
        # cut below layer k (free body = layers k..top)
        if k > 0:
            lo, hi = state[k - 1], state[k]
            ey = lo if (k - 1) % 2 == 0 else hi  # x-running layer bounds y
            ex = hi if (k - 1) % 2 == 0 else lo  # y-running layer bounds x
            if not (LO[ex] * m <= sx <= HI[ex] * m):
                return False
        else:
            ey = state[0]
        if not (LO[ey] * m <= sy <= HI[ey] * m):
            return False
    return True


def canon(state):
    """Least representative under the reflection group: y-reflection flips
    slots of even (x-running) layers, x-reflection flips odd layers."""
    c0 = state
    c1 = tuple(FLIP[l] if k % 2 == 0 else l for k, l in enumerate(state))
    c2 = tuple(FLIP[l] if k % 2 == 1 else l for k, l in enumerate(state))
    c3 = tuple(FLIP[l] for l in state)
    return min(c0, c1, c2, c3)


def legal_moves(state):
    """Return successor states (not canonicalized)."""
    top = len(state) - 1
    if state[top] == FULL:
        H = top
        fill = False
        slots = (1, 2, 4)
    else:
        H = top - 1
        fill = True
        slots = tuple(1 << s for s in (0, 1, 2) if not state[top] >> s & 1)
    out = []
    for k in range(H):
        lay = state[k]
        if CNT[lay] == 1:
            continue
        for s in (0, 1, 2):
            b = 1 << s
            if not lay & b:
                continue
            inter = state[:k] + (lay ^ b,) + state[k + 1:]
            if not fast_stable(inter):
                continue
            for p in slots:
                if fill:
                    nxt = inter[:top] + (inter[top] | p,)
                else:
                    nxt = inter + (p,)
                if fast_stable(nxt):
                    out.append(nxt)
    return out


class Solver:
    def __init__(self):
        self.memo = {}  # canon state -> (win, grundy, length)
        # length: winner's fastest forced win / loser's slowest forced loss

    def solve(self, state):
        state = canon(state)
        hit = self.memo.get(state)
        if hit is not None:
            return hit
        kids = {canon(s) for s in legal_moves(state)}
        if not kids:
            res = (False, 0, 0)  # to-move loses immediately
        else:
            sub = [self.solve(k) for k in kids]
            losing = [r for r in sub if not r[0]]
            seen = {r[1] for r in sub}
            g = 0
            while g in seen:
                g += 1
            if losing:
                res = (True, g, 1 + min(r[2] for r in losing))
            else:
                res = (False, g, 1 + max(r[2] for r in sub))
        self.memo[state] = res
        return res
