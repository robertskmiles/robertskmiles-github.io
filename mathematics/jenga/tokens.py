"""The token game that idealized Jenga (game.py) reduces to.

Discovery (2026-08-28): across every state reachable from full towers of
2..7 layers (~2M canonical states solved exactly by game.Solver), win/loss
AND Grundy value depend only on three numbers:

  F  = number of removable full (3-block) layers
  S2 = number of removable side+middle pairs (LM/MR)
  t  = blocks in the incomplete top layer (0 if the top is complete)

"Removable" = strictly below the highest completed level (official rule).
All other layer content is strategically dead:
  LR   (both sides, no middle): no block can ever be removed (taking a side
       leaves a lone side = collapse, by the lone-side theorem).
  M1   (lone middle): last block of a layer, never removable.
  Layer positions, placement slot choices, and all COM coupling turn out
  never to change the game value - only the counts matter.

Token game rules (exactly mirroring the real move structure):
  A move consumes one heap token and advances t by 1; when t reaches 3 the
  top layer is complete: t resets to 0 and the previously frozen complete
  layer is released into the pool (F += 1).
    - from an F layer: remove its middle  -> layer becomes dead LR  (F -= 1)
                       or remove a side   -> layer becomes S2       (F -= 1, S2 += 1)
    - from an S2 layer: remove its side   -> layer becomes dead M1  (S2 -= 1)
      (the middle of an S2 layer is pinned: removing it = lone side = fall)
  No moves available => player to move loses (they must topple the tower).

Solution: Grundy values are periodic with period 3 in F and in S2 for each
t (verified far beyond the printed grids). Full towers of n layers start at
(F, S2, t) = (n-1, 0, 0):

  FIRST player wins iff n is NOT a multiple of 3, and the winning openings
  are exactly: remove a MIDDLE block (from any allowed layer, place it
  anywhere). Standard 18-layer Jenga: SECOND player wins.

P-positions (previous player wins), from the grids (r = F mod 3, s = S2 mod 3):
  t=0:  P  iff  (r=1 and s=1)  or  (r=2 and s!=1)   [exception: F=0 acts like r=2]
  t=1:  P  iff  s=0 and r != 1
  t=2:  P  iff  (r=0 and s!=2) or  (r=1 and s=2)
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def solve(F, S2, t):
    """Return (first_player_wins, grundy) for token state (F, S2, t)."""
    def step(F2, S22):
        return (F2 + 1, S22, 0) if t == 2 else (F2, S22, t + 1)
    opts = []
    if F:
        opts += [step(F - 1, S2), step(F - 1, S2 + 1)]  # middle-mine / side-mine
    if S2:
        opts += [step(F, S2 - 1)]
    if not opts:
        return (False, 0)
    sub = [solve(*o) for o in opts]
    seen = {r[1] for r in sub}
    g = 0
    while g in seen:
        g += 1
    return (any(not r[0] for r in sub), g)


def full_tower_value(n):
    """(first_player_wins, grundy) for the full n-layer starting tower."""
    return solve(n - 1, 0, 0)


if __name__ == "__main__":
    for n in range(2, 31):
        w, g = full_tower_value(n)
        print(f"n={n:2d}: {'first' if w else 'SECOND'} player wins (Grundy {g})")
