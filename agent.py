"""A tabular Monte-Carlo-control agent.

It plays whole games, then credits every (state, action) it took with the
final result of that game (+1 win / 0 draw / -1 loss) and keeps a running
mean.  That mean is Q(s, a).  There is no lookahead at play time: it just
reads the table.  Everything is stored from the mover's own point of view
(see engine.canonical), so a single table plays both X and O.
"""

import json
import os

from engine import canonical

UNSEEN = 0.0  # an untried move is assumed to draw -- better than a known loss

_CHARS = {1: "x", -1: "o", 0: "."}


def encode(state):
    """Canonical board -> short string key (x = me, o = opponent)."""
    return "".join(_CHARS[v] for v in state)


def state_key(board, player):
    return encode(canonical(board, player))


class MonteCarloAgent:
    def __init__(self, name="agent", alpha=0.0):
        self.name = name
        self.episodes = 0
        self.meta = {}
        # Step size for the Q update.  0 means a true running mean (1/N);
        # a positive alpha is a constant step, which forgets the returns
        # collected under an older, worse policy much faster.
        self.alpha = alpha
        # key -> {move: [visits, mean_return]}
        self.table = {}

    # --- policy ----------------------------------------------------------

    def q(self, key, move):
        entry = self.table.get(key)
        if not entry:
            return UNSEEN
        rec = entry.get(move)
        return rec[1] if rec else UNSEEN

    def greedy(self, key, moves):
        """All moves tied for the highest Q -- the agent may pick any of them."""
        entry = self.table.get(key, {})
        best = None
        out = []
        for m in moves:
            rec = entry.get(m)
            v = rec[1] if rec else UNSEEN
            if best is None or v > best:
                best, out = v, [m]
            elif v == best:
                out.append(m)
        return out

    def act(self, key, moves, epsilon, rng):
        if epsilon and rng.random() < epsilon:
            return rng.choice(moves)
        return rng.choice(self.greedy(key, moves))

    # --- learning --------------------------------------------------------

    def update(self, key, move, ret):
        entry = self.table.setdefault(key, {})
        rec = entry.get(move)
        if rec is None:
            entry[move] = [1, float(ret)]
        else:
            rec[0] += 1
            step = self.alpha if self.alpha > 0 else 1.0 / rec[0]
            rec[1] += (ret - rec[1]) * step

    def learn_from(self, trajectory, result):
        """trajectory: list of (key, move, player); result: winning side or EMPTY."""
        for key, move, player in trajectory:
            if result == 0:
                ret = 0.0
            else:
                ret = 1.0 if result == player else -1.0
            self.update(key, move, ret)
        self.episodes += 1

    # --- persistence -----------------------------------------------------

    def save(self, path):
        blob = {
            "name": self.name,
            "episodes": self.episodes,
            "alpha": self.alpha,
            "meta": self.meta,
            "table": {k: {str(m): v for m, v in e.items()} for k, e in self.table.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        a = cls(blob.get("name") or os.path.basename(path), blob.get("alpha", 0.0))
        a.episodes = blob.get("episodes", 0)
        a.meta = blob.get("meta", {})
        a.table = {
            k: {int(m): [rec[0], rec[1]] for m, rec in e.items()}
            for k, e in blob["table"].items()
        }
        return a

    def clone(self):
        c = MonteCarloAgent(self.name, self.alpha)
        c.episodes = self.episodes
        c.meta = dict(self.meta)
        c.table = {k: {m: list(v) for m, v in e.items()} for k, e in self.table.items()}
        return c

    @property
    def known_states(self):
        return len(self.table)
