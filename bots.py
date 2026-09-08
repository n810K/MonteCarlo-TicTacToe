"""Opponents the learner trains against and is measured against.

Every bot exposes move(board) for simulation and distribution(board) for the
exact (non-sampled) evaluation in evaluate.py.
"""

import random

from agent import state_key
from engine import legal_moves, optimal_moves, to_move, winning_moves


class RandomBot:
    """Uniformly random legal move."""
    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def _candidates(self, board):
        return legal_moves(board)

    def move(self, board):
        return self.rng.choice(self._candidates(board))

    def distribution(self, board):
        cands = self._candidates(board)
        p = 1.0 / len(cands)
        return [(m, p) for m in cands]


class HeuristicBot(RandomBot):
    """Take the win, else block the loss, else move at random.

    The classic decent-human bot: it never misses a one-move tactic, but it
    has no plan, so it walks into forks.
    """
    name = "heuristic"

    def _candidates(self, board):
        me = to_move(board)
        wins = winning_moves(board, me)
        if wins:
            return wins
        blocks = winning_moves(board, -me)
        if blocks:
            return blocks
        return legal_moves(board)


class PerfectBot(RandomBot):
    """Full minimax. Cannot be beaten; wins if you slip."""
    name = "perfect"

    def _candidates(self, board):
        return list(optimal_moves(board))


class AgentBot(RandomBot):
    """Wraps a trained checkpoint so it can be an opponent too."""

    def __init__(self, agent, rng=None):
        super().__init__(rng)
        self.agent = agent
        self.name = agent.name

    def _candidates(self, board):
        p = to_move(board)
        return self.agent.greedy(state_key(board, p), legal_moves(board))


BOTS = {"random": RandomBot, "heuristic": HeuristicBot, "perfect": PerfectBot}


def make_bot(name, rng=None):
    return BOTS[name](rng)
