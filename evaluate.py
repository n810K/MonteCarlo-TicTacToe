"""Measure every checkpoint, and prove when one becomes unbeatable.

Nothing here is sampled.  Tic-tac-toe is small enough to answer the real
questions exhaustively:

  audit()      walks every position reachable when the agent plays its own
               greedy policy and the opponent is allowed to play *anything*.
               If no leaf of that tree is a loss, the agent cannot be beaten
               by any opponent, ever -- that is the "complete" checkpoint.

  outcome_vs() computes exact win/draw/loss probabilities against a bot by
               summing over the whole game tree weighted by move
               probabilities, so the numbers have no sampling noise.
"""

import argparse
import json
import os

from agent import MonteCarloAgent, state_key
from bots import make_bot
from engine import (EMPTY, EMPTY_BOARD, O, X, legal_moves, optimal_moves,
                    play, to_move, winner)

SIDE_NAME = {X: "X", O: "O"}


def audit(agent, side):
    """Exhaustive adversarial walk of the agent's greedy policy.

    The agent may pick any move tied for best (so the result also holds when
    it breaks ties randomly); the opponent may pick anything at all.
    """
    seen = set()
    blunders = set()
    loss_line = []

    def rec(board, path):
        if board in seen:
            return
        seen.add(board)
        w = winner(board)
        if w != EMPTY:
            if w != side and not loss_line:
                loss_line.append(list(path))
            return
        if EMPTY not in board:
            return
        p = to_move(board)
        if p == side:
            cands = agent.greedy(state_key(board, p), legal_moves(board))
            if not set(cands) <= set(optimal_moves(board)):
                blunders.add(board)
        else:
            cands = legal_moves(board)
        for m in cands:
            path.append(m)
            rec(play(board, m, p), path)
            path.pop()

    rec(EMPTY_BOARD, [])
    return {
        "unbeatable": not loss_line,
        "losing_line": loss_line[0] if loss_line else None,
        "blunders": len(blunders),
        "positions": len(seen),
    }


def attack_line(agent, side):
    """A concrete way to beat the agent, replayable move for move.

    audit() lets the agent pick any move tied for best, so the line it finds
    depends on how ties fall.  Here the agent is pinned to its first-listed
    best move -- exactly what `play.py --deterministic` does -- so the line
    this returns can actually be played back.
    """
    seen = set()
    found = []

    def rec(board, path):
        if found or board in seen:
            return
        seen.add(board)
        w = winner(board)
        if w != EMPTY:
            if w != side:
                found.append(list(path))
            return
        if EMPTY not in board:
            return
        p = to_move(board)
        if p == side:
            cands = agent.greedy(state_key(board, p), legal_moves(board))[:1]
        else:
            cands = legal_moves(board)
        for m in cands:
            path.append(m)
            rec(play(board, m, p), path)
            path.pop()

    rec(EMPTY_BOARD, [])
    return found[0] if found else None


def outcome_vs(agent, side, bot):
    """Exact (win, draw, loss) probabilities for the agent playing `side`."""
    memo = {}

    def rec(board):
        hit = memo.get(board)
        if hit is not None:
            return hit
        w = winner(board)
        if w != EMPTY:
            res = (1.0, 0.0, 0.0) if w == side else (0.0, 0.0, 1.0)
        elif EMPTY not in board:
            res = (0.0, 1.0, 0.0)
        else:
            p = to_move(board)
            if p == side:
                cands = agent.greedy(state_key(board, p), legal_moves(board))
                dist = [(m, 1.0 / len(cands)) for m in cands]
            else:
                dist = bot.distribution(board)
            win = draw = loss = 0.0
            for m, prob in dist:
                a, b, c = rec(play(board, m, p))
                win += prob * a
                draw += prob * b
                loss += prob * c
            res = (win, draw, loss)
        memo[board] = res
        return res

    win, draw, loss = rec(EMPTY_BOARD)
    return {"win": win, "draw": draw, "loss": loss}


def _avg(a, b):
    return {k: (a[k] + b[k]) / 2.0 for k in a}


def evaluate_agent(agent, opponents=("random", "heuristic", "perfect")):
    sides = {SIDE_NAME[s]: audit(agent, s) for s in (X, O)}
    report = {
        "episodes": agent.episodes,
        "states": agent.known_states,
        "sides": sides,
        "unbeatable": all(v["unbeatable"] for v in sides.values()),
        "blunders": sum(v["blunders"] for v in sides.values()),
        "vs": {},
    }
    report["optimal"] = report["unbeatable"] and report["blunders"] == 0
    for name in opponents:
        bot = make_bot(name)
        as_x = outcome_vs(agent, X, bot)
        as_o = outcome_vs(agent, O, bot)
        report["vs"][name] = {"X": as_x, "O": as_o, "avg": _avg(as_x, as_o)}
    return report


# --- CLI ------------------------------------------------------------------

def _pct(x):
    return "{:5.1f}".format(100.0 * x)


def _bar(x, width=22):
    filled = int(round(x * width))
    return "#" * filled + "." * (width - filled)


def moves_to_str(line):
    """Label each move with the side that played it; X always starts."""
    return " ".join("{}{}".format("XO"[i % 2], m + 1) for i, m in enumerate(line))


def print_table(rows):
    head = ("{:>10}  {:>6}  {:^7}  {:>17}  {:>17}  {:>17}  {:>8}"
            .format("episodes", "states", "unbeat", "vs random W/D/L",
                    "vs heuristic W/D/L", "vs perfect W/D/L", "blunders"))
    print(head)
    print("-" * len(head))
    for r in rows:
        flags = "{}{}".format("X" if r["sides"]["X"]["unbeatable"] else "-",
                              "O" if r["sides"]["O"]["unbeatable"] else "-")
        cells = []
        for name in ("random", "heuristic", "perfect"):
            v = r["vs"][name]["avg"]
            cells.append("{} {} {}".format(_pct(v["win"]), _pct(v["draw"]), _pct(v["loss"])))
        print("{:>10}  {:>6}  {:^7}  {}  {}  {}  {:>8}".format(
            r["episodes"], r["states"], flags, cells[0], cells[1], cells[2], r["blunders"]))


def print_curve(rows):
    print()
    print("How often it still loses (averaged over playing X and O):")
    print("{:>10}  {:<22} {:<22}".format("episodes", "vs random", "vs heuristic"))
    for r in rows:
        lr = r["vs"]["random"]["avg"]["loss"]
        lh = r["vs"]["heuristic"]["avg"]["loss"]
        print("{:>10}  {} {:>5}%  {} {:>5}%".format(
            r["episodes"], _bar(lr), round(100 * lr, 1), _bar(lh), round(100 * lh, 1)))


def load_manifest(ckpt_dir):
    with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description="Evaluate every training checkpoint.")
    ap.add_argument("--checkpoints", default="checkpoints", help="checkpoint directory")
    ap.add_argument("--out", default=None, help="where to write results.json")
    args = ap.parse_args()

    manifest = load_manifest(args.checkpoints)
    rows = []
    print("Evaluating {} checkpoints from {}/ ...\n".format(
        len(manifest["checkpoints"]), args.checkpoints))
    for entry in manifest["checkpoints"]:
        agent = MonteCarloAgent.load(os.path.join(args.checkpoints, entry["file"]))
        rep = evaluate_agent(agent)
        rep["file"] = entry["file"]
        rep["label"] = entry.get("label", str(entry["episodes"]))
        rows.append(rep)

    print_table(rows)
    print_curve(rows)

    complete = next((r for r in rows if r["unbeatable"]), None)
    perfect = next((r for r in rows if r["optimal"]), None)

    print()
    print("=" * 78)
    if complete:
        print("COMPLETE at {} episodes ({}).".format(complete["episodes"], complete["file"]))
        print("  Proven by exhaustive search: playing either side, against every")
        print("  possible opponent reply, it never reaches a lost position.")
    else:
        print("No checkpoint is unbeatable yet -- train for longer.")
    if perfect:
        print("PERFECT at {} episodes ({}): every move it plays is minimax-optimal,"
              .format(perfect["episodes"], perfect["file"]))
        print("  so it also punishes every mistake the opponent makes.")
    elif complete:
        print("Still {} position(s) where it settles for less than the best move"
              .format(rows[-1]["blunders"]))
        print("  -- safe, but it lets some winnable games slip to a draw.")
    print("=" * 78)

    last_beatable = None
    for r in rows:
        if not r["unbeatable"]:
            last_beatable = r
    if last_beatable:
        print()
        print("Last beatable checkpoint: {} episodes -- here is how to beat it."
              .format(last_beatable["episodes"]))
        weak = MonteCarloAgent.load(os.path.join(args.checkpoints, last_beatable["file"]))
        for side, code in (("X", X), ("O", O)):
            line = attack_line(weak, code)
            if line:
                you = "O" if side == "X" else "X"
                yours = [m for i, m in enumerate(line) if (i % 2 == 0) == (you == "X")]
                print("  It plays {}: {}".format(side, moves_to_str(line)))
                print("    python play.py -c {} --side {} --deterministic   "
                      "then play {}".format(last_beatable["episodes"], you,
                                            ", ".join(str(m + 1) for m in yours)))
                last_beatable["sides"][side]["attack_line"] = line
            elif last_beatable["sides"][side]["losing_line"]:
                print("  It plays {}: beatable only when its tie-breaks fall a "
                      "certain way -- one such line is {}".format(
                          side, moves_to_str(last_beatable["sides"][side]["losing_line"])))

    out = args.out or os.path.join(args.checkpoints, "results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"complete_at": complete["episodes"] if complete else None,
                   "perfect_at": perfect["episodes"] if perfect else None,
                   "rows": rows}, fh, indent=2)
    print("\nFull results written to {}".format(out))


if __name__ == "__main__":
    main()
