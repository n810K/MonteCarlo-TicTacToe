"""Play against any training checkpoint -- or watch two of them play.

  python play.py                      pick a checkpoint from a menu and play
  python play.py --checkpoint 800     jump straight to the 800-episode agent
  python play.py --list               show every checkpoint and its verdict
  python play.py --watch 400 final    watch two checkpoints play one game
  python play.py --arena              every checkpoint vs the final agent
"""

import argparse
import json
import os
import random
import sys

from agent import MonteCarloAgent, state_key
from bots import AgentBot, make_bot
from engine import (EMPTY, EMPTY_BOARD, O, X, is_over, legal_moves, play,
                    render, to_move, winner)
from evaluate import outcome_vs

SIDE_NAME = {X: "X", O: "O"}


# --- checkpoint bookkeeping ----------------------------------------------

def load_index(ckpt_dir):
    with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    verdicts = {}
    res_path = os.path.join(ckpt_dir, "results.json")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as fh:
            for row in json.load(fh)["rows"]:
                verdicts[row["file"]] = row
    return manifest["checkpoints"], verdicts


def describe(entry, verdicts):
    row = verdicts.get(entry["file"])
    if not row:
        return "{:>8} episodes  {}".format(entry["episodes"], entry["label"])
    vs = row["vs"]["random"]["avg"]
    tag = "PERFECT" if row.get("optimal") else ("unbeatable" if row["unbeatable"] else "beatable")
    return ("{:>8} episodes  {:<11}  vs random: win {:>5.1f}%  lose {:>5.1f}%"
            .format(entry["episodes"], tag, 100 * vs["win"], 100 * vs["loss"]))


def resolve(spec, entries):
    """Accept a label, a filename, an episode count, or a menu index."""
    spec = str(spec).strip()
    for e in entries:
        if spec in (e["label"], e["file"], str(e["episodes"])):
            return e
    if spec.isdigit():
        n = int(spec)
        if 1 <= n <= len(entries):
            return entries[n - 1]
        return min(entries, key=lambda e: abs(e["episodes"] - n))
    raise SystemExit("No checkpoint matching {!r}".format(spec))


def load_agent(ckpt_dir, entry):
    a = MonteCarloAgent.load(os.path.join(ckpt_dir, entry["file"]))
    a.name = "{} ({} episodes)".format(entry["label"], entry["episodes"])
    return a


# --- a game ---------------------------------------------------------------

def show_thinking(agent, board, player):
    key = state_key(board, player)
    entry = agent.table.get(key, {})
    if not entry:
        print("    (it has never seen this position -- it is guessing)")
        return
    scored = sorted(((rec[1], m, rec[0]) for m, rec in entry.items()), reverse=True)
    bits = ["{}:{:+.2f}".format(m + 1, v) for v, m, _ in scored]
    print("    it rates the squares: " + "  ".join(bits))


def human_move(board):
    while True:
        raw = input("  your move (1-9, q to quit): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= 9:
            cell = int(raw) - 1
            if board[cell] == EMPTY:
                return cell
            print("  that square is taken.")
        else:
            print("  enter a square number from the grid above.")


def choose(agent, board, player, rng, deterministic=False):
    """The agent's move. Ties are broken at random unless pinned, which makes
    the attack lines printed by evaluate.py replayable."""
    cands = agent.greedy(state_key(board, player), legal_moves(board))
    return cands[0] if deterministic else rng.choice(cands)


def play_game(agent, human_side, rng, show_q=False, deterministic=False):
    board = EMPTY_BOARD
    print("\nYou are {} -- {} is {}.\n".format(
        SIDE_NAME[human_side], agent.name, SIDE_NAME[-human_side]))
    while not is_over(board):
        print(render(board))
        p = to_move(board)
        if p == human_side:
            m = human_move(board)
            if m is None:
                return None
        else:
            m = choose(agent, board, p, rng, deterministic)
            print("  it plays {}".format(m + 1))
            if show_q:
                show_thinking(agent, board, p)
        board = play(board, m, p)
        print()
    print(render(board, numbered=False))
    w = winner(board)
    if w == EMPTY:
        print("\n  Draw.")
    elif w == human_side:
        print("\n  You win -- this checkpoint is beatable.")
    else:
        print("\n  You lose.")
    return w


def watch(a, b, rng, quiet=False, deterministic=False):
    """One game: agent `a` as X, agent `b` as O."""
    board = EMPTY_BOARD
    players = {X: a, O: b}
    while not is_over(board):
        p = to_move(board)
        ag = players[p]
        m = choose(ag, board, p, rng, deterministic)
        board = play(board, m, p)
        if not quiet:
            print("  {} ({}) plays {}".format(SIDE_NAME[p], ag.name, m + 1))
            print(render(board, numbered=False))
            print()
    w = winner(board)
    if not quiet:
        print("  Result: {}".format("draw" if w == EMPTY else SIDE_NAME[w] + " wins"))
    return w


# --- modes ----------------------------------------------------------------

def do_list(entries, verdicts):
    print("\nCheckpoints:")
    for i, e in enumerate(entries, 1):
        print("  {:>2}. {}".format(i, describe(e, verdicts)))
    if not verdicts:
        print("\n  (run: python evaluate.py   to fill in the verdict column)")


def do_arena(ckpt_dir, entries, verdicts, against):
    opponent_entry = resolve(against, entries)
    opponent = load_agent(ckpt_dir, opponent_entry)
    bot = AgentBot(opponent)
    print("\nEvery checkpoint against {} -- exact outcome, both sides averaged:\n"
          .format(opponent.name))
    print("{:>10}  {:>7}  {:>7}  {:>7}".format("episodes", "win", "draw", "lose"))
    print("-" * 38)
    for e in entries:
        ag = load_agent(ckpt_dir, e)
        x = outcome_vs(ag, X, bot)
        o = outcome_vs(ag, O, bot)
        avg = {k: (x[k] + o[k]) / 2 for k in x}
        print("{:>10}  {:>6.1f}%  {:>6.1f}%  {:>6.1f}%".format(
            e["episodes"], 100 * avg["win"], 100 * avg["draw"], 100 * avg["loss"]))


def do_bot_match(ckpt_dir, entry, bot_name, games, rng):
    ag = load_agent(ckpt_dir, entry)
    bot = make_bot(bot_name, rng)
    print("\n{} vs {} -- exact probabilities:\n".format(ag.name, bot_name))
    for side in (X, O):
        r = outcome_vs(ag, side, bot)
        print("  as {}:  win {:>5.1f}%   draw {:>5.1f}%   lose {:>5.1f}%".format(
            SIDE_NAME[side], 100 * r["win"], 100 * r["draw"], 100 * r["loss"]))


def interactive(ckpt_dir, entries, verdicts, rng, show_q, deterministic=False):
    do_list(entries, verdicts)
    while True:
        raw = input("\nCheckpoint to play against (number, episode count, or q): ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            return
        try:
            entry = resolve(raw, entries)
        except SystemExit as exc:
            print("  " + str(exc))
            continue
        agent = load_agent(ckpt_dir, entry)
        side_raw = input("Play as X (first) or O (second)? [X/o]: ").strip().upper()
        human_side = O if side_raw.startswith("O") else X
        while True:
            if play_game(agent, human_side, rng, show_q, deterministic) is None:
                break
            again = input("\nAgain? [y = rematch, s = swap sides, "
                          "Enter = another checkpoint, q = quit]: ").strip().lower()
            if again == "s":
                human_side = -human_side
            elif again == "y":
                continue
            elif again in ("q", "quit"):
                return
            else:
                break
        do_list(entries, verdicts)


def main():
    ap = argparse.ArgumentParser(description="Play against a training checkpoint.")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--checkpoint", "-c", default=None,
                    help="label, filename, or episode count")
    ap.add_argument("--side", default="X", choices=["X", "O", "x", "o"])
    ap.add_argument("--list", action="store_true", help="list checkpoints and exit")
    ap.add_argument("--show-q", action="store_true",
                    help="print the values the agent assigns to each square")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin the agent's tie-breaks, so the attack lines "
                         "printed by evaluate.py replay exactly")
    ap.add_argument("--watch", nargs=2, metavar=("X_AGENT", "O_AGENT"),
                    help="watch two checkpoints play")
    ap.add_argument("--arena", action="store_true",
                    help="score every checkpoint against one opponent")
    ap.add_argument("--against", default="final", help="opponent for --arena")
    ap.add_argument("--vs-bot", choices=["random", "heuristic", "perfect"],
                    help="score one checkpoint against a scripted bot")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.checkpoints, "manifest.json")):
        raise SystemExit("No checkpoints found in {}/ -- run: python train.py"
                         .format(args.checkpoints))
    entries, verdicts = load_index(args.checkpoints)
    rng = random.Random(args.seed)

    if args.list:
        do_list(entries, verdicts)
        return
    if args.arena:
        do_arena(args.checkpoints, entries, verdicts, args.against)
        return
    if args.watch:
        a = load_agent(args.checkpoints, resolve(args.watch[0], entries))
        b = load_agent(args.checkpoints, resolve(args.watch[1], entries))
        print("\n{} (X)  vs  {} (O)\n".format(a.name, b.name))
        watch(a, b, rng, deterministic=args.deterministic)
        return
    if args.vs_bot:
        entry = resolve(args.checkpoint or "final", entries)
        do_bot_match(args.checkpoints, entry, args.vs_bot, 0, rng)
        return

    if args.checkpoint:
        entry = resolve(args.checkpoint, entries)
        agent = load_agent(args.checkpoints, entry)
        human_side = O if args.side.upper() == "O" else X
        play_game(agent, human_side, rng, args.show_q, args.deterministic)
        return

    try:
        interactive(args.checkpoints, entries, verdicts, rng, args.show_q,
                    args.deterministic)
    except (EOFError, KeyboardInterrupt):
        print("\nbye.")


if __name__ == "__main__":
    main()
