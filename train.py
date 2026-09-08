"""Train the Monte Carlo agent by simulation, saving checkpoints as it goes.

Each episode is one complete simulated game.  The agent plays a mix of
self-play and games against scripted bots, always from a random side, and
sometimes from a random mid-game position (exploring starts) so that no
corner of the board goes unlearned.  When the game ends, every move it made
is credited with the final result.

Training does not stop at a fixed episode count: it keeps going until an
exhaustive adversarial search proves the agent cannot lose as either side.
"""

import argparse
import json
import os
import random
import time

from agent import MonteCarloAgent, state_key
from bots import make_bot
from engine import (EMPTY_BOARD, O, X, is_over, legal_moves, play, to_move,
                    winner)
from evaluate import audit, outcome_vs

DEFAULT_SCHEDULE = [0, 25, 50, 100, 200, 400, 800, 1500, 3000, 6000, 12000,
                    25000, 40000, 60000, 80000, 100000, 150000, 200000,
                    300000, 400000]

# The perfect opponent is deliberately the largest share.  Monte Carlo learns
# the AVERAGE value of a move over the opponents it meets; an opponent that
# punishes every mistake is what turns "usually fine" into "never loses".
DEFAULT_MIX = "self=0.30,perfect=0.50,heuristic=0.10,random=0.10"


def parse_mix(text):
    pairs = []
    for part in text.split(","):
        name, _, weight = part.partition("=")
        pairs.append((name.strip(), float(weight)))
    total = sum(w for _, w in pairs)
    cum, acc = [], 0.0
    for name, w in pairs:
        acc += w / total
        cum.append((name, acc))
    return cum


def pick(cum, rng):
    r = rng.random()
    for name, edge in cum:
        if r <= edge:
            return name
    return cum[-1][0]


def random_start(rng):
    """A random legal, still-playable position -- the exploring start."""
    while True:
        board = EMPTY_BOARD
        for _ in range(rng.randint(0, 6)):
            p = to_move(board)
            board = play(board, rng.choice(legal_moves(board)), p)
            if is_over(board):
                break
        else:
            return board


def run_episode(agent, rng, epsilon, opponent, explore_start):
    """Play one game; return the winning side (or 0). Learns from it.

    An exploring start means a random position *and* a random first action,
    so every (state, action) pair keeps getting sampled no matter how
    confident the current policy has become.
    """
    board = random_start(rng) if explore_start else EMPTY_BOARD
    learner_side = rng.choice((X, O)) if opponent else None
    forced = explore_start
    trajectory = []
    while not is_over(board):
        p = to_move(board)
        moves = legal_moves(board)
        if opponent is None or p == learner_side:
            key = state_key(board, p)
            m = rng.choice(moves) if forced else agent.act(key, moves, epsilon, rng)
            trajectory.append((key, m, p))
        else:
            m = rng.choice(moves) if forced else opponent.move(board)
        forced = False
        board = play(board, m, p)
    result = winner(board)
    agent.learn_from(trajectory, result)
    return result


def annealed(step, total, start, end, anneal):
    """Linear decay from `start` to `end` over the first `anneal` of training."""
    frac = min(1.0, step / max(1.0, total * anneal))
    return start + (end - start) * frac


def status(agent):
    ax, ao = audit(agent, X), audit(agent, O)
    return ax, ao, "{}{}".format("X" if ax["unbeatable"] else "-",
                                 "O" if ao["unbeatable"] else "-")


def main():
    ap = argparse.ArgumentParser(description="Train a Monte Carlo tic-tac-toe agent.")
    ap.add_argument("--episodes", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--alpha-start", type=float, default=0.20,
                    help="learning rate at the start; 0 means a 1/N running mean")
    ap.add_argument("--alpha-end", type=float, default=0.01,
                    help="learning rate once annealing finishes")
    ap.add_argument("--eps-start", type=float, default=0.20)
    ap.add_argument("--eps-end", type=float, default=0.0)
    ap.add_argument("--anneal", type=float, default=0.6,
                    help="fraction of training over which alpha and epsilon decay")
    ap.add_argument("--explore-starts", type=float, default=0.75,
                    help="probability an episode starts from a random position")
    ap.add_argument("--mix", default=DEFAULT_MIX,
                    help="opponent mix, e.g. " + DEFAULT_MIX)
    ap.add_argument("--every", type=int, default=0,
                    help="also checkpoint every N episodes")
    ap.add_argument("--extra-round", type=int, default=25000,
                    help="episodes per top-up round if the goal is not met yet")
    ap.add_argument("--max-extra", type=int, default=40,
                    help="maximum top-up rounds")
    ap.add_argument("--until", choices=["unbeatable", "optimal"], default="optimal",
                    help="unbeatable = never loses; optimal = also never misses a win")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cum = parse_mix(args.mix)
    bots = {name: make_bot(name, rng) for name, _ in cum if name != "self"}

    os.makedirs(args.out, exist_ok=True)
    schedule = sorted({s for s in DEFAULT_SCHEDULE if s <= args.episodes} | {args.episodes})
    if args.every:
        schedule = sorted(set(schedule) | set(range(args.every, args.episodes + 1, args.every)))

    agent = MonteCarloAgent("mc", alpha=args.alpha_start)
    agent.meta = {"seed": args.seed, "mix": args.mix,
                  "alpha": [args.alpha_start, args.alpha_end],
                  "eps": [args.eps_start, args.eps_end]}

    entries = []

    def checkpoint(label=None):
        fname = "ckpt_{:07d}.json".format(agent.episodes)
        agent.name = label or "ep{}".format(agent.episodes)
        agent.save(os.path.join(args.out, fname))
        entries.append({"episodes": agent.episodes, "file": fname,
                        "label": agent.name})
        return fname

    print("Training {} episodes  (mix: {})".format(args.episodes, args.mix))
    print("{:>10}  {:>7}  {:>6}  {:>6}  {:>9}".format(
        "episodes", "epsilon", "states", "unbeat", "loss% rnd"))

    checkpoint("untrained")
    started = time.time()
    next_idx = 1

    for step in range(1, args.episodes + 1):
        eps = annealed(step, args.episodes, args.eps_start, args.eps_end, args.anneal)
        agent.alpha = annealed(step, args.episodes, args.alpha_start,
                               args.alpha_end, args.anneal)
        opponent = bots.get(pick(cum, rng))
        run_episode(agent, rng, eps, opponent, rng.random() < args.explore_starts)

        if next_idx < len(schedule) and step == schedule[next_idx]:
            _, _, flags = status(agent)
            rnd = make_bot("random")
            loss = (outcome_vs(agent, X, rnd)["loss"] + outcome_vs(agent, O, rnd)["loss"]) / 2
            print("{:>10}  {:>7.3f}  {:>6}  {:>6}  {:>8.2f}%".format(
                step, eps, agent.known_states, flags, 100 * loss))
            checkpoint()
            next_idx += 1

    # --- keep training until the goal is provably met ---------------------
    # The audit is exhaustive, so a table that passes it is unbeatable for
    # good: we stop the moment it passes and save exactly that table.
    best = None                      # (blunders, snapshot)
    rounds = 0
    while True:
        ax, ao, flags = status(agent)
        blunders = ax["blunders"] + ao["blunders"]
        if ax["unbeatable"] and ao["unbeatable"]:
            if best is None or blunders < best[0]:
                best = (blunders, agent.clone())
            if blunders == 0 or args.until == "unbeatable":
                break
        if rounds >= args.max_extra:
            break
        rounds += 1
        print("Goal not met (unbeatable {}, {} sub-optimal move(s)); "
              "top-up round {}/{} ...".format(flags, blunders, rounds, args.max_extra))
        agent.alpha = args.alpha_end
        for _ in range(args.extra_round):
            run_episode(agent, rng, args.eps_end, bots.get(pick(cum, rng)),
                        rng.random() < args.explore_starts)
        checkpoint()

    if best is not None and best[1].episodes != agent.episodes:
        print("Keeping the strongest table seen ({} sub-optimal move(s) left).".format(best[0]))
        agent = best[1]

    ax, ao, flags = status(agent)
    agent.name = "final"
    agent.meta["unbeatable"] = ax["unbeatable"] and ao["unbeatable"]
    agent.meta["blunders"] = ax["blunders"] + ao["blunders"]
    agent.save(os.path.join(args.out, "final.json"))
    entries = [e for e in entries if e["episodes"] != agent.episodes]
    entries.sort(key=lambda e: e["episodes"])
    entries.append({"episodes": agent.episodes, "file": "final.json", "label": "final"})

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"episodes": agent.episodes, "config": vars(args),
                   "checkpoints": entries}, fh, indent=2)

    elapsed = time.time() - started
    print()
    print("Done in {:.1f}s -- {} episodes, {} states learned.".format(
        elapsed, agent.episodes, agent.known_states))
    print("Final agent unbeatable as X: {}   as O: {}   sub-optimal moves left: {}".format(
        ax["unbeatable"], ao["unbeatable"], ax["blunders"] + ao["blunders"]))
    print("{} checkpoints in {}/  -- run: python evaluate.py".format(len(entries), args.out))


if __name__ == "__main__":
    main()
