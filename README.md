# Monte Carlo Tic-Tac-Toe

A tic-tac-toe player that starts out knowing nothing, learns purely by simulating
games, and ends up **provably unable to lose** — plus the tooling to find the exact
checkpoint where that happened and to play against any checkpoint along the way.

Pure Python 3, standard library only. Training takes about 20 seconds.

```bash
python train.py        # simulate, learn, save checkpoints
python evaluate.py     # score every checkpoint, find the "complete" one
python play.py         # play against any checkpoint
```

## How it learns

Tabular **Monte Carlo control**. There is no lookahead at play time — the agent
just reads a table.

1. Play a whole game by simulation (an *episode*).
2. When it ends, credit every move the agent made with the final result:
   `+1` win, `0` draw, `-1` loss.
3. Keep a running estimate `Q(state, action)` of that credit, updated with a
   decaying step size.
4. Play greedily with respect to `Q`, with a little exploration.

Two details do most of the work:

- **One table for both sides.** Every position is stored from the mover's own
  point of view (`engine.canonical`), so what it learns as X transfers to O.
- **Exploring starts.** 75% of episodes begin from a random legal position *and*
  a random first move, so no square of the state space stays unvisited just
  because the current policy avoids it.

### Why the opponent mix matters

Monte Carlo learns the **average** value of a move against the opponents it
actually meets. Against sloppy opponents, a losing move still averages out
positive — nobody punishes it — so the agent stays quietly beatable forever.
The default mix is therefore weighted toward an opponent that punishes
*everything*:

```
self=0.30  perfect=0.50  heuristic=0.10  random=0.10
```

The perfect (minimax) opponent turns "usually fine" into "never loses". The
weak opponents are still worth having: they teach the agent which of several
equally-safe moves is most likely to make a fallible opponent go wrong. In
testing, training against the perfect opponent alone still reached unbeatable,
but cost more than 10 percentage points of win rate against the heuristic bot
(≈38% vs ≈51%, averaged over three seeds) for no gain in safety.

## What "complete" means, and how it is proven

Not a win rate over sampled games — an exhaustive proof. `evaluate.audit()`
walks the entire game tree under two rules:

- the **agent** may play any move tied for best under its own table,
- the **opponent** may play *anything at all*.

If no leaf of that tree is a loss, no opponent can ever beat it. That is the
`unbeatable` column. A second, stricter bar (`blunders = 0`) checks that every
move it makes is minimax-optimal, so it never lets a won position slip to a
draw either.

Win/draw/loss numbers against the bots are also exact — computed by summing over
the whole game tree weighted by move probabilities, not sampled — so there is no
noise in the learning curve.

Training does not stop at a fixed episode count. It keeps running top-up rounds
until the audit passes, and saves *exactly* the table that passed, so
`final.json` is unbeatable by construction rather than by luck.

## Results from the shipped run

| episodes | vs random W/D/L | vs heuristic W/D/L | vs perfect | unbeatable | sub-optimal moves |
|---:|---|---|---|---|---:|
| 0 | 43.7 / 12.7 / 43.7 | 3.7 / 16.5 / 79.8 | loses 87% | no | 3191 |
| 1 500 | 59.4 / 12.1 / 28.5 | 10.6 / 30.4 / 59.0 | loses 63% | no | 784 |
| 12 000 | 83.9 / 9.5 / 6.5 | 20.0 / 69.0 / 11.1 | loses 16% | no | 126 |
| 40 000 | 89.1 / 10.3 / 0.7 | 48.9 / 51.1 / 0.0 | draws | no | 21 |
| 80 000 | 87.5 / 12.2 / 0.3 | 36.9 / 63.1 / 0.0 | draws | no | 7 |
| **100 000** | 90.7 / 9.3 / **0.0** | 51.5 / 48.5 / 0.0 | draws | **yes** | 2 |
| **150 000** | 90.9 / 9.1 / 0.0 | 35.0 / 65.0 / 0.0 | draws | yes | **0** |
| 400 000 (final) | 93.3 / 6.7 / **0.0** | 55.1 / 44.9 / 0.0 | draws 100% | yes | 0 |

- **Complete at 100 000 episodes** — from here on it cannot be beaten by anyone.
- **Perfect at 150 000 episodes** — from here on it also never misses a forced win.

At 80 000 episodes it was still beatable, and `evaluate.py` hands you the recipe:

```
Last beatable checkpoint: 80000 episodes -- here is how to beat it.
  It plays O: X1 O5 X7 O4 X9 O2 X8
    python play.py -c 80000 --side X --deterministic   then play 1, 7, 9, 8
```

That works move for move. Run the same four moves against `final` and you lose
instead — the difference between the two checkpoints is 20 000 episodes and about
seven positions.

(As X, the 80 000-episode agent is beatable only when its tie-breaks fall a
particular way, so no fixed sequence is guaranteed; `evaluate.py` says so and
prints one such line anyway.)

Note that the interesting part of the curve is the *last* 0.3%: the agent looks
excellent at 40 000 episodes and is still losable-to. Averages hide the handful
of positions that matter, which is exactly why the audit is exhaustive.

## Playing

```bash
python play.py                       # menu of checkpoints, then play
python play.py --list                # every checkpoint with its verdict
python play.py -c 3000               # play the 3 000-episode agent (still weak)
python play.py -c final --side O     # let it move first
python play.py -c 12000 --show-q     # see the value it puts on each square
python play.py -c 80000 --deterministic   # pin tie-breaks, to replay attack lines
python play.py --watch 6000 final    # watch two checkpoints play a game
python play.py --arena               # every checkpoint scored against the final agent
python play.py -c final --vs-bot perfect
```

Squares are numbered 1–9, left to right, top to bottom — the empty ones show
their number on the board.

`--show-q` is the most interesting flag for seeing learning happen. An early
checkpoint is wildly overconfident — after losing once from a position it will
stamp `-1.00` on every move it has tried there, because one game is its whole
experience:

```
ep800:  9:-1.00  8:-1.00  7:-1.00  6:-1.00  4:-1.00  3:-1.00  2:-1.00  1:-1.00
final:  7:+0.08  9:+0.04  3:+0.03  1:+0.03  4:-0.85  2:-0.88  6:-0.89  8:-0.95
```

The final agent has separated the moves that lose (near `-1`) from the moves
that are safe. The safe ones sit just *above* zero rather than at zero, and that
gap is the point: they all draw against perfect play, so what is left in the
number is how often each one wins against a fallible opponent. The ordering
among those near-zero values is the agent's taste in traps.

## Files

| file | what it holds |
|---|---|
| `engine.py` | rules, plus a minimax solver used only as a reference/opponent |
| `agent.py` | the Q-table, the greedy policy, the Monte Carlo update, save/load |
| `bots.py` | random, heuristic (win/block/else random), perfect, and checkpoint-as-opponent |
| `train.py` | the simulation loop, checkpointing, and the train-until-proven logic |
| `evaluate.py` | the exhaustive audit and the exact outcome maths; writes `results.json` |
| `play.py` | human vs checkpoint, checkpoint vs checkpoint, arena |
| `checkpoints/` | 20 snapshots + `final.json`, `manifest.json`, `results.json` |

## Retraining

```bash
python train.py --seed 7 --episodes 200000 --out checkpoints_seed7
python evaluate.py --checkpoints checkpoints_seed7
```

Useful knobs: `--mix` (opponent weights), `--explore-starts`, `--alpha-start` /
`--alpha-end`, `--eps-start` / `--eps-end`, `--every N` for a denser checkpoint
schedule, and `--until unbeatable` to stop as soon as it stops losing instead of
holding out for minimax-optimal play.

Different seeds complete at different points — roughly 60 000–130 000 episodes in
testing — so the "complete" checkpoint is a property of the run, not a constant.
