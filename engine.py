"""Tic-tac-toe rules plus a perfect (minimax) reference solver.

A board is a 9-tuple read left-to-right, top-to-bottom.
X (1) always moves first, O (-1) second, 0 is an empty square.
"""

from functools import lru_cache

EMPTY, X, O = 0, 1, -1

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

EMPTY_BOARD = (EMPTY,) * 9

MARK = {X: "X", O: "O", EMPTY: "."}


def legal_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]


def play(board, move, player):
    b = list(board)
    b[move] = player
    return tuple(b)


def winner(board):
    for a, b, c in LINES:
        v = board[a]
        if v != EMPTY and v == board[b] == board[c]:
            return v
    return EMPTY


def to_move(board):
    """Whose turn it is, derived from the board alone."""
    return X if sum(1 for v in board if v != EMPTY) % 2 == 0 else O


def is_over(board):
    return winner(board) != EMPTY or EMPTY not in board


def canonical(board, player):
    """The board seen from the given side: own marks +1, opponent -1.

    This is what lets one Q-table serve both X and O.
    """
    return board if player == X else tuple(-v for v in board)


def winning_moves(board, player):
    """Moves that immediately complete a line for the given player."""
    return [m for m in legal_moves(board) if winner(play(board, m, player)) == player]


# --- perfect play ---------------------------------------------------------

@lru_cache(maxsize=None)
def value(board):
    """Game-theoretic value for the side to move: +1 win, 0 draw, -1 loss."""
    if winner(board) != EMPTY:
        return -1                      # a line is already complete: the mover lost
    moves = legal_moves(board)
    if not moves:
        return 0
    p = to_move(board)
    return max(-value(play(board, m, p)) for m in moves)


@lru_cache(maxsize=None)
def optimal_moves(board):
    """Every move that preserves the game-theoretic value of the position."""
    p = to_move(board)
    best = value(board)
    return tuple(m for m in legal_moves(board) if -value(play(board, m, p)) == best)


# --- display --------------------------------------------------------------

def render(board, numbered=True, indent="  "):
    cells = []
    for i, v in enumerate(board):
        if v == EMPTY:
            cells.append(str(i + 1) if numbered else " ")
        else:
            cells.append(MARK[v])
    rows = [" {} | {} | {} ".format(*cells[i:i + 3]) for i in (0, 3, 6)]
    sep = "---+---+---"
    return "\n".join(indent + r for r in (rows[0], sep, rows[1], sep, rows[2]))
