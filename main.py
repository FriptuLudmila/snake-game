# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____/______>___|__(______/__|__\_____>
#
# Tasks 1–5:
#  - Task 1: Evaluation = length advantage + reachable space
#  - Task 2: MiniMax with Alpha–Beta pruning (+ move ordering)
#  - Task 3: Improved evaluation (local openness / hazard danger / #shorter)
#  - Task 4: Flood-Fill Caching (memoize reachable-space BFS)
#  - Task 5: Dead-end avoidance using BFS region size threshold in move ordering
#
# Starter template from: https://github.com/BattlesnakeOfficial/starter-snake-python

import random
import typing
from copy import deepcopy
from collections import deque
from functools import lru_cache


# ------------------------ Appearance / API handlers ------------------------

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "",            # TODO: add your Battlesnake username
        "color": "#16a085",      # cosmetic
        "head": "beluga",        # cosmetic
        "tail": "bolt",          # cosmetic
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


# ------------------------------ Core helpers ------------------------------

# Movement vectors
MOVE_VECTORS = {
    "up":    (0, 1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1, 0),
}


def _in_bounds(pt: typing.Dict, width: int, height: int) -> bool:
    return 0 <= pt["x"] < width and 0 <= pt["y"] < height


def _add(a: typing.Dict, dv: typing.Tuple[int, int]) -> typing.Dict:
    return {"x": a["x"] + dv[0], "y": a["y"] + dv[1]}


def _copy_state(gs: typing.Dict) -> typing.Dict:
    return deepcopy(gs)


def _occupied_after_move(gs: typing.Dict) -> set[tuple]:
    """
    Cells currently occupied by all snakes' bodies.
    (Simplified model used by Tasks 1–5: no future growth considered here.)
    """
    occ: set[tuple] = set()
    for s in gs["board"]["snakes"]:
        for seg in s["body"]:
            occ.add((seg["x"], seg["y"]))
    return occ


def _legal_moves_for_snake(gs: typing.Dict, snake_idx: int) -> list[str]:
    """
    Legal moves for the snake at index `snake_idx` with simplified rules:
    - in-bounds only
    - cannot step onto any currently occupied cell
    - BUT may step onto your *own* current tail (it moves this turn)
    - avoid immediate 180° backtrack into your neck
    """
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    snakes = gs["board"]["snakes"]
    snake = snakes[snake_idx]
    head = snake["body"][0]
    tail = snake["body"][-1]

    occupied = _occupied_after_move(gs)
    # allow stepping into your current tail (it will pop this turn)
    occupied.discard((tail["x"], tail["y"]))

    moves: list[str] = []
    for m, dv in MOVE_VECTORS.items():
        nh = _add(head, dv)
        if not _in_bounds(nh, width, height):
            continue
        if (nh["x"], nh["y"]) in occupied:
            continue
        # avoid turning back into neck if length >= 2
        if len(snake["body"]) >= 2:
            neck = snake["body"][1]
            if nh["x"] == neck["x"] and nh["y"] == neck["y"]:
                continue
        moves.append(m)
    return moves


def _simulate_step(gs: typing.Dict, chosen_moves: dict[str, str]) -> typing.Dict:
    """
    Advance state by one turn for the subset of snakes in `chosen_moves`.
    For each specified snake: add a head in the move direction, pop tail (no growth).
    Other snakes are left as-is for this ply (treated as static obstacles while we search).
    """
    next_state = _copy_state(gs)
    snakes = next_state["board"]["snakes"]
    id_to_idx = {s["id"]: i for i, s in enumerate(snakes)}

    for sid, move in chosen_moves.items():
        i = id_to_idx.get(sid)
        if i is None:
            continue
        snake = snakes[i]
        dv = MOVE_VECTORS[move]
        new_head = _add(snake["body"][0], dv)
        snake["body"].insert(0, new_head)
        snake["body"].pop()  # no food / no growth in these tasks

    return next_state


# -------------------- Task 4: Flood-Fill Caching --------------------

@lru_cache(maxsize=8192)
def _reachable_space_cached(width: int, height: int,
                            occupied_key: tuple, start_t: tuple) -> int:
    """
    Cached BFS flood-fill count of reachable empty cells.
    `occupied_key` is a tuple of (x,y) tuples representing blocked cells.
    """
    occupied = set(occupied_key)
    q = deque([start_t])
    seen = {start_t}
    count = 0

    while q:
        x, y = q.popleft()
        if (x, y) != start_t:
            count += 1
        for dv in MOVE_VECTORS.values():
            nx, ny = x + dv[0], y + dv[1]
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in occupied:
                continue
            if (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    return count


def _reachable_space(gs: typing.Dict, start: typing.Dict) -> int:
    """
    Wrapper that builds a hashable cache key and calls the cached flood-fill.
    """
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    start_t = (start["x"], start["y"])
    occupied = _occupied_after_move(gs)
    occupied_key = tuple(sorted(occupied))  # stable, hashable
    return _reachable_space_cached(width, height, occupied_key, start_t)


# ---------------------------- Task 3 Weights ------------------------------
TASK3_WEIGHTS = {
    "w_space": 1.0,           # reachable space
    "w_len_adv": 2.0,         # length advantage
    "w_shorter": 1.5,         # bonus per opponent shorter than you
    "w_local_open": 2.0,      # local openness (free neighbors around head)
    "hazard_in_penalty": 25.0,   # penalty if head is in a hazard
    "hazard_prox_weight": 5.0,   # proximity penalty ~ weight/(1+distance)
}


def _free_neighbors_count(gs: typing.Dict, snake: typing.Dict, pos: typing.Dict) -> int:
    """Count non-blocked adjacent cells (allows stepping onto own tail)."""
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    occupied = _occupied_after_move(gs)
    # allow stepping into your own current tail
    if snake["body"]:
        tail = snake["body"][-1]
        occupied.discard((tail["x"], tail["y"]))

    cnt = 0
    for dv in MOVE_VECTORS.values():
        nx, ny = pos["x"] + dv[0], pos["y"] + dv[1]
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if (nx, ny) in occupied:
            continue
        cnt += 1
    return cnt


def _longest_opponent_length(gs: typing.Dict, you_id: str) -> int:
    best = 0
    for s in gs["board"]["snakes"]:
        if s["id"] == you_id:
            continue
        best = max(best, len(s["body"]))
    return best


def _evaluate(gs: typing.Dict, you_id: str) -> float:
    """
    Task 3 evaluation (builds on Task 1):
      score = w_len_adv*(your_len - longest_opp_len)
            + w_space*reachable_space
            + w_local_open*local_openness
            + w_shorter*count_shorter_opponents
            + hazard_penalty
    """
    # locate you
    you = None
    for s in gs["board"]["snakes"]:
        if s["id"] == you_id:
            you = s
            break
    if you is None:
        return -1e9

    head = you["body"][0]
    your_len = len(you["body"])

    # Core features
    length_advantage = your_len - _longest_opponent_length(gs, you_id)
    space = _reachable_space(gs, head)
    local_open = _free_neighbors_count(gs, you, head)
    shorter_count = sum(1 for s in gs["board"]["snakes"]
                        if s["id"] != you_id and len(s["body"]) < your_len)

    # Hazards
    hazards = {(h["x"], h["y"]) for h in gs["board"].get("hazards", [])}
    in_hazard = (head["x"], head["y"]) in hazards
    if hazards:
        # Manhattan distance to nearest hazard
        nearest = min(abs(head["x"] - x) + abs(head["y"] - y) for (x, y) in hazards)
    else:
        nearest = 99

    hazard_penalty = (-TASK3_WEIGHTS["hazard_in_penalty"] if in_hazard else 0.0) \
                     - (TASK3_WEIGHTS["hazard_prox_weight"] / (1 + nearest))

    # Weighted sum
    score = (
        TASK3_WEIGHTS["w_len_adv"] * float(length_advantage)
        + TASK3_WEIGHTS["w_space"] * float(space)
        + TASK3_WEIGHTS["w_local_open"] * float(local_open)
        + TASK3_WEIGHTS["w_shorter"] * float(shorter_count)
        + float(hazard_penalty)
    )
    return score


def _is_dead(gs: typing.Dict, you_id: str) -> bool:
    return all(s["id"] != you_id for s in gs["board"]["snakes"])


def _pick_minimax_opponent(gs: typing.Dict, you_id: str) -> typing.Optional[str]:
    """Pick the longest rival to act as the minimizing player."""
    opp_id = None
    best_len = -1
    for s in gs["board"]["snakes"]:
        if s["id"] == you_id:
            continue
        L = len(s["body"])
        if L > best_len:
            best_len = L
            opp_id = s["id"]
    return opp_id


# ------------------ Task 5: Dead-end avoidance (BFS) ------------------

def _trap_threshold(snake_len: int, board_cells: int) -> int:
    """
    Region-size threshold below which a move is considered 'trap-prone'.
    - Absolute floor: 3 (room to turn)
    - Relative component: 60% of current length
    """
    abs_min = 3
    rel_min = int(0.6 * snake_len)
    # Keep threshold reasonable on tiny boards
    return max(abs_min, rel_min)


def _region_size_after_move(gs: typing.Dict, snake_idx: int, move: str) -> int:
    """Simulate a single move for snake_idx and return reachable space from new head."""
    snakes = gs["board"]["snakes"]
    snake = snakes[snake_idx]
    sim = _simulate_step(gs, {snake["id"]: move})
    head_after = sim["board"]["snakes"][snake_idx]["body"][0]
    return _reachable_space(sim, head_after)


# -------------------------- Task 2: Alpha–Beta ----------------------------

def _order_moves(gs: typing.Dict, snake_idx: int) -> list[str]:
    """
    Move ordering with Task 5 trap-avoidance:
      1) Compute BFS region size after each legal move
      2) Filter out 'tiny-region' moves if at least one non-tiny option exists
      3) Sort remaining moves by region size (descending) to improve pruning
    """
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    board_cells = width * height
    snakes = gs["board"]["snakes"]
    snake = snakes[snake_idx]

    legal = _legal_moves_for_snake(gs, snake_idx)

    scored: list[tuple[int, str]] = []
    for m in legal:
        region = _region_size_after_move(gs, snake_idx, m)
        scored.append((region, m))

    # Task 5: trap-aware filtering
    thr = _trap_threshold(len(snake["body"]), board_cells)
    good = [(r, m) for (r, m) in scored if r >= thr]

    # If at least one move meets the threshold, keep only those; otherwise keep all
    use = good if good else scored

    # Sort by region size (desc)
    use.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in use]


def _minimax_ab(gs: typing.Dict, depth: int, you_id: str, turn: str,
                alpha: float, beta: float) -> tuple[float, typing.Optional[str]]:
    """
    Alpha–Beta pruning variant of minimax (two-player: you vs. longest opponent).
    Returns (score, best_move_for_current_turn).
    """
    if depth == 0 or _is_dead(gs, you_id):
        return _evaluate(gs, you_id), None

    snakes = gs["board"]["snakes"]
    you_idx = next(i for i, s in enumerate(snakes) if s["id"] == you_id)
    opp_id = _pick_minimax_opponent(gs, you_id)

    if opp_id is None:
        # Solo board: just maximize your next steps
        best_score = -1e18
        best_move = None
        for m in _order_moves(gs, you_idx):
            new_state = _simulate_step(gs, {snakes[you_idx]["id"]: m})
            sc, _ = _minimax_ab(new_state, depth-1, you_id, "you", alpha, beta)
            if sc > best_score:
                best_score, best_move = sc, m
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        if best_move is None:
            return -1e9, None
        return best_score, best_move

    opp_idx = next(i for i, s in enumerate(snakes) if s["id"] == opp_id)

    if turn == "you":
        legal = _order_moves(gs, you_idx)
        if not legal:
            return -1e9, None  # no move → lose
        best_score = -1e18
        best_move = random.choice(legal)
        for m in legal:
            new_state = _simulate_step(gs, {snakes[you_idx]["id"]: m})
            sc, _ = _minimax_ab(new_state, depth, you_id, "opp", alpha, beta)
            if sc > best_score:
                best_score, best_move = sc, m
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # prune
        return best_score, best_move
    else:
        legal = _order_moves(gs, opp_idx)
        if not legal:
            # opponent stuck → good for us
            return _evaluate(gs, you_id), None
        worst_score = 1e18
        worst_move = random.choice(legal)
        for m in legal:
            new_state = _simulate_step(gs, {snakes[opp_idx]["id"]: m})
            sc, _ = _minimax_ab(new_state, depth-1, you_id, "you", alpha, beta)
            if sc < worst_score:
                worst_score, worst_move = sc, m
            beta = min(beta, worst_score)
            if beta <= alpha:
                break  # prune
        return worst_score, worst_move


# ------------------------------ Move handler ------------------------------

def move(game_state: typing.Dict) -> typing.Dict:
    """
    Choose next move using Alpha–Beta (Task 2) with Task 3 evaluation,
    Task 4 flood-fill caching, and Task 5 trap-aware move ordering.
    """
    you_id = game_state["you"]["id"]
    # Depth-3 search (you → opp → you → opp), fast thanks to alpha–beta + caching
    score, best_move = _minimax_ab(
        game_state, depth=3, you_id=you_id, turn="you", alpha=-1e18, beta=1e18
    )

    # Fallback if something odd happens
    if not best_move:
        is_move_safe = {"up": True, "down": True, "left": True, "right": True}
        my_head = game_state["you"]["body"][0]
        my_neck = game_state["you"]["body"][1] if len(game_state["you"]["body"]) > 1 else None
        if my_neck:
            if my_neck["x"] < my_head["x"]:
                is_move_safe["left"] = False
            elif my_neck["x"] > my_head["x"]:
                is_move_safe["right"] = False
            elif my_neck["y"] < my_head["y"]:
                is_move_safe["down"] = False
            elif my_neck["y"] > my_head["y"]:
                is_move_safe["up"] = False
        safe_moves = [m for m, ok in is_move_safe.items() if ok]
        best_move = random.choice(safe_moves) if safe_moves else "down"

    print(f"MOVE {game_state['turn']}: {best_move} (alpha-beta score {score:.1f})")
    return {"move": best_move}


# ------------------------------- Entrypoint -------------------------------

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
