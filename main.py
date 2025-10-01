# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____/______>___|__(______/__|__\_____>

# Tasks integrated:
#  - Task 1: Base eval (space + length)
#  - Task 2: MiniMax + Alpha–Beta
#  - Task 3: Rich eval (local openness, hazards, #shorter)
#  - Task 4: Flood-Fill caching
#  - Task 5: Trap-aware ordering (BFS region threshold)
#  - Task 6: Rules-faithful full-turn sim (food, growth, hazards, head-to-head)

import random
import typing
from copy import deepcopy
from collections import deque
from functools import lru_cache


# ------------------------ Appearance / API handlers ------------------------

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "LudaSnake",  # TODO: your Battlesnake username
        "color": "#16a085",
        "head": "beluga",
        "tail": "bolt",
    }


def start(game_state: typing.Dict):
    print("GAME START")


def end(game_state: typing.Dict):
    print("GAME OVER\n")


# ------------------------------ Core helpers ------------------------------

MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


def _in_bounds(pt: typing.Dict, width: int, height: int) -> bool:
    return 0 <= pt["x"] < width and 0 <= pt["y"] < height


def _add(a: typing.Dict, dv: typing.Tuple[int, int]) -> typing.Dict:
    return {"x": a["x"] + dv[0], "y": a["y"] + dv[1]}


def _copy_state(gs: typing.Dict) -> typing.Dict:
    return deepcopy(gs)


def _ruleset_settings(gs: typing.Dict) -> tuple[int, int]:
    """
    Returns (max_health, hazard_damage_extra) from ruleset settings when available.
    Defaults to (100, 14). (Standard maps often have 0 hazards; Royale uses extra hazard damage.)
    """
    max_health = 100
    hz = 14
    try:
        rs = gs.get("game", {}).get("ruleset", {}).get("settings", {})
        hz = int(rs.get("hazardDamagePerTurn", hz))
    except Exception:
        pass
    return max_health, hz


def _occupied_after_move(gs: typing.Dict) -> set[tuple]:
    occ: set[tuple] = set()
    for s in gs["board"]["snakes"]:
        for seg in s["body"]:
            occ.add((seg["x"], seg["y"]))
    return occ


def _legal_moves_for_snake(gs: typing.Dict, snake_idx: int) -> list[str]:
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    snakes = gs["board"]["snakes"]
    snake = snakes[snake_idx]
    head = snake["body"][0]
    tail = snake["body"][-1]

    occupied = _occupied_after_move(gs)
    # allow stepping onto your current tail; whether it actually moves is handled in the full simulator
    occupied.discard((tail["x"], tail["y"]))

    moves: list[str] = []
    for m, dv in MOVE_VECTORS.items():
        nh = _add(head, dv)
        if not _in_bounds(nh, width, height):
            continue
        if (nh["x"], nh["y"]) in occupied:
            continue
        if len(snake["body"]) >= 2:
            neck = snake["body"][1]
            if nh["x"] == neck["x"] and nh["y"] == neck["y"]:
                continue
        moves.append(m)
    return moves


# -------------------- Flood-Fill (Task 4 caching) --------------------

@lru_cache(maxsize=8192)
def _reachable_space_cached(width: int, height: int,
                            occupied_key: tuple, start_t: tuple) -> int:
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
            if (nx, ny) in occupied or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    return count


def _reachable_space(gs: typing.Dict, start: typing.Dict) -> int:
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    start_t = (start["x"], start["y"])
    occupied_key = tuple(sorted(_occupied_after_move(gs)))
    return _reachable_space_cached(width, height, occupied_key, start_t)


# ---------------------------- Task 3 Weights ------------------------------
TASK3_WEIGHTS = {
    "w_space": 1.0,
    "w_len_adv": 2.0,
    "w_shorter": 1.5,
    "w_local_open": 2.0,
    "hazard_in_penalty": 25.0,
    "hazard_prox_weight": 5.0,
    "w_health": 0.05,

    # NEW: food attraction (base pull + hunger-weighted pull)
    "w_food_base": 0.8,  # always prefer getting closer to food a bit
    "w_food_hunger": 2.5,  # ramps up as health drops
}


# ---------------------------- Food utilities -----------------------------

def _nearest_food_distance(gs: typing.Dict, pos: typing.Dict) -> typing.Optional[int]:
    foods = gs["board"].get("food", [])
    if not foods:
        return None
    px, py = pos["x"], pos["y"]
    return min(abs(px - f["x"]) + abs(py - f["y"]) for f in foods)


# ------------- Eval: space + openness + hazards + food attraction ---------

def _free_neighbors_count(gs: typing.Dict, snake: typing.Dict, pos: typing.Dict) -> int:
    width = gs["board"]["width"]
    height = gs["board"]["height"]
    occupied = _occupied_after_move(gs)
    if snake["body"]:
        tail = snake["body"][-1]
        occupied.discard((tail["x"], tail["y"]))
    cnt = 0
    for dv in MOVE_VECTORS.values():
        nx, ny = pos["x"] + dv[0], pos["y"] + dv[1]
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in occupied:
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
    you = next((s for s in gs["board"]["snakes"] if s["id"] == you_id), None)
    if you is None:
        return -1e9

    max_health, _ = _ruleset_settings(gs)
    head = you["body"][0]
    your_len = len(you["body"])
    your_health = you.get("health", max_health)

    length_advantage = your_len - _longest_opponent_length(gs, you_id)
    space = _reachable_space(gs, head)
    local_open = _free_neighbors_count(gs, you, head)
    shorter_count = sum(1 for s in gs["board"]["snakes"] if s["id"] != you_id and len(s["body"]) < your_len)

    hazards = {(h["x"], h["y"]) for h in gs["board"].get("hazards", [])}
    in_hazard = (head["x"], head["y"]) in hazards
    nearest_hazard = min((abs(head["x"] - x) + abs(head["y"] - y) for (x, y) in hazards), default=99)
    hazard_penalty = (-TASK3_WEIGHTS["hazard_in_penalty"] if in_hazard else 0.0) \
                     - (TASK3_WEIGHTS["hazard_prox_weight"] / (1 + nearest_hazard))

    # NEW: food attraction (always on, stronger when hungry)
    food_term = 0.0
    fd = _nearest_food_distance(gs, head)
    if fd is not None:
        hunger = 1.0 - (your_health / max_health)
        # negative distance because closer is better
        food_term = TASK3_WEIGHTS["w_food_base"] * (-fd) + TASK3_WEIGHTS["w_food_hunger"] * hunger * (-fd)

    score = (
            TASK3_WEIGHTS["w_len_adv"] * float(length_advantage)
            + TASK3_WEIGHTS["w_space"] * float(space)
            + TASK3_WEIGHTS["w_local_open"] * float(local_open)
            + TASK3_WEIGHTS["w_shorter"] * float(shorter_count)
            + TASK3_WEIGHTS["w_health"] * float(your_health)
            + float(hazard_penalty)
            + float(food_term)
    )
    return score


def _is_dead(gs: typing.Dict, you_id: str) -> bool:
    return all(s["id"] != you_id for s in gs["board"]["snakes"])


def _pick_minimax_opponent(gs: typing.Dict, you_id: str) -> typing.Optional[str]:
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


# ------------------ Head-to-head danger map (Task 6) ------------------

def _opponent_head_danger(gs: typing.Dict, you_id: str) -> dict[tuple, int]:
    board = gs["board"]
    snakes = board["snakes"]
    width, height = board["width"], board["height"]
    occ = _occupied_after_move(gs)

    danger: dict[tuple, int] = {}
    for s in snakes:
        if s["id"] == you_id:
            continue
        head = s["body"][0]
        tail = s["body"][-1]
        occ_minus_tail = set(occ)
        occ_minus_tail.discard((tail["x"], tail["y"]))
        for dv in MOVE_VECTORS.values():
            nh = (head["x"] + dv[0], head["y"] + dv[1])
            if not (0 <= nh[0] < width and 0 <= nh[1] < height):
                continue
            if nh in occ_minus_tail:
                continue
            danger[nh] = max(danger.get(nh, 0), len(s["body"]))
    return danger


# --------------- Task 5: Dead-end avoidance (BFS threshold) ---------------

def _trap_threshold(snake_len: int, board_cells: int) -> int:
    return max(3, int(0.6 * snake_len))


def _simulate_partial_move(gs: typing.Dict, chosen: dict[str, str]) -> typing.Dict:
    ns = _copy_state(gs)
    snakes = ns["board"]["snakes"]
    id_to_idx = {s["id"]: i for i, s in enumerate(snakes)}
    for sid, mv in chosen.items():
        i = id_to_idx.get(sid)
        if i is None:
            continue
        dv = MOVE_VECTORS[mv]
        snake = snakes[i]
        new_head = _add(snake["body"][0], dv)
        snake["body"].insert(0, new_head)
        snake["body"].pop()
    return ns


def _region_size_after_move(gs: typing.Dict, snake_idx: int, move: str) -> int:
    snakes = gs["board"]["snakes"]
    snake = snakes[snake_idx]
    sim = _simulate_partial_move(gs, {snake["id"]: move})
    head_after = next(s for s in sim["board"]["snakes"] if s["id"] == snake["id"])["body"][0]
    return _reachable_space(sim, head_after)


# ---------------- Task 6: Full-turn sim (food, growth, hazards, H2H) ----------------

def _simulate_full_turn(gs: typing.Dict, moves: dict[str, str], you_id: str) -> typing.Dict:
    ns = _copy_state(gs)
    board = ns["board"]
    snakes = board["snakes"]
    width, height = board["width"], board["height"]
    foods = {(f["x"], f["y"]) for f in board.get("food", [])}
    hazards = {(h["x"], h["y"]) for h in board.get("hazards", [])}
    max_health, hazard_extra = _ruleset_settings(ns)

    id_to_idx = {s["id"]: i for i, s in enumerate(snakes)}

    # 1) apply moves: add head, pop tail, health -1 (and hazard extra)
    for sid, mv in moves.items():
        idx = id_to_idx.get(sid)
        if idx is None:
            continue
        s = snakes[idx]
        dv = MOVE_VECTORS[mv]
        new_head = {"x": s["body"][0]["x"] + dv[0], "y": s["body"][0]["y"] + dv[1]}
        s["body"].insert(0, new_head)
        s["body"].pop()
        s["health"] = s.get("health", max_health) - 1
        if (new_head["x"], new_head["y"]) in hazards:
            s["health"] -= hazard_extra

    # 2) food: reset health to max + grow (tail does not move that turn)
    new_foods = set(foods)
    for sid in moves.keys():
        idx = id_to_idx.get(sid)
        if idx is None:
            continue
        s = snakes[idx]
        hx, hy = s["body"][0]["x"], s["body"][0]["y"]
        if (hx, hy) in new_foods:
            s["health"] = max_health
            tail = s["body"][-1]
            s["body"].append({"x": tail["x"], "y": tail["y"]})
            new_foods.discard((hx, hy))

    # 3) eliminate: OOB, <=0 health, self/body, head-to-head
    eliminated_ids: set[str] = set()
    # bounds & health & self
    for s in snakes:
        head = s["body"][0]
        if not _in_bounds(head, width, height):
            eliminated_ids.add(s["id"])
            continue
        if s.get("health", max_health) <= 0:
            eliminated_ids.add(s["id"])
            continue
        if any(head["x"] == seg["x"] and head["y"] == seg["y"] for seg in s["body"][1:]):
            eliminated_ids.add(s["id"])
            continue

    # body collisions (head into other body)
    body_sets = {s["id"]: {(seg["x"], seg["y"]) for seg in s["body"][1:]} for s in snakes}
    for s in snakes:
        if s["id"] in eliminated_ids:
            continue
        hx, hy = s["body"][0]["x"], s["body"][0]["y"]
        for other in snakes:
            if other["id"] == s["id"]:
                continue
            if (hx, hy) in body_sets[other["id"]]:
                eliminated_ids.add(s["id"])
                break

    # head-to-head
    heads_map: dict[tuple, list[typing.Dict]] = {}
    for s in snakes:
        if s["id"] in eliminated_ids:
            continue
        h = (s["body"][0]["x"], s["body"][0]["y"])
        heads_map.setdefault(h, []).append(s)
    for cell, lst in heads_map.items():
        if len(lst) >= 2:
            max_len = max(len(s["body"]) for s in lst)
            survivors = [s for s in lst if len(s["body"]) == max_len]
            if len(survivors) == 1:
                for s in lst:
                    if s is not survivors[0]:
                        eliminated_ids.add(s["id"])
            else:
                for s in lst:
                    eliminated_ids.add(s["id"])

    ns["board"]["snakes"] = [s for s in snakes if s["id"] not in eliminated_ids]
    ns["board"]["food"] = [{"x": x, "y": y} for (x, y) in new_foods]
    you = next((s for s in ns["board"]["snakes"] if s["id"] == you_id), None)
    ns["you"] = you if you else {"id": you_id, "body": [], "health": 0}
    return ns


# -------------------------- Alpha–Beta Minimax ----------------------------

def _order_moves(gs: typing.Dict, snake_idx: int, you_id: str) -> list[str]:
    """
    Ordering = (1) avoid losing head-to-head, (2) avoid tiny regions (unless hungry),
               (3) prefer larger regions, and (4) ALWAYS prefer getting closer to food
                   (stronger when health is low).
    """
    board = gs["board"]
    width, height = board["width"], board["height"]
    board_cells = width * height
    snakes = board["snakes"]
    you = snakes[snake_idx]
    your_len = len(you["body"])
    max_health, _ = _ruleset_settings(gs)
    your_health = you.get("health", max_health)

    legal = _legal_moves_for_snake(gs, snake_idx)
    if not legal:
        return []

    # (1) Head-to-head danger
    danger = _opponent_head_danger(gs, you_id)

    def is_hth_danger(move: str) -> bool:
        dv = MOVE_VECTORS[move]
        pos = (you["body"][0]["x"] + dv[0], you["body"][0]["y"] + dv[1])
        opp_len = danger.get(pos, 0)
        return opp_len >= your_len

    safe_hth = [m for m in legal if not is_hth_danger(m)]
    use1 = safe_hth if safe_hth else legal

    # (2) Region size and (3) trap-aware filtering
    scored_regions: list[tuple[int, str]] = []
    for m in use1:
        region = _region_size_after_move(gs, snake_idx, m)
        scored_regions.append((region, m))

    # Allow narrower corridors when hungry
    thr = _trap_threshold(your_len, board_cells)
    hungry = your_health <= 40
    good = [(r, m) for (r, m) in scored_regions if (r >= thr or hungry)]
    use2 = good if good else scored_regions

    # (4) Always prefer getting closer to food; weight increases with hunger
    foods = [(f["x"], f["y"]) for f in board.get("food", [])]

    def food_dist_after(m: str) -> int:
        if not foods:
            return 999
        dv = MOVE_VECTORS[m]
        hx, hy = you["body"][0]["x"] + dv[0], you["body"][0]["y"] + dv[1]
        return min(abs(hx - fx) + abs(hy - fy) for (fx, fy) in foods)

    # Sort: big region first, then smaller food distance (hungrier -> stronger)
    hunger_factor = 1.0 + 2.0 * (1.0 - your_health / max_health)  # 1..3
    use2.sort(key=lambda t: (-t[0], hunger_factor * food_dist_after(t[1])))
    return [m for _, m in use2]


def _minimax_ab_fullturn(gs: typing.Dict, depth: int, you_id: str,
                         alpha: float, beta: float) -> tuple[float, typing.Optional[str]]:
    if depth == 0 or _is_dead(gs, you_id):
        return _evaluate(gs, you_id), None

    snakes = gs["board"]["snakes"]
    if not any(s["id"] == you_id for s in snakes):
        return -1e9, None

    opp_id = _pick_minimax_opponent(gs, you_id)
    you_idx = next(i for i, s in enumerate(snakes) if s["id"] == you_id)
    your_moves = _order_moves(gs, you_idx, you_id)
    if not your_moves:
        return -1e9, None

    if opp_id is None:
        best_score = -1e18
        best_move = None
        for m in your_moves:
            sim = _simulate_full_turn(gs, {you_id: m}, you_id)
            sc, _ = _minimax_ab_fullturn(sim, depth - 1, you_id, alpha, beta)
            if sc > best_score:
                best_score, best_move = sc, m
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move

    opp_idx = next(i for i, s in enumerate(snakes) if s["id"] == opp_id)
    opp_moves = _order_moves(gs, opp_idx, opp_id)
    if not opp_moves:
        opp_moves = ["up", "down", "left", "right"]

    best_move = your_moves[0]
    best_score = -1e18

    for my in your_moves:
        worst_for_us = 1e18
        for om in opp_moves:
            sim = _simulate_full_turn(gs, {you_id: my, opp_id: om}, you_id)
            sc, _ = _minimax_ab_fullturn(sim, depth - 1, you_id, alpha, beta)
            if sc < worst_for_us:
                worst_for_us = sc
            beta = min(beta, worst_for_us)
            if beta <= alpha:
                break
        if worst_for_us > best_score:
            best_score, best_move = worst_for_us, my
        alpha = max(alpha, best_score)
        if beta <= alpha:
            break
    return best_score, best_move


# ------------------------------ Move handler ------------------------------

def move(game_state: typing.Dict) -> typing.Dict:
    you_id = game_state["you"]["id"]
    score, best_move = _minimax_ab_fullturn(
        game_state, depth=2, you_id=you_id, alpha=-1e18, beta=1e18
    )
    if not best_move:
        # emergency fallback (starter-style)
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

    print(f"MOVE {game_state['turn']}: {best_move} (score {score:.1f})")
    return {"move": best_move}


# ------------------------------- Entrypoint -------------------------------

if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
