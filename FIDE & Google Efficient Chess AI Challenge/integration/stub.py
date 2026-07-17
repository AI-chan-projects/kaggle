import requests
import random
import json
import pathlib
import pyspiel
from kaggle_environments import make

class Constants:
    PIECE_VALUES = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    
    MOVE_SCORES = {
        "CENTER": {"e4", "d4", "e5", "d5", "c4", "f4", "c5", "f5"},
        "KNIGHT": {"Nf3", "Nc3", "Nf6", "Nc6"},
        "BISHOP": {"Bc4", "Bb5", "Bc5", "Bb4"}
    }

class Environment:
    @staticmethod
    def check_network():
        print("인터넷 연결 확인 중...")
        try:
            if requests.get('http://www.google.com', timeout=10).ok:
                print("인터넷 연결 정상 확인")
            else:
                print("인터넷 연결을 확인해 주세요")
        except Exception:
            print("인터넷 연결 실패")

    @staticmethod
    def setup():
        return make("open_spiel_chess", configuration={"includeLegalActions": True}, debug=True)

class FENParser:
    @staticmethod
    def parse(fen):
        board = []
        placement = fen.split()[0]
        for row in placement.split("/"):
            board_row = []
            for ch in row:
                if ch.isdigit():
                    board_row.extend(["."] * int(ch))
                else:
                    board_row.append(ch)
            board.append(board_row)
        return board

class BoardEvaluator:
    @staticmethod
    def get_material_score(board):
        return sum(Constants.PIECE_VALUES.get(piece, 0) for row in board for piece in row)

class MoveEvaluator:
    @staticmethod
    def evaluate(move):
        score = 0
        if "#" in move: score += 100000
        elif "+" in move: score += 10000
        
        if "=Q" in move: score += 9000
        elif "=" in move: score += 8000
        
        if "x" in move: score += 1000
        if move in ("O-O", "O-O-O"): score += 500
        
        if move in Constants.MOVE_SCORES["CENTER"]: score += 100
        if move in Constants.MOVE_SCORES["KNIGHT"]: score += 50
        if move in Constants.MOVE_SCORES["BISHOP"]: score += 30
        
        return score

class ActionSelector:
    @staticmethod
    def choose(legal_actions, legal_strings):
        best_score = -1
        candidates = []

        for action, move in zip(legal_actions, legal_strings):
            score = MoveEvaluator.evaluate(move)
            if score > best_score:
                best_score = score
                candidates = [action]
            elif score == best_score:
                candidates.append(action)

        return random.choice(candidates)


class OpeningBook:
    """
    kaggle_environments가 내장한 openings.jsonl(2수짜리 짧은 오프닝 목록)을 그대로 활용.
    커버리지가 깊지 않아서(초반 1수 정도) 결정적인 해결책은 아니지만,
    책에 있는 동안은 탐색 없이 공짜로 빠르게 둘 수 있음.
    """
    _entries = None

    @classmethod
    def _load(cls):
        if cls._entries is None:
            import kaggle_environments
            path = (
                pathlib.Path(kaggle_environments.__file__).parent
                / "envs" / "open_spiel_env" / "games" / "chess" / "openings.jsonl"
            )
            entries = []
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(json.loads(line))
            cls._entries = entries
        return cls._entries

    @classmethod
    def lookup(cls, history):
        """history: 지금까지 실제로 둔 action(int) 리스트. 매칭되는 다음 수가 있으면 반환, 없으면 None."""
        candidates = []
        for entry in cls._load():
            actions = entry.get("initialActions", [])
            if len(actions) > len(history) and actions[: len(history)] == list(history):
                candidates.append(actions[len(history)])
        return random.choice(candidates) if candidates else None


class MinimaxSelector:
    """
    Alpha-Beta 가지치기를 적용한 미니맥스 탐색.

    개선 사항:
    1) mate-distance 스코어링: 체크메이트를 단순 ±100000이 아니라 "몇 수 만에 나느냐"로 차등을 둬서,
       이길 땐 가장 빠른 메이트를, 질 수밖에 없는 상황이면 최대한 버티는 수를 선호하게 만듦.
       -> "어차피 진다면 아무 수나 반복"하던 문제 완화.
    2) 반복(3수 동형) 회피: 최상위(지금 실제로 둘 수) 레벨에서 동점 후보가 여러 개면,
       최근 실제로 나왔던 국면(FEN)으로 돌아가는 수는 후순위로 미룸.
       -> 같은 자리를 왔다갔다하는 루프를 직접적으로 줄임.
    """

    MATE_BASE = 100000

    @staticmethod
    def _evaluate(state, ply_from_root):
        if state.is_terminal():
            white_return = state.returns()[1]  # 0=Black, 1=White
            # 이기는 쪽엔 ply_from_root가 작을수록(빠른 메이트) 높은 점수,
            # 지는 쪽엔 ply_from_root가 클수록(오래 버팀) 덜 나쁜 점수.
            return white_return * (MinimaxSelector.MATE_BASE - ply_from_root)
        board = FENParser.parse(state.observation_string())
        return BoardEvaluator.get_material_score(board)

    @staticmethod
    def _search(state, depth, alpha, beta, maximizing, ply_from_root):
        if depth == 0 or state.is_terminal():
            return MinimaxSelector._evaluate(state, ply_from_root)

        if maximizing:
            value = float("-inf")
            for a in state.legal_actions():
                child = state.clone()
                child.apply_action(a)
                value = max(value, MinimaxSelector._search(child, depth - 1, alpha, beta, False, ply_from_root + 1))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # 가지치기
            return value
        else:
            value = float("inf")
            for a in state.legal_actions():
                child = state.clone()
                child.apply_action(a)
                value = min(value, MinimaxSelector._search(child, depth - 1, alpha, beta, True, ply_from_root + 1))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    @staticmethod
    def choose_from_state(state, depth=3, recent_fens=None):
        """
        state: pyspiel.State (이미 deserialize된 것)
        recent_fens: 실제 게임에서 최근에 나왔던 FEN들의 집합/리스트 (반복 회피용, 없으면 무시)
        """
        maximizing = state.current_player() == 1  # White(1)면 기물점수를 최대화하는 쪽
        recent_fens = set(recent_fens or [])

        best_score = float("-inf") if maximizing else float("inf")
        best_actions = []
        for a in state.legal_actions():
            child = state.clone()
            child.apply_action(a)
            score = MinimaxSelector._search(child, depth - 1, float("-inf"), float("inf"), not maximizing, 1)
            if maximizing:
                if score > best_score:
                    best_score, best_actions = score, [a]
                elif score == best_score:
                    best_actions.append(a)
            else:
                if score < best_score:
                    best_score, best_actions = score, [a]
                elif score == best_score:
                    best_actions.append(a)

        if len(best_actions) > 1 and recent_fens:
            # 동점 후보 중, 두었을 때 최근 실제 국면으로 되돌아가지 않는 수를 우선
            non_repeating = []
            for a in best_actions:
                child = state.clone()
                child.apply_action(a)
                if child.observation_string() not in recent_fens:
                    non_repeating.append(a)
            if non_repeating:
                best_actions = non_repeating

        return random.choice(best_actions)

    @staticmethod
    def choose(serialized_game_and_state, depth=3, recent_fens=None):
        """serializedGameAndState(observation의 필드)를 받아 depth수 앞을 읽고 최선의 액션(int)을 반환."""
        _game, state = pyspiel.deserialize_game_and_state(serialized_game_and_state)
        return MinimaxSelector.choose_from_state(state, depth=depth, recent_fens=recent_fens)

def inspect_object(obj, depth=0, max_depth=4):
    if depth > max_depth: return
    items = obj.__dict__.items() if hasattr(obj, '__dict__') else (obj.items() if isinstance(obj, dict) else None)
    if items is None:
        print("  " * depth + f"{type(obj)}")
        return
    for key, value in items:
        print("  " * depth + f"{key}: {type(value)}")
        if hasattr(value, '__dict__') or isinstance(value, dict):
            inspect_object(value, depth + 1, max_depth)