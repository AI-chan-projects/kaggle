import requests
import random
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

# 유틸리티 함수 (객체 디버깅용)
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