import requests
from kaggle_environments import make
import random

CENTER = {
    "e4","d4","e5","d5",
    "c4","f4","c5","f5"
}

GOOD_KNIGHT = {
    "Nf3","Nc3",
    "Nf6","Nc6"
}

GOOD_BISHOP = {
    "Bc4","Bb5",
    "Bc5","Bb4"
}

PIECE_VALUE = {

    'P':1,

    'N':3,

    'B':3,

    'R':5,

    'Q':9,

    'K':0,

    'p':-1,

    'n':-3,

    'b':-3,

    'r':-5,

    'q':-9,

    'k':0

}

def network_check():
    # first let's make sure you have internet enabled
    print("인터넷 연결 확인 중")
    if requests.get('http://www.google.com',timeout=10).ok == True :
        return print("인터넷 연결 정상 확인")
    else : 
        return print("인터넷 연결을 확인해 주세요")

def set_up():
    # Now let's set up the chess environment!
    env = make(
        "open_spiel_chess",
        configuration={
            "includeLegalActions" : True,
        },
        debug=True
    )
    # print(env.__dict__)
    return env

# 평가 함수
def evaluate_move(move):

    score = 0

    if "#" in move:
        score += 100000

    if "+" in move:
        score += 10000

    if "=Q" in move:
        score += 9000

    elif "=" in move:
        score += 8000

    if "x" in move:
        score += 1000

    if move in ("O-O", "O-O-O"):
        score += 500

    if move in CENTER:
        score += 100

    if move in GOOD_KNIGHT:
        score += 50

    if move in GOOD_BISHOP:
        score += 30

    return score

# 선택 함수
def choose_action(legal_actions, legal_strings):

    best_score = -1
    candidates = []

    for action, move in zip(legal_actions, legal_strings):

        score = evaluate_move(move)

        if score > best_score:
            best_score = score
            candidates = [action]

        elif score == best_score:
            candidates.append(action)

    return random.choice(candidates)


# FEN parse
def parse_fen(fen):

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

# score board
def material_score(board):

    score = 0

    for row in board:

        for piece in row:

            score += PIECE_VALUE.get(piece,0)

    return score

def inspect_object(obj, depth=0, max_depth=4):
    if depth > max_depth:
        return
    
    # 딕셔너리처럼 접근 가능한지 확인
    if hasattr(obj, '__dict__'):
        items = obj.__dict__.items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        print("  " * depth + f"{type(obj)}")
        return

    for key, value in items:
        # 주요 데이터 속성 출력
        print("  " * depth + f"{key}: {type(value)}")
        # 재귀적으로 더 깊이 탐색
        if hasattr(value, '__dict__') or isinstance(value, dict):
            inspect_object(value, depth + 1, max_depth)

