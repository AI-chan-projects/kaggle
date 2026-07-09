import requests
from kaggle_environments import make

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
    # print(dir(env))
    return env

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