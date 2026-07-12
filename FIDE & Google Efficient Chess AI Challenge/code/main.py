import random
import time
import os
from integration import stub
import json

# 1. 초기화
env = stub.set_up()
env.reset()

# 2. 첫 번째 step으로 게임 시작
agent_states = env.step([None, None])

def clear_screen():
    # Windows는 'cls', macOS/Linux는 'clear' 명령을 사용
    os.system('cls' if os.name == 'nt' else 'clear')

print("게임 시작!")

while not env.done:
    # 화면 초기화 (깔끔한 터미널 출력을 위해)
    clear_screen()
    
    # 현재 플레이어 인덱스 확인
    current_player_idx = agent_states[0].observation.get('currentPlayer', 0)
    player_name = "백(White)" if current_player_idx == 1 else "흑(Black)"
    
    # 렌더링 (보드판 출력)
    print(f"--- 현재 턴: {player_name} (idx: {current_player_idx}) ---")
    print(env.render(mode="ansi"))
    
    current_agent_state = agent_states[current_player_idx]
    
    # 가능한 액션(정수 인덱스)
    obs = current_agent_state.observation

    # 주요 관찰(observation) 정보 출력 및 파싱
    # print(f"현재 관찰(observation): {obs.keys()}")

    # print(type(obs["serializedGameAndState"]))
    # print(obs["serializedGameAndState"][:300])
    # print(obs["legalActionStrings"])
    # print(obs["observationString"])

    moves = obs.get('legalActions', [])
    
    
    if not moves:
        print("더 이상 둘 수 있는 수가 없습니다.")
        break
        
    # 3. 액션 선택 및 딕셔너리 포맷 적용
    action_idx = random.choice(moves)
    action_payload = {"submission": action_idx}
    
    actions = [None, None]
    actions[current_player_idx] = action_payload
    
    # 4. 환경 업데이트
    agent_states = env.step(actions)
    
    print(f"Player {current_player_idx} ({player_name})가 액션 '{action_idx}'를 수행했습니다.")
    
    # 턴 사이의 간격 (너무 빨리 지나가면 확인이 어려우므로)
    time.sleep(10)

# 게임 종료 후 마지막 상태 출력
clear_screen()
print(env.render(mode="ansi"))
print("게임 종료!")

json_data = env.toJSON()
json_data = json.dumps(json_data, indent=4)
with open("game_result.json", "w") as f:
    f.write(json_data)