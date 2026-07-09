import random
from integration import stub

# 1. 초기화
env = stub.set_up()
env.reset()

# 2. 첫 번째 step으로 게임 시작
agent_states = env.step([None, None])

while not env.done:
    # 현재 플레이어 인덱스 확인
    current_player_idx = agent_states[0].observation.get('currentPlayer', 0)
    current_agent_state = agent_states[current_player_idx]
    
    # 가능한 액션(정수 인덱스)
    moves = current_agent_state.observation.get('legalActions', [])
    
    if not moves:
        break
        
    # 3. 액션 선택 및 딕셔너리 포맷 적용
    action_idx = random.choice(moves)
    
    # 핵심 수정: 액션을 {"submission": action_idx} 형태로 감싸기
    action_payload = {"submission": action_idx}
    
    actions = [None, None]
    actions[current_player_idx] = action_payload
    
    # 4. 환경 업데이트
    agent_states = env.step(actions)
    
    print(f"Player {current_player_idx}가 액션 {action_idx}를 수행했습니다.")

print("게임 종료!")