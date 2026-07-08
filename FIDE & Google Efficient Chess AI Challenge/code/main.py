from integration import stub
import time

# init
stub.network_check()
env = stub.set_up()

print(f"환경 이름: {env.name}")
print(f"설정 정보: {env.configuration}")
agents = ["random", "random"]

env.reset()

# 1. 사용할 에이전트 정의 ("random" 등)
agents_list = ["random", "random"]

# 2. 루프를 돌며 step 진행
while not env.done:
    # 렌더링
    print(env.render(mode="ansi"))
    
    # env.step에 [None, None]을 주면, 환경 내부에서 
    # 현재 턴에 맞는 에이전트(random)를 찾아 자동으로 액션을 실행합니다.
    # 즉, 우리가 직접 act()를 호출할 필요가 없습니다.
    
    # 현재 환경의 observation을 얻고
    obs = env.state[0].observation
    
    # step() 호출 시 에이전트들의 행동 리스트를 전달
    # random 에이전트들을 이미 등록했으므로 None을 전달해도 환경이 알아서 처리합니다.
    # 만약 직접 act()를 안 쓴다면 아래처럼 간단히 할 수 있습니다.
    env.step(agents)
    
    time.sleep(0.5)

print("게임 종료!")