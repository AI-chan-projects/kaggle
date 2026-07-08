from integration import stub

# init
stub.network_check()
env = stub.set_up()

agents = ["random", "random"]
env.reset()
print("현재 턴:", env.state[0].status)
result = env.step(agents)