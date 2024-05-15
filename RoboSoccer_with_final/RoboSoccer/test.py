import threading
import socketComm
import time

socketThread=threading.Thread(target=socketComm.main)

socketThread.start()

while True:
    if(socketComm.Agents):
        if(len(socketComm.Agents.items())>= 1):
            a, *actions = input().split(' ')
            print(a, actions)
            agent_id = int(a)
            socketComm.Agents[agent_id].SendAction(actions[0])
        else:
            print("Waiting for Agents to Connect, Connected Agents",list(socketComm.Agents.keys()))
    else:
        print(".",end="")
        # for i, agent in socketComm.Agents.items():
        #     action=list(input())
        #     print(action)
        #     agent.SendAction(action)
        
    time.sleep(0.5)