import threading
import Camera
import socketComm
import time

mainThread=threading.Thread(target=Camera.main_function,args=[socketComm.Agents])
socketThread=threading.Thread(target=socketComm.main)

socketThread.start()


mainThreadRunning=False
while True:
    if(socketComm.Agents):
        if(len(socketComm.Agents.items())==2 and not mainThreadRunning):
            mainThread = threading.Thread(target=Camera.main_function,args=[socketComm.Agents])
            mainThreadRunning=True
            mainThread.start()
        elif mainThreadRunning:
            mainThread.join()
            mainThreadRunning=False
        else:
            print("Waiting for Agents to Connect, Connected Agents",list(socketComm.Agents.keys()))
    else:
        print(".",end="")
        # for i, agent in socketComm.Agents.items():
        #     action=list(input())
        #     print(action)
        #     agent.SendAction(action)
        
    time.sleep(0.5)