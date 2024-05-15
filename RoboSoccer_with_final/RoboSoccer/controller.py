import pygame
import time
import threading
import socketComm
# import alokcv


# thread_third=threading.Thread(target=alokcv.main_function)
# thread_third.start()

socketThread=threading.Thread(target=socketComm.main)

socketThread.start()


def get_action(axis_values,JoystickNumber):
    action = [0, 0]
    for i in range(2):
        if axis_values[i] > 0.5:
            action[i] = "1"  # forward or right
        elif axis_values[i] < -0.5:
            action[i] = '2' # backward or left
        else:
            action[i] = '0'  # no movement
    

    # Print the determined action
    if(action[0]=='2'):
        action[0]='1'
    elif(action[0]=='1'):
        action[0]='2'
    
    print("Action->",JoystickNumber,":" ,action)
    return [action[0],'0',action[1]]

def main():
    pygame.init()
    pygame.joystick.init()

    # Check if there's at least one joystick/gamepad connected
    if pygame.joystick.get_count() == 0:
        print("No joystick/gamepad found.")
        return

    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick1=pygame.joystick.Joystick(1)
    joystick.init()
    joystick1.init()
    action1=action2=['0','0','0']

    print("Joystick Name:", joystick.get_name())
    print("Number of Axes:", joystick.get_numaxes())
    print("Number of Buttons:", joystick.get_numbuttons())
    print("Number of Hats:", joystick.get_numhats())

    try:
        while True:
            pygame.event.pump()
            axis_values = [joystick.get_axis(2), joystick.get_axis(3)]
            action1 = get_action(axis_values,1)
            
            axis_values = [joystick1.get_axis(1), joystick1.get_axis(0)]
            action2 = get_action(axis_values,2)


            if(socketComm.Agents):
                if(len(socketComm.Agents.items())== 2):
                    socketComm.Agents[8].SendAction(action1)
                    socketComm.Agents[3].SendAction(action2)
                else:
                    print("Waiting for Agents to Connect, Connected Agents",list(socketComm.Agents.keys()))
            # else:
                # print(".",end="")
                
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Exiting...")
        pygame.quit()

if __name__ == "__main__":
    main()