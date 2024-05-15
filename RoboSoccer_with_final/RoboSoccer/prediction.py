import onnxruntime as ort
import numpy as np
from itertools import chain
import math
import json

ort_sess = ort.InferenceSession('SoccerTwos.onnx')
# sequence = [
#     1, 0, 0, 0, 0, 0, 0, 0.7973102, 0, 0, 0, 1, 0, 0, 0, 0.6304114, 0, 0, 1, 0, 0, 0, 0, 0.8077778,
#     0, 0, 0, 1, 0, 0, 0, 0.5291209, 0, 0, 0, 0, 0, 1, 0, 0.7021397, 0, 0, 0, 1, 0, 0, 0, 0.4737113,
#     0, 0, 1, 0, 0, 0, 0, 0.8468977, 0, 0, 0, 1, 0, 0, 0, 0.44647, 0, 0, 0, 1, 0, 0, 0, 0.7295504,
#     0, 0, 0, 1, 0, 0, 0, 0.4403918, 0, 0, 0, 1, 0, 0, 0, 0.5603973,
#     0, 0, 0, 0, 0, 1, 0, 0.01129425, 0, 0, 0, 0, 0, 1, 0, 0.0202023, 0, 0, 0, 0, 0, 1, 0, 0.0179803
# ]

# print (len(sequence))


# a = [math.pow(math.e, x) for x in [-0.15081546, -2.9515185 , -2.4334497 , -0.32032397, -2.3399467 , -1.7273567 , -0.47313505, -2.7305908 , -1.1654779 ]]

baseSpeed = '100'
stepSpeed = '20'
forwardDelay = '100'
backwardDelay = '100'
leftrotateDelay = '500'
rightrotateDelay = '500'
rotateFactor="0"
resetDelay="50"

with open('configs.json') as file:
    config = json.load(file)

Config = [baseSpeed,stepSpeed,forwardDelay,backwardDelay,leftrotateDelay, rightrotateDelay,rotateFactor,resetDelay]

class AgentPrediction:
    def __init__(self, id,conn) -> None:
        self.observations = []
        self.id = id
        self.conn=conn
        self.SendConfig(config[str(id)])

    def AddObservation(self, observation):
        print("Obv rec",len(observation)*8)
        if len(observation) != 14:
            return None
        
        obv=list(chain.from_iterable(observation))
        self.observations.append(obv)
        if len(self.observations) == 3:
            print("Called for action")
            actions = self.GetAction(self.observations)
            print("sent Action",actions)
            # Check
            self.observations.pop(0)
            self.SendAction(actions)
    
    def SendAction(self, actions):
        try:
            print("action sent", b'A'+" ".join(actions).encode() + b' \n')
            self.conn.send((b'A'+" ".join(actions).encode() + b' \n'))
        except Exception as e:
            print("Agent Send Action failed", e, self.id)

    def SendConfig(self, config):
        config = config.values()
        print(list(config))
        try:
            self.conn.send(('C'+ " ".join(config)).encode() + b' \n')
            print(f"Sent Config to Agent {self.id}")
        except Exception as e:
            print("Agent Send Config failed", e, self.id)

    def GetAction(self, seq):
        print("Get Action length:",len(seq))
        if(len(seq)!=3): 
            return None
        
        seq = list(chain.from_iterable(seq))
        x = np.array([seq]).astype(np.float32)
        y = np.array([[1]*3+[0]*3+[1]*3]).astype(np.float32)
        outputs = ort_sess.run(['action'], {'vector_observation': x, 'action_masks': y})

        probs = [math.pow(math.e, x) for x in outputs[0][0]]
        forwardAxis,rightAxis,rotateAxis=[np.argmax(probs[i:i+3]) for i in range(0,9,3)]
        if(forwardAxis==2):
            forwardAxis=1
        elif(forwardAxis==1):
            forwardAxis=2
        return [str(forwardAxis),str(rightAxis),str(rotateAxis)]
    
    def close(self):
        try:
            self.conn.close()
        except:
            print('failed to close socket AgentID',self.id)
    


# print(GetAction(sequence*3))