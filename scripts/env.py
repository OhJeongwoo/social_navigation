import rospy
import rospkg
import numpy as np

from social_navigation.msg import Status, Command
from social_navigation.srv import Step, State, Jackal, Reset, StepResponse, StateResponse, JackalResponse, StepRequest, StateRequest, JackalRequest, ResetRequest, ResetResponse

class PedSimEnv:
    def __init__(self):
        self.client_step_ = rospy.ServiceProxy('step', Step)
        self.client_state_ = rospy.ServiceProxy('state', State)
        self.client_jackal_ = rospy.ServiceProxy('jackal', Jackal)
        self.client_reset_ = rospy.ServiceProxy('reset', Reset)

    def step(self, a):
        s = self.get_obs()
        self.jackal_cmd(a)
        req = StepRequest()
        req.request = True
        res = self.client_step_(req)
        ns = self.get_obs()
        done = False
        info = {}
        reward = 0

        # robot reward
        

        # pedestrian reward

        # map reward
        return ns, reward, done, info

    def reset_model(self):
        req = ResetRequest()
        req.request = True
        print(req)
        res = self.client_reset_(req)
        return self.get_obs()

    def get_obs(self):
        req = StateRequest()
        req.request = True
        res = self.client_state_(req)
        return np.array(res.state)

    def jackal_cmd(self, a):
        req = JackalRequest()
        req.accel = a[0]
        req.omega = a[1]
        req.request = True
        res = self.client_jackal_(req)
        return res.success