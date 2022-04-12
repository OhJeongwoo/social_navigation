import numpy as np
import rospy
import time

# from env import PedSimEnv
from gazebo_master import GazeboMaster

rospy.init_node("demo")
ped_sim = GazeboMaster()
time.sleep(3.0)
max_step=1000
while True:
    print("reset")
    s = ped_sim.reset()
    print("Waiting to start")
    step = 0
    print("start")
    while True:
        print(step)
        a = [0,1]
        s, r, d, info = ped_sim.step(a)
        step += 1
        print(info)
        d = d or step == max_step
        if d:
            break
