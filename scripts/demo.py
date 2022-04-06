import numpy as np
import rospy
import time

from env import PedSimEnv

rospy.init_node("demo")
ped_sim = PedSimEnv()
max_step=1000
while True:
    print("reset")
    s = ped_sim.reset_model()
    print("Waiting to start")
    time.sleep(10.0)
    step = 0
    print("start")
    while True:
        print(step)
        a = [0,1]
        s, r, d, _ = ped_sim.step(a)
        step += 1
        if d or step == max_step:
            break
        if step % 100 == 0:
            print("take a rest")
            time.sleep(10.0)
    
