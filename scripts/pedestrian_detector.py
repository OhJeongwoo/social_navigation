#!/usr/bin/python3
import numpy as np
import torch

import rospy

from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from social_navigation.msg import InstanceArray, InstanceImage

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class PedDetector:
    def __init__(self):
        # load detectron configuration
        self.cfg_ = get_cfg()
        self.cfg_.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor_ = DefaultPredictor(self.cfg_)
        self.seq_ = 0

        # set hyperparameter
        self.score_threshold_ = 0.8
        
        # define topic communicator
        self.pub_ = rospy.Publisher("/instances", InstanceArray, queue_size=10)
        self.pub_sig_ = rospy.Publisher("/signal", Int32, queue_size=10)
        self.sub_ = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        
    def callback(self, msg):
        self.seq_ += 1
        
        # send signal for synchronization between image and point cloud.
        # if signal subscriber receive this topic, they lock the point cloud msg at that moment.
        sig = Int32()
        sig.data = self.seq_
        self.pub_sig_.publish(sig)

        # inference mask rcnn
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        outputs = self.predictor_(im)
        
        # filter the result
        pedestrians_unfiltered = outputs['instances'][outputs['instances'].pred_classes==0]
        pedestrians = pedestrians_unfiltered[pedestrians_unfiltered.scores > self.score_threshold_]
        pred_masks = pedestrians.pred_masks
        pred_masks = torch.flatten(pred_masks,1,2)

        # generate topic to pedestrian detector
        # In the other node, they receive semantic segmentation result and fuse with point cloud.
        # We assign the instance id for each point in point cloud data, and estimate the pedestrian position.
        N = pred_masks.shape[0]
        rt = InstanceArray()
        rt.width = msg.width
        rt.height = msg.height
        rt.seq = self.seq_
        for i in range(N):
            mask = InstanceImage()
            mask.data = pred_masks[i].to("cpu").tolist()
            rt.instances.append(mask)
        self.pub_.publish(rt)


if __name__=="__main__":
    rospy.init_node("pedestrian_detector")
    ped_detector = PedDetector()
    rospy.spin()