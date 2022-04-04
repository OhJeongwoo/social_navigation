#!/usr/bin/python3
import time
import argparse
import os
import sys
import yaml
import json
import rospkg
import rospy
import struct
import torch
import numpy as np
from PIL import Image as Image_PIL
import math
import itertools

from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int32
from image_geometry import PinholeCameraModel
from social_navigation.msg import InstanceArray, InstanceImage

from cv_bridge import CvBridge
import cv2
import ros_numpy

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo


class PedDetector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.camera = PinholeCameraModel()
        self.dep_w = 480
        self.dep_h = 848
        self.rgb_w = 1280
        self.rgb_h = 720
        self.seq = 0
        self.pub = rospy.Publisher("/instances", InstanceArray, queue_size=10)
        self.pub_sig = rospy.Publisher("/signal", Int32, queue_size=10)
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        

    def callback(self, msg):
        self.seq += 1
        st = time.time()
        sig = Int32()
        sig.data = self.seq
        self.pub_sig.publish(sig)
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        print(time.time()- st)
        outputs = self.predictor(im)
        print(time.time()- st)
        pedestrians_unfiltered = outputs['instances'][outputs['instances'].pred_classes==0]
        pedestrians = pedestrians_unfiltered[pedestrians_unfiltered.scores > 0.8]
        pred_masks = pedestrians.pred_masks
        print(time.time()- st)
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(pedestrians.to("cpu"))
        # cv2.imwrite("img.png",out.get_image()[:, :, ::-1])
        # print(time.time()- st)
        pred_masks = torch.flatten(pred_masks,1,2)
        print(time.time()- st)
        N = pred_masks.shape[0]
        rt = InstanceArray()
        rt.width = msg.width
        rt.height = msg.height
        rt.seq = self.seq
        for i in range(N):
            mask = InstanceImage()
            mask.data = pred_masks[i].to("cpu").tolist()
            rt.instances.append(mask)
        self.pub.publish(rt)
        print(time.time()- st)


if __name__=="__main__":
    rospy.init_node("pedestrian_detector")

    ped_detector = PedDetector()
    rospy.spin()