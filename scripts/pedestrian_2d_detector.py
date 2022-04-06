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
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.camera = PinholeCameraModel()
        self.dep_w = 480
        self.dep_h = 848
        self.rgb_w = 1280
        self.rgb_h = 720
        self.seq = 0
        # self.pub = rospy.Publisher("/instances", InstanceArray, queue_size=10)
        self.pub_sig = rospy.Publisher("/signal", Int32, queue_size=10)
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        

    def callback(self, msg):
        self.seq += 1
        st = time.time()
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        cv2.imwrite("image_raw/"+str(self.seq).zfill(6)+".png", im)
        outputs = self.predictor(im)
        pedestrians_unfiltered = outputs['instances'][outputs['instances'].pred_classes==0]
        pedestrians = pedestrians_unfiltered[pedestrians_unfiltered.scores > 0.8].to("cpu")
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(pedestrians)
        cv2.imwrite("result/"+str(self.seq).zfill(6)+".png",out.get_image()[:, :, ::-1])
        print(time.time()- st)
        save_dict = {}
        N = len(pedestrians.pred_boxes)
        save_dict['num_instance'] = N
        save_dict['box'] = pedestrians.pred_boxes.tensor.numpy().tolist()
        save_dict['class'] = pedestrians.pred_classes.numpy().tolist()
        with open("label/"+str(self.seq).zfill(6)+".json", 'w') as jf:
            json.dump(save_dict, jf, indent=4)
        print(time.time()- st)


if __name__=="__main__":
    rospy.init_node("pedestrian_detector")

    ped_detector = PedDetector()
    rospy.spin()