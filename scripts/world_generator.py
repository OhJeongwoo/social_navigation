import time
import argparse
import os
import sys
import yaml
import json
import rospkg
from lxml import etree
from lxml.etree import Element

from utils import *


if __name__=="__main__":
    init_time_ = time.time()
    
    parser = argparse.ArgumentParser(description='World Generator')
    parser.add_argument('--yaml', default='test', type=str)
    args = parser.parse_args()

    rospack = rospkg.RosPack()
    PKG_PATH = rospack.get_path("social_navigation")
    CFG_PATH = PKG_PATH + '/config/'
    YML_PATH = PKG_PATH + '/yaml/'
    WRD_PATH = PKG_PATH + '/worlds/'
    
    YML_FILE = YML_PATH + args.yaml + '.yaml'
    
    # set yaml path
    with open(YML_FILE) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        print(args)

    MAP_FILE = CFG_PATH + args['map_file_name'] + '.json'
    ACT_FILE = CFG_PATH + args['act_file_name'] + '.json'
    BSE_FILE = WRD_PATH + args['base_world_name'] + '.world'
    WRD_FILE = WRD_PATH + args['save_world_name'] + '.world'

    # load wall file
    with open(MAP_FILE) as file:
        map_dict = json.load(file)
    
    with open(ACT_FILE) as file:
        act_dict = json.load(file)

    wall_info_list = transform_wall_info(map_dict)
    actor_info_list = transform_actor_info(act_dict)
    
    tree_ = etree.parse(BSE_FILE)
    world_ = tree_.getroot().getchildren()[0]
    
    

    walls = Element("model", name="walls")
    base = Element("pose")
    base.text = "0 0 0 0 0 0"
    walls.append(base)
    walls.append(get_element("static", "1"))

    for wall_info in wall_info_list:
        wall = get_wall(wall_info)
        walls.append(wall)
    
    world_.append(walls)

    for actor_info in actor_info_list:
        actor = get_actor(actor_info)
        world_.append(actor)
    
    etree.indent(tree_)
    tree_.write(WRD_FILE, pretty_print=True, xml_declaration=True, encoding="utf-8")
