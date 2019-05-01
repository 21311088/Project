# coding=utf-8
from __future__ import print_function,division
import logging
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

from openvino.inference_engine import IENetwork,IEPlugin


size=64

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)

    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="MYRIAD",
                        type=str)
    return parser
 
def main():
    args = build_argparser().parse_args()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        for i in range(len(net.layers.keys())):
            l =net.layers.keys()
            print (l)
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    log.info("Preparing inputs")
    input_blob = next(iter(net.inputs))
    out_blob=next(iter(net.outputs))
    net.batch_size=1
    n,c,h,w=net.inputs[input_blob].shape
    print(net.outputs[out_blob].shape)
#    input_blob=next(iter(net.inputs))
    detector = dlib.get_frontal_face_detector()

    cam = cv2.VideoCapture(0)  
    log.info("Loading model to the plugin")
    exec_net=plugin.load(network=net)
    del net

    while True:  
        _, img = cam.read()  
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_image, 1)
        if not len(dets):
            #print('Can`t get face.')
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff  
            if key == 27:
                sys.exit(0)
            
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            face = img[x1:y1,x2:y2]
            # 调整图片的尺寸
            face=cv2.resize(face,(w,h))
            face=face.transpose((2,0,1))
            face=face.reshape((n,c,h,w))
           # print('Is this my face? %s' % is_my_face(face))
            res=exec_net.infer(inputs={input_blob:face/255.0})
            outname=list(res.keys())[0]
            res=res[outname]
            probs=np.argsort(res)
            if probs[0][0]==1:
                print('yeah!!!!you are you')
            else:
                print('Noooooo!!who are you???')
            cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
            cv2.imshow('image',img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
    del exec_net
    del plugin
if __name__=='__main__':
    sys.exit(main() or 0)  
