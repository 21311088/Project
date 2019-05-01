# coding=utf-8
import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
my_faces_path = './my_faces'
other_faces_path = './other_faces'
size = 64

imgs = []
labs = []
#x = tf.placeholder(tf.float32, [None, size, size, 3])
 
tf.Graph().as_default()
graph_def=tf.GraphDef()
with gfile.FastGFile('./peggy.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    ans=tf.import_graph_def(graph_def,name="")
sess=tf.Session()
sess.run(tf.global_variables_initializer())
x=sess.graph.get_tensor_by_name("inputx:0")
outlayer=sess.graph.get_tensor_by_name("fuckyou:0")
#out_label=sess.graph.get_tensor_by_name("output:0")
predict = tf.argmax(outlayer,axis= 1) 
       
def is_my_face(image):
      
#    res = sess.run(outlayer, feed_dict={x: [image/255.0]})  
#    prediction=tf.argmax(res,axis=1)
    res=sess.run(predict,feed_dict={x:[image/255.0]})
    if res[0] == 1:  
        return True  
    else:  
        return False  

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  
   
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
        face = cv2.resize(face, (size,size))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2,x1),(y2,y1), (255,0,0),3)
        cv2.imshow('image',img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close() 
