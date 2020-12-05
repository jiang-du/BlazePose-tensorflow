import os
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode

model = BlazePose()

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=199))

"""
0-Right ankle
1-Right knee
2-Right hip
3-Left hip
4-Left knee
5-Left ankle
6-Right wrist
7-Right elbow
8-Right shoulder
9-Left shoulder
10-Left elbow
11-Left wrist
12-Neck
13-Head top
"""

assert train_mode

import cv2
import numpy as np
from yolov4.tf import YOLOv4
yolo = YOLOv4()
yolo.classes = "./../yolov4/coco.names"
yolo.make_model()
yolo.load_weights("./../yolov4/yolov4.weights", weights_type="yolo")
cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    # ------ YOLO detection for the boxes ------
    d = yolo.predict(frame)
    # len(d): num of objects
    # for d[i], dims 0 to 3 is position ranged [0, 1] -- center_x, center_y, w, h; dim 4 is class (0 for person); dim 5 is score.
    img_size = frame.shape  # (480, 640, 3)
    
    # ------ get boxes ------
    for bbox in d:
        if (bbox[5]>=0.4) and (bbox[4]==0):
            # high score person
            for i in (1, 3):
                bbox[i] *= img_size[0]
            for i in (0, 2):
                bbox[i] *= img_size[1]
            # box position
            c_x = int(bbox[0])
            c_y = int(bbox[1])
            half_w = int(bbox[2] / 2)
            half_h = int(bbox[3] / 2)
            top_left = (c_x - half_w, c_y - half_h)
            bottom_right = (c_x + half_w, c_y + half_h)
            
            # ------ draw the box ------
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 1)
            
            # ------ skeleton detection ------
            img = frame[top_left[1]:bottom_right[1]][top_left[0]:bottom_right[0]]
            img = cv2.resize(frame, (256, 256))
            y = model(tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32))
            skeleton = y[0].numpy() # x,y range in 256
            # normalize pose
            skeleton[:, 0] = skeleton[:, 0] * half_w / 128 + top_left[0]
            skeleton[:, 1] = skeleton[:, 1] * half_h / 128 + top_left[1]
            skeleton = skeleton.astype(np.int16)
            # draw the joints
            for i in range(14):
                cv2.circle(frame, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
            # draw the lines
            for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5)):
                cv2.line(frame, tuple(skeleton[j[0]][0:2]), tuple(skeleton[j[1]][0:2]), color=(0, 0, 255), thickness=1)
            # solve the mid point of the hips
            cv2.line(frame, tuple(skeleton[12][0:2]), tuple(skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2), color=(0, 0, 255), thickness=1)
    
    frame = cv2.resize(frame, (1280, 960))
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
pass
