import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import numpy as np
from nms import nms
from utils import draw_bboxes, preprocess_image, denormalize_bboxes
import cv2
import math

ort_sess = ort.InferenceSession('models/face_detection_yunet_2023mar.onnx')


# pred_image, image = preprocess_image('images/image.jpg')
image = cv2.imread('images/image.jpg')
w, h = image.shape[1], image.shape[0]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# preprocess image with blobFromImage opencv
blob = cv2.dnn.blobFromImage(image, 1.0/255, (640, 640), (0, 0, 0), True)

outputs = ort_sess.run(None, {'input': blob})


cls_8 = outputs[0]
cls_16 = outputs[1]
cls_32 = outputs[2]

obj_8 = outputs[3]
obj_16 = outputs[4]
obj_32 = outputs[5]

bbox_8 = outputs[6]
bbox_16 = outputs[7]
bbox_32 = outputs[8]

kps_8 = outputs[9]
kps_16 = outputs[10]
kps_32 = outputs[11]


cls_all = np.concatenate([cls_8, cls_16, cls_32], axis=1, dtype=np.float64)  # Shape: (1, 8400, 1)
obj_all = np.concatenate([obj_8, obj_16, obj_32], axis=1, dtype=np.float64) # shape: (1, 8400, 1)
bbox_all = np.concatenate([bbox_8, bbox_16, bbox_32], axis=1, dtype=np.float64) # shape: (1, 8400, 4)
kps_all = np.concatenate([kps_8, kps_16, kps_32], axis=1, dtype=np.float64) # shape: (1, 8400, 10)

face_score = []
face_bboxes = []

for i, bbox in enumerate(bbox_16[0]):

    # Get Score
    cls_score = cls_all[0][i][0]
    obj_score = obj_all[0][i][0]

    cls_score = min(cls_score, 1.0)
    cls_score = max(cls_score, 0.0)
    obj_score = min(obj_score, 1.0)
    obj_score = max(obj_score, 0.0)

    score = math.sqrt(cls_score * obj_score)
    face_score.append(score)

    # get BBox
    cx = bbox[0]
    cy = bbox[1]
    w  = bbox[2]
    h  = bbox[3]

    x1 = cx - w/2.0
    y1 = cy - h/2.0

    face_bboxes.append([float(x1), float(y1), float(w), float(h)])

keepIdx = cv2.dnn.NMSBoxes(face_bboxes, face_score, 0.002, 0.2)

score = face_score[keepIdx[0]]
bbox = face_bboxes[keepIdx[0]]

x1, y1, w, h = bbox
x2, y2 = x1 + w, y1 + h
