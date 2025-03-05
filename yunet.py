import cv2
import numpy as np

class FaceDetectorYN:
    def __init__(self, model, config, input_size, score_threshold, nms_threshold, top_k, backend_id, target_id):
        self.divisor = 32
        self.strides = [8, 16, 32]
        
        # Load DNN model
        self.net = cv2.dnn.readNet(model, config)
        if self.net.empty():
            raise ValueError("Failed to load model")

        self.net.setPreferableBackend(backend_id)
        self.net.setPreferableTarget(target_id)

        # Set input size
        self.inputW, self.inputH = input_size
        self.padW = ((self.inputW - 1) // self.divisor + 1) * self.divisor
        self.padH = ((self.inputH - 1) // self.divisor + 1) * self.divisor

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def set_input_size(self, input_size):
        self.inputW, self.inputH = input_size
        self.padW = ((self.inputW - 1) // self.divisor + 1) * self.divisor
        self.padH = ((self.inputH - 1) // self.divisor + 1) * self.divisor

    def get_input_size(self):
        return (self.inputW, self.inputH)

    def set_score_threshold(self, score_threshold):
        self.score_threshold = score_threshold

    def get_score_threshold(self):
        return self.score_threshold

    def set_nms_threshold(self, nms_threshold):
        self.nms_threshold = nms_threshold

    def get_nms_threshold(self):
        return self.nms_threshold

    def set_top_k(self, top_k):
        self.top_k = top_k

    def get_top_k(self):
        return self.top_k

    def pad_with_divisor(self, input_image):
        bottom = self.padH - self.inputH
        right = self.padW - self.inputW
        return cv2.copyMakeBorder(input_image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=0)

    def detect(self, input_image):
        if input_image is None or input_image.size == 0:
            return None
        
        if input_image.shape[:2] != (self.inputH, self.inputW):
            raise ValueError("Size does not match. Call set_input_size(size) if input size does not match the preset size")

        # Pad input image
        pad_image = self.pad_with_divisor(input_image)

        # Convert image to blob
        input_blob = cv2.dnn.blobFromImage(pad_image)
        print(input_blob)
        # Forward pass
        output_names = ["cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", 
                        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"]
        self.net.setInput(input_blob)
        output_blobs = self.net.forward(output_names)
        # print(output_blobs.shape)

        # Post-process results
        faces = self.post_process(output_blobs)
        return faces

    def post_process(self, output_blobs):
        faces = []
        for i, stride in enumerate(self.strides):
            cols = self.padW // stride
            rows = self.padH // stride

            cls = output_blobs[i]
            obj = output_blobs[i + len(self.strides)]
            bbox = output_blobs[i + 2 * len(self.strides)]
            kps = output_blobs[i + 3 * len(self.strides)]

            cls_v = cls.flatten()
            obj_v = obj.flatten()
            bbox_v = bbox.flatten()
            kps_v = kps.flatten()

            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    cls_score = np.clip(cls_v[idx], 0, 1)
                    obj_score = np.clip(obj_v[idx], 0, 1)
                    score = np.sqrt(cls_score * obj_score)

                    face = np.zeros(15, dtype=np.float32)
                    face[14] = score

                    # Bounding box
                    cx = (c + bbox_v[idx * 4 + 0]) * stride
                    cy = (r + bbox_v[idx * 4 + 1]) * stride
                    w = np.exp(bbox_v[idx * 4 + 2]) * stride
                    h = np.exp(bbox_v[idx * 4 + 3]) * stride

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0

                    face[0] = x1
                    face[1] = y1
                    face[2] = w
                    face[3] = h

                    # Keypoints
                    for n in range(5):
                        face[4 + 2 * n] = (kps_v[idx * 10 + 2 * n] + c) * stride
                        face[4 + 2 * n + 1] = (kps_v[idx * 10 + 2 * n + 1] + r) * stride

                    faces.append(face)

        faces = np.array(faces)

        # Apply Non-Maximum Suppression (NMS)
        if len(faces) > 1:
            face_boxes = [tuple(map(int, face[:4])) for face in faces]
            face_scores = [face[14] for face in faces]

            keep_idx = cv2.dnn.NMSBoxes(face_boxes, face_scores, self.score_threshold, self.nms_threshold, self.top_k)
            if len(keep_idx) > 0:
                faces = faces[keep_idx.flatten()]
        
        return faces
