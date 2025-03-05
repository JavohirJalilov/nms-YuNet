import cv2
import numpy as np
from yunet import FaceDetectorYN
# Load face detector model
model_path = "models/face_detection_yunet_2023mar.onnx"  # Change to your actual model path
config_path = ""  # ONNX models don't usually require a config file

# Set parameters
input_size = (320, 320)  # Resize input image to this size
score_threshold = 0.5
nms_threshold = 0.3
top_k = 5000
backend_id = cv2.dnn.DNN_BACKEND_OPENCV  # Use OpenCV's default backend
target_id = cv2.dnn.DNN_TARGET_CPU  # Run on CPU

# Initialize face detector
face_detector = FaceDetectorYN(model_path, config_path, input_size, score_threshold, nms_threshold, top_k, backend_id, target_id)

# Open webcam
cap = cv2.VideoCapture(0)  # Use webcam (change to video file path if needed)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # get w, h
    W, H = frame.shape[1], frame.shape[0]
    # Resize frame to match model input
    frame_resized = cv2.resize(frame.copy(), input_size)

    # Detect faces
    faces = face_detector.detect(frame_resized)

    # Draw results
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            x = int(x * W / input_size[0])
            y = int(y * H / input_size[1])
            w = int(w * W / input_size[0])
            h = int(h * H / input_size[1])
            confidence = face[14]

            # Draw bounding box
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw keypoints (eyes, nose, mouth corners)
            for i in range(5):
                kx, ky = map(int, face[4 + 2 * i : 6 + 2 * i])
                kx = int(kx * W / input_size[0])
                ky = int(ky * H / input_size[1])
                frame = cv2.circle(frame, (kx, ky), 2, (0, 0, 255), -1)

            # Put confidence score
            frame = cv2.putText(frame, f"{confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show result
    cv2.imshow("Face Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()