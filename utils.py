import numpy as np
import cv2
def denormalize_bboxes(bboxes, image):
    # Denormalize the bounding boxes
    h, w, _ = image.shape
    # cx, cy, w, h

    bboxes[:, 0] = bboxes[:, 0] * w
    bboxes[:, 1] = bboxes[:, 1] * h
    bboxes[:, 2] = bboxes[:, 2] * w
    bboxes[:, 3] = bboxes[:, 3] * h
    
    # to convert x1, y1, w, h
    bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

    return bboxes

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)  # Shape: (H, W, 3) in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Resize to (640, 640)
    img = cv2.resize(img, (640, 640))
    img_org = img.copy()
    # Convert to float32 and normalize (if needed)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Change shape from (640, 640, 3) to (1, 3, 640, 640)
    img = np.transpose(img, (2, 0, 1))  # Change to (3, 640, 640)
    img = np.expand_dims(img, axis=0)  # Add batch dimension -> (1, 3, 640, 640)

    return img, img_org

def draw_bboxes(img, bboxes, img_size=(640, 640), confidences=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    :param image_path: Path to the image
    :param bboxes: List of denormalized bounding boxes in (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    :param img_size: Image size (width, height) used for denormalization
    :param confidences: List of confidence scores (optional)
    :param color: Bounding box color (default: green)
    :param thickness: Line thickness (default: 2)
    :return: Image with bounding boxes
    """
    # Load image
    h_img, w_img = img.shape[:2]  # Get original image dimensions

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Add confidence score if available
        if confidences is not None:
            label = f"{confidences[i]:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img