import cv2
import numpy as np

def nms(bboxes, confidences, threshold=0.3):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    
    Parameters:
        detections (numpy.ndarray): Array of shape (N, 5), where each row is [x1, y1, x2, y2, confidence].
        iou_threshold (float): Threshold for Intersection over Union (IoU) to suppress overlapping boxes.
        
    Returns:
        numpy.ndarray: Filtered detections after NMS.
    """
    detections = np.concatenate((bboxes, confidences), axis=1)
    
    if len(detections) == 0:
        return np.array([])
    
    # Sort detections by confidence score in descending order
    detections = detections[np.argsort(detections[:, 4])[::-1]]
    
    keep = []
    while len(detections) > 0:
        # Pick the detection with the highest confidence and remove it from the list
        best = detections[0]
        keep.append(best)
        detections = detections[1:]
        
        if len(detections) == 0:
            break
        
        # Compute IoU between the best detection and remaining ones
        x1 = np.maximum(best[0], detections[:, 0])
        y1 = np.maximum(best[1], detections[:, 1])
        x2 = np.minimum(best[2], detections[:, 2])
        y2 = np.minimum(best[3], detections[:, 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box1_area = (best[2] - best[0]) * (best[3] - best[1])
        box2_area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area
        print("IoU:",iou)
        print("box1_area:",box1_area)
        print("box2_area:",box2_area)
        
        # Keep boxes that have IoU less than the threshold
        detections = detections[iou < threshold]
    
    return np.array(keep)


# """
#     Non-max Suppression Algorithm

#     @param list  Object candidate bounding boxes
#     @param list  Confidence score of bounding boxes
#     @param float IoU threshold

#     @return Rest boxes after nms operation
# """
# def non_maximum_suppression(bounding_boxes, confidence_score, threshold):
#     # If no bounding boxes, return empty list
#     if len(bounding_boxes) == 0:
#         return [], []

#     # Bounding boxes
#     boxes = bounding_boxes

#     start_x = boxes[:, 0]
#     start_y = boxes[:, 1]
#     end_x = boxes[:, 2]
#     end_y = boxes[:, 3]
#     # Confidence scores of bounding boxes
#     score = np.array(confidence_score)

#     # Picked bounding boxes
#     picked_boxes = []
#     picked_score = []

#     # Compute areas of bounding boxes
#     areas = (end_x - start_x) * (end_y - start_y)

#     # Sort by confidence score of bounding boxes
#     # add confidence score to bounding box as 4th element
#     boxes = np.column_stack((boxes, score))
#     # order through confidence score, lambda x: x[-1] is the last element
#     ordered_boxes = np.array(sorted(boxes, key=lambda x: x[-1], reverse=True))

#     while ordered_boxes.shape[0] > 0:
#         # The index of largest confidence score
#         max_index = np.argmax(ordered_boxes[:, 4])
#         picked_boxes.append(ordered_boxes[max_index])
#         picked_score.append(ordered_boxes[max_index][-1])

#         # Compute ordinates of intersection-over-union(IOU)
#         x1 = np.maximum(ordered_boxes[:, 0], ordered_boxes[max_index, 0])
#         x2 = np.minimum(ordered_boxes[:, 2], ordered_boxes[max_index, 2])
#         y1 = np.maximum(ordered_boxes[:, 1], ordered_boxes[max_index, 1])
#         y2 = np.minimum(ordered_boxes[:, 3], ordered_boxes[max_index, 3])

#         # Compute areas of intersection-over-union
#         w = np.maximum(0.0, x2 - x1 + 1)
#         h = np.maximum(0.0, y2 - y1 + 1)
#         intersection = w * h

#         # Compute the ratio between intersection and union
#         ratio = intersection / (areas + ordered_boxes[:, 4][max_index] - intersection)

#         left = np.where(ratio < threshold)
#         ordered_boxes = np.delete(ordered_boxes, left, axis=0)

#     return picked_boxes, picked_score

#     order = np.argsort(score)

#     # Iterate bounding boxes
#     while order.size > 0:
#         # The index of largest confidence score
#         index = order[-1]

#         # Pick the bounding box with largest confidence score
#         picked_boxes.append(bounding_boxes[index])
#         picked_score.append(confidence_score[index])

#         # Compute ordinates of intersection-over-union(IOU)
#         x1 = np.maximum(start_x[index], start_x[order[:-1]])
#         x2 = np.minimum(end_x[index], end_x[order[:-1]])
#         y1 = np.maximum(start_y[index], start_y[order[:-1]])
#         y2 = np.minimum(end_y[index], end_y[order[:-1]])

#         # Compute areas of intersection-over-union
#         w = np.maximum(0.0, x2 - x1 + 1)
#         h = np.maximum(0.0, y2 - y1 + 1)
#         intersection = w * h
#         # break

#         # Compute the ratio between intersection and union
#         ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

#         left = np.where(ratio < threshold)
#         order = order[left]

#     return picked_boxes, picked_score
