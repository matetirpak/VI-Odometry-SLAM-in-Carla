import numpy as np
import ultralytics


def min_distance_in_bbox(x, y, z, bbox, K):
    """ 
    Associates every pixel inside the bounding box with a 3D point 
    and returns the minimum distance.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    min_distance = 1000
    step_size = 4
    for row in range(y_min, y_max, step_size):
        for col in range(x_min, x_max, step_size):
            dist = np.sqrt(y[row][col]**2 + x[row][col]**2 + z[row][col]**2)
            min_distance = min(min_distance, dist)
    
    return min_distance


#######################################################
# Yolo detections are not used in the current version #
# of the code, but this may change in the future.     #
#######################################################

def plot_detections(agent, img, yolo_names, yolo_output, bbox_threshold=0.7):
    """
    Plots bounding boxes and their minimum distance from the agent.
    """
    z = agent.depth_map
    x, y = xy_from_depth(z, agent.K)

    for i in range(len(yolo_output['boxes'])):
        if 'scores' in yolo_output.keys() and \
        yolo_output['scores'][i].cpu().item() < bbox_threshold:
            continue
        box = yolo_output['boxes'][i].cpu()
        x_min, y_min, x_max, y_max = map(int, box)

        min_distance = min_distance_in_bbox(x, y, z, box, agent.K)

        label = yolo_names[yolo_output['labels'][i].cpu().item()]
        annotation = label
        if 'scores' in yolo_output.keys():
            annotation += f" {yolo_output['scores'][i].cpu().item():.2f}"
        annotation += f" {min_distance:.1f}m"
        

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


        text = f"{annotation}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img, (x_min, y_min - text_height - 4),
            (x_min + text_width, y_min), (0, 255, 0), -1
        )
        cv2.putText(
            img, text, (x_min, y_min - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
    return img

def inference_yolo(model, X):
    if not isinstance(model, ultralytics.models.yolo.model.YOLO):
        msg = "Model must be of type 'ultralytics.models.yolo.model.YOLO'"
        raise ValueError(msg)
    ret = []
    output = model(X, verbose=False, device='cpu')
    for sample in output:
        ret.append(_process_single_yolo_output(sample.boxes))
    return ret

def process_single_yolo_output(boxes_obj):
    return {
        "boxes": boxes_obj.xyxy.cpu(),          # shape [N, 4]
        "scores": boxes_obj.conf.cpu(),         # shape [N]
        "labels": boxes_obj.cls.cpu().long(),   # shape [N]
    }
