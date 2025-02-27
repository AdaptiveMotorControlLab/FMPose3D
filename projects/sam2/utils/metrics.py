import motmetrics as mm
import numpy as np
import json


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
    # Convert [x_min, y_min, x_max, y_max] format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0
    return 0

def convert_to_x1y1x2y2(bbox):
    """Convert [x, y, w, h] to [x1, y1, x2, y2] format"""
    if bbox is None:
        return None
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


class MOTATracker:
    def __init__(self):
        """Initialize MOTA tracker"""
        self.acc = mm.MOTAccumulator(auto_id=True)
        
    def update(self, gt_bboxes, pred_bboxes):
        """Update metrics with new frame data
        Args:
            gt_bboxes: List of ground truth bboxes [x, y, w, h]
            pred_bboxes: List of predicted bboxes [x, y, w, h]
        """

        if not gt_bboxes and not pred_bboxes:
            return
            
        # Convert boxes to x1y1x2y2 format
        gt_bboxes = [convert_to_x1y1x2y2(bbox) for bbox in gt_bboxes]
        valid_pred_bboxes = [convert_to_x1y1x2y2(bbox) for bbox in pred_bboxes if bbox is not None]
        
        # if len(gt_bboxes) != len(valid_pred_bboxes):
        #     raise ValueError('Dimension mismatch. Check the inputs.') ??
        
        # Convert to numpy arrays for mm.distances.iou_matrix
        gt_bboxes = np.array(gt_bboxes)
        valid_pred_bboxes = np.array(valid_pred_bboxes)
        
        if len(valid_pred_bboxes) == 0:
            # Handle case with no valid predictions
            distances = np.empty((len(gt_bboxes), 0))
        else:
            # Calculate IoU distance matrix using motmetrics
            distances = mm.distances.iou_matrix(gt_bboxes, valid_pred_bboxes)
        
        # Update accumulator
        self.acc.update(
            [i for i in range(len(gt_bboxes))],      # Ground truth objects
            [i for i in range(len(valid_pred_bboxes))], # Predicted objects
            distances
        )
   
    def get_metrics(self):
        """Calculate current metrics"""
        mh = mm.metrics.create()
        return mh.compute(
            self.acc,
            metrics=['mota', 'motp', 'num_switches', 'num_false_positives', 'num_misses'],
            name='acc'
        )
    
    def save_metrics(self, output_path):
        """Save current metrics to JSON file"""
        summary = self.get_metrics()
        with open(output_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        return summary