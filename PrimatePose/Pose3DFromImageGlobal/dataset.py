import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class PrimateDataset(Dataset):
    """
    Dataset for primate pose estimation from JSON annotations.
    Compatible with AP10K and similar COCO-format datasets.
    Returns image path, 2D pose keypoints, and bounding box.
    """
    
    def __init__(self, json_file, image_root=None, transform=None, image_size=(256, 256)):
        """
        Args:
            json_file (str): Path to the JSON annotation file
            image_root (str): Root directory for images. If None, uses absolute paths from JSON
            transform: Optional transform to be applied on images
            image_size (tuple): Target image size for resizing
        """
        self.json_file = json_file
        self.image_root = image_root
        self.image_size = image_size
        
        # Load JSON data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = self.data['annotations']
        
        # Get keypoint names and count
        self.keypoint_names = self.data['categories'][0]['keypoints']
        self.num_keypoints = len(self.keypoint_names)
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_path (str): Path to the image
            pose_2d (torch.Tensor): 2D pose keypoints (num_keypoints, 3) with [x, y, visibility]
            bbox (torch.Tensor): Bounding box [x, y, width, height]
            image (torch.Tensor): Preprocessed image tensor
            original_size (tuple): Original image (width, height)
        """
        annotation = self.annotations[idx]
        image_info = self.images[annotation['image_id']]
        
        # Get image path
        if self.image_root:
            image_path = os.path.join(self.image_root, image_info['file_name'])
        else:
            # Assume full path or relative to current directory
            image_path = image_info['file_name']
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        original_size = (image_info['width'], image_info['height'])
        
        # Get 2D pose keypoints
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)  # (num_keypoints, 3)
        pose_2d = torch.tensor(keypoints, dtype=torch.float32)
        
        # Get bounding box
        bbox = torch.tensor(annotation['bbox'], dtype=torch.float32)  # [x, y, w, h]
        
        # Transform image
        if self.transform:
            # Calculate scale factors for keypoint adjustment
            scale_x = self.image_size[0] / original_size[0]
            scale_y = self.image_size[1] / original_size[1]
            
            # Adjust keypoints for resizing
            pose_2d_adjusted = pose_2d.clone()
            pose_2d_adjusted[:, 0] *= scale_x  # x coordinates
            pose_2d_adjusted[:, 1] *= scale_y  # y coordinates
            
            # Adjust bbox for resizing
            bbox_adjusted = bbox.clone()
            bbox_adjusted[0] *= scale_x  # x
            bbox_adjusted[1] *= scale_y  # y
            bbox_adjusted[2] *= scale_x  # width
            bbox_adjusted[3] *= scale_y  # height
            
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            pose_2d_adjusted = pose_2d
            bbox_adjusted = bbox
        
        return {
            'image_path': image_path,
            'pose_2d': pose_2d_adjusted,
            'bbox': bbox_adjusted,
            'image': image_tensor,
            'original_size': original_size,
            'image_id': annotation['image_id'],
            'annotation_id': annotation['id']
        }
    
    def get_keypoint_names(self):
        """Return list of keypoint names"""
        return self.keypoint_names
    
    def get_valid_keypoints_mask(self, pose_2d):
        """
        Get mask for valid keypoints (visibility > 0)
        Args:
            pose_2d: (num_keypoints, 3) tensor with [x, y, visibility]
        Returns:
            mask: (num_keypoints,) boolean tensor
        """
        return pose_2d[:, 2] > 0
    
    def normalize_pose_2d(self, pose_2d, bbox):
        """
        Normalize 2D pose relative to bounding box
        Args:
            pose_2d: (num_keypoints, 3) tensor
            bbox: (4,) tensor [x, y, w, h]
        Returns:
            normalized_pose: (num_keypoints, 3) tensor with normalized coordinates
        """
        normalized_pose = pose_2d.clone()
        
        # Normalize x coordinates
        normalized_pose[:, 0] = (pose_2d[:, 0] - bbox[0]) / bbox[2]
        # Normalize y coordinates  
        normalized_pose[:, 1] = (pose_2d[:, 1] - bbox[1]) / bbox[3]
        
        return normalized_pose


def create_data_loaders(train_json, test_json, image_root=None, batch_size=32, 
                       num_workers=4, image_size=(256, 256)):
    """
    Create train and test data loaders
    """
    # Create datasets
    train_dataset = PrimateDataset(
        json_file=train_json,
        image_root=image_root,
        image_size=image_size
    )
    
    test_dataset = PrimateDataset(
        json_file=test_json,
        image_root=image_root,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    train_json = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets/ap10k_train.json"
    test_json = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/ap10k_test.json"
    
    dataset = PrimateDataset(test_json)
    print(f"Dataset size: {len(dataset)}")
    print(f"Keypoints: {dataset.get_keypoint_names()}")
    print(f"Number of keypoints: {dataset.num_keypoints}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pose 2D shape: {sample['pose_2d'].shape}")
    print(f"Bbox: {sample['bbox']}")
    print(f"Original size: {sample['original_size']}") 