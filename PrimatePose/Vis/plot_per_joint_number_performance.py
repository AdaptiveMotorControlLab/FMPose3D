import os 

# Joint data structure containing counts for each keypoint
# This can be extended later to include performance metrics for each joint
joint_data = {
    'head': {     'index': 1,    'count': 250231, 'rmse': 11.568162293700691},
    'left_eye': { 'index': 2, 'count': 133930, 'rmse': 22.593621609738666},
    'right_eye': {'index': 3, 'count': 134111, 'rmse': 25.184937635096215},
    'nose': {     'index': 4, 'count': 273261, 'rmse': 11.494327189581833},
    'left_ear': { 'index': 5, 'count': 15370, 'rmse': 7.513362592300672},
    'right_ear': {'index': 6, 'count': 15345, 'rmse': 8.712391733688653},
    'mouth_front_top': {'index': 7, 'count': 711, 'rmse': 12.646792293110229},
    'mouth_front_bottom': {'index': 8, 'count': 706, 'rmse':11.601130435889232},
    'mouth_back_left': {'index': 9, 'count': 420, 'rmse': 14.703917634495392},
    'mouth_back_right': {'index': 10, 'count': 497, 'rmse': 17.37634478310056},
    'neck': {'index': 11, 'count': 264389, 'rmse': 21.341959814143838},
    'left_shoulder': {'index': 12, 'count': 281518, 'rmse': 23.494618466304786},
    'right_shoulder': {'index': 13, 'count': 281316, 'rmse': 23.237039839009913},
    'left_elbow': {'index': 14, 'count': 139295, 'rmse': 83.19652440520564},
    'right_elbow': {'index': 15, 'count': 139226, 'rmse': 82.33362199420029},
    'left_wrist': {'index': 16, 'count': 84392, 'rmse': 20.93366179539563},
    'right_wrist': {'index': 17, 'count': 84564, 'rmse': 20.809732782749826},
    'left_hand': {'index': 18, 'count': 191373, 'rmse': 39.44266285661285},
    'right_hand': {'index': 19, 'count': 191270, 'rmse': 39.83201431271569},
    'center_hip': {'index': 20, 'count': 246225, 'rmse': 24.23774538857192},
    'left_hip': {'index': 21, 'count': 30193, 'rmse': 31.335065218077467},
    'right_hip': {'index': 22, 'count': 30220, 'rmse': 33.0188235148643},
    'left_knee': {'index': 23, 'count': 271883, 'rmse': 27.9753857458248},
    'right_knee': {'index': 24, 'count': 271931, 'rmse': 27.865908344267343},
    'left_ankle': {'index': 25, 'count': 81837, 'rmse': 30.403213517134613},
    'right_ankle': {'index': 26, 'count': 81837, 'rmse': 31.489002846995415},
    'left_foot': {'index': 27, 'count': 147271, 'rmse': 11.137645273697562},
    'right_foot': {'index': 28, 'count': 147213, 'rmse': 11.052245595414114},
    'root_tail': {'index': 29, 'count': 16140, 'rmse': 35.1746409370051},
    'mid_tail': {'index': 30, 'count': 299, 'rmse': 91.70855898833835},
    'mid_end_tail': {'index': 31, 'count': 257, 'rmse': 58.77969381128916},
    'end_tail': {'index': 32, 'count': 203060, 'rmse': 6.445599736448217}
}

# Helper functions for working with joint data
def get_joint_count(joint_name):
    """Get the count for a specific joint."""
    return joint_data.get(joint_name, {}).get('count', 0)

def get_joint_index(joint_name):
    """Get the index for a specific joint."""
    return joint_data.get(joint_name, {}).get('index', -1)

def get_joint_by_index(index):
    """Get the joint name by its index."""
    for joint_name, joint_info in joint_data.items():
        if joint_info['index'] == index:
            return joint_name
    return None

def get_all_joint_names():
    """Get a list of all joint names."""
    return list(joint_data.keys())

def get_joints_with_zero_count():
    """Get joints that have zero count."""
    return [joint for joint, data in joint_data.items() if data['count'] == 0]

def get_top_joints_by_count(n=10):
    """Get top N joints by count."""
    sorted_joints = sorted(joint_data.items(), key=lambda x: x[1]['count'], reverse=True)
    return sorted_joints[:n]

# Example usage and statistics
if __name__ == "__main__":
    print("Joint Count Statistics:")
    print(f"Total joints: {len(joint_data)}")
    print(f"Joints with zero count: {get_joints_with_zero_count()}")
    print(f"Total annotations: {sum(data['count'] for data in joint_data.values())}")
    
    print("\nTop 10 joints by count:")
    for joint, data in get_top_joints_by_count(10):
        print(f"  {joint}: {data['count']}")
