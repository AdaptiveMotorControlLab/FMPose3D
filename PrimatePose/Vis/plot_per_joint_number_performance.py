import os 
import matplotlib.pyplot as plt
import numpy as np

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

def plot_keypoint_distribution(joint_data, output_dir='./plots', dataset_name='PrimatePose'):
    """
    Plot histogram of keypoint distributions using the joint_data structure
    
    Args:
        joint_data: Dictionary containing joint information with 'count' field
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the plot title
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for publication-quality plots
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Get keypoint names and counts in index order
    keypoint_names = []
    keypoint_counts = []
    
    # Sort by index to maintain order
    sorted_joints = sorted(joint_data.items(), key=lambda x: x[1]['index'])
    
    for joint_name, joint_info in sorted_joints:
        keypoint_names.append(joint_name)
        keypoint_counts.append(joint_info['count'])
    
    total_annotations = sum(keypoint_counts)
    
    # Create figure with publication-quality dimensions
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create the bar plot
    x = np.arange(len(keypoint_names))
    
    # Create bars with uniform color
    max_count = max(keypoint_counts)
    bars = ax.bar(x, keypoint_counts, color='#4A90E2', edgecolor='white', linewidth=0.5, alpha=0.9)
    
    # Add value labels on top of bars with better formatting
    for i, (bar, count) in enumerate(zip(bars, keypoint_counts)):
        # Format large numbers with K/M notation for cleaner labels
        if count >= 1000000:
            label = f'{count/1000000:.1f}M'
        elif count >= 1000:
            label = f'{count/1000:.0f}K'
        else:
            label = f'{count}'
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_count*0.02, 
                label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#2E3440')
    
    # Customize the plot with publication-quality styling
    ax.set_xticks(x)
    ax.set_xticklabels(keypoint_names, rotation=45, ha='right', fontsize=9, fontweight='normal')
    ax.set_xlabel('Keypoint Name', fontsize=12, fontweight='bold', color='#2E3440', labelpad=10)
    ax.set_ylabel('Annotation Count', fontsize=12, fontweight='bold', color='#2E3440', labelpad=10)
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set y-axis to use K/M notation for large numbers
    def format_y_axis(x, pos):
        if x >= 1000000:
            return f'{x/1000000:.0f}M'
        elif x >= 1000:
            return f'{x/1000:.0f}K'
        else:
            return f'{x:.0f}'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y_axis))
    

    
    # Set axis limits with some padding
    ax.set_ylim(0, max_count * 1.15)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make left and bottom spines more prominent
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#2E3440')
    ax.spines['bottom'].set_color('#2E3440')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high quality
    plot_path = os.path.join(output_dir, f"{dataset_name}_keypoint_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Publication-quality plot saved to: {plot_path}")
    plt.close()

def plot_keypoint_performance(joint_data, output_dir='./plots', dataset_name='PrimatePose'):
    """
    Plot performance (RMSE) for each keypoint
    
    Args:
        joint_data: Dictionary containing joint information with 'rmse' field
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the plot title
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get keypoint names and RMSE values in index order
    keypoint_names = []
    rmse_values = []
    
    # Sort by index to maintain order
    sorted_joints = sorted(joint_data.items(), key=lambda x: x[1]['index'])
    
    for joint_name, joint_info in sorted_joints:
        keypoint_names.append(joint_name)
        rmse_values.append(joint_info['rmse'])
    
    plt.figure(figsize=(20, 10))
    
    # Create the bar plot
    x = np.arange(len(keypoint_names))
    
    # Create bars with color based on performance (lower RMSE = better performance)
    colors = ['green' if rmse < 20 else 'orange' if rmse < 50 else 'red' for rmse in rmse_values]
    bars = plt.bar(x, rmse_values, color=colors, edgecolor='black', alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, rmse) in enumerate(zip(bars, rmse_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01, 
                f'{rmse:.1f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Customize the plot
    plt.xticks(x, keypoint_names, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Keypoint Name', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    # plt.title(f'Keypoint Performance (RMSE) for {dataset_name}', 
            #   fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good (RMSE < 20)'),
        Patch(facecolor='orange', alpha=0.7, label='Medium (RMSE 20-50)'),
        Patch(facecolor='red', alpha=0.7, label='Poor (RMSE > 50)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{dataset_name}_keypoint_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {plot_path}")
    plt.close()

def plot_keypoint_count_vs_performance(joint_data, output_dir='./plots', dataset_name='PrimatePose'):
    """
    Create a scatter plot showing the relationship between count and performance
    
    Args:
        joint_data: Dictionary containing joint information with 'count' and 'rmse' fields
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the plot title
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data in index order
    keypoint_names = []
    counts = []
    rmse_values = []
    
    # Sort by index to maintain order
    sorted_joints = sorted(joint_data.items(), key=lambda x: x[1]['index'])
    
    for joint_name, joint_info in sorted_joints:
        keypoint_names.append(joint_name)
        counts.append(joint_info['count'])
        rmse_values.append(joint_info['rmse'])
    
    plt.figure(figsize=(16, 10))
    
    # Create scatter plot with different colors based on performance
    colors = ['green' if rmse < 20 else 'orange' if rmse < 50 else 'red' for rmse in rmse_values]
    sizes = [150 if count > 100000 else 100 if count > 50000 else 50 for count in counts]
    
    plt.scatter(counts, rmse_values, s=sizes, alpha=0.7, c=colors, edgecolors='black', linewidth=1)
    
    # Use log scale for count since there's a large range
    plt.xscale('log')
    
    # Add labels with better positioning to avoid overlap
    for i, (name, count, rmse) in enumerate(zip(keypoint_names, counts, rmse_values)):
        # Special handling for similar keypoints to avoid overlap
        if name == 'left_shoulder':
            offset_x, offset_y = 15, 10
            ha, va = 'left', 'bottom'
        elif name == 'right_shoulder':
            offset_x, offset_y = 15, -10
            ha, va = 'left', 'top'
        elif name == 'left_eye':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_eye':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_ear':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_ear':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_elbow':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_elbow':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_wrist':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_wrist':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_hand':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_hand':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_hip':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_hip':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_knee':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_knee':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_ankle':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_ankle':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif name == 'left_foot':
            offset_x, offset_y = 10, 15
            ha, va = 'left', 'bottom'
        elif name == 'right_foot':
            offset_x, offset_y = 10, -15
            ha, va = 'left', 'top'
        elif count < 1000:  # Small counts
            offset_x, offset_y = 10, 5
            ha, va = 'left', 'bottom'
        elif rmse > 60:  # High RMSE
            offset_x, offset_y = 5, -15
            ha, va = 'left', 'top'
        else:  # Default
            offset_x, offset_y = 5, 5
            ha, va = 'left', 'bottom'
        
        plt.annotate(name, (count, rmse), 
                    xytext=(offset_x, offset_y), textcoords='offset points', 
                    fontsize=9, ha=ha, va=va, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Customize the plot
    plt.xlabel('Count (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title(f'Keypoint Count vs Performance for {dataset_name}', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good (RMSE < 20)'),
        Patch(facecolor='orange', alpha=0.7, label='Medium (RMSE 20-50)'),
        Patch(facecolor='red', alpha=0.7, label='Poor (RMSE > 50)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set axis limits with some padding
    plt.xlim(min(counts) * 0.8, max(counts) * 1.2)
    plt.ylim(0, max(rmse_values) * 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{dataset_name}_count_vs_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Count vs Performance plot saved to: {plot_path}")
    plt.close()

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
    print(f"Total annotations: {sum(data['count'] for data in joint_data.values()):,}")
    
    print("\nTop 10 joints by count:")
    for joint, data in get_top_joints_by_count(10):
        print(f"  {joint}: {data['count']:,}")
    
    # Create all the keypoint plots
    plot_keypoint_distribution(joint_data, dataset_name='PrimatePose')
    plot_keypoint_performance(joint_data, dataset_name='PrimatePose')
    plot_keypoint_count_vs_performance(joint_data, dataset_name='PrimatePose')
