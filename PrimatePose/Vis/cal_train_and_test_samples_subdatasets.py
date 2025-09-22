# cal_train_and_test_samples for each subdataset, I want to know the total annotations, and the pose annotations for each subdataset.
# test folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets
# train folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets

import json
import os
from pathlib import Path
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import colorsys

def gcd(a, b):
    """Calculate greatest common divisor"""
    while b:
        a, b = b, a % b
    return a

def calculate_ratio_proportion(train_count, test_count):
    """
    Calculate train:test ratio as proportion of total (like 8:2, 7:3)
    
    Args:
        train_count (int): Number of training samples
        test_count (int): Number of test samples
        
    Returns:
        str: Ratio string like "8:2" or "7:3"
    """
    if train_count == 0 and test_count == 0:
        return "0:0"
    if test_count == 0:
        return "Train Only"
    if train_count == 0:
        return "Test Only"
    
    total = train_count + test_count
    train_pct = round((train_count / total) * 10)
    test_pct = round((test_count / total) * 10)
    
    # Ensure they add up to 10 (handle rounding issues)
    if train_pct + test_pct != 10:
        if train_count > test_count:
            train_pct = 10 - test_pct
        else:
            test_pct = 10 - train_pct
    
    # Simplify the ratio
    common_divisor = gcd(train_pct, test_pct)
    if common_divisor > 0:
        train_ratio = train_pct // common_divisor
        test_ratio = test_pct // common_divisor
    else:
        train_ratio = train_pct
        test_ratio = test_pct
    
    return f"{train_ratio}:{test_ratio}"

def count_annotations(json_file_path):
    """
    Count total annotations and pose annotations in a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON annotation file
        
    Returns:
        tuple: (total_annotations, pose_annotations, total_images)
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Count total images
        total_images = len(data.get('images', []))
        
        # Count total annotations
        annotations = data.get('annotations', [])
        total_annotations = len(annotations)
        
        # Count pose annotations (annotations with keypoints)
        pose_annotations = 0
        for annotation in annotations:
            if 'keypoints' in annotation and annotation['keypoints']:
                # Check if keypoints actually contain valid data (not all -1 or 0)
                keypoints = annotation['keypoints']
                if any(val > 0 for val in keypoints):
                    pose_annotations += 1
        
        return total_annotations, pose_annotations, total_images
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return 0, 0, 0

def analyze_subdatasets():
    """
    Analyze all subdatasets and count annotations for train and test splits.
    """
    train_folder = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets"
    test_folder = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets"
    
    # Get all subdataset names from train folder
    train_files = [f for f in os.listdir(train_folder) if f.endswith('_train.json')]
    subdatasets = [f.replace('_train.json', '') for f in train_files]
    subdatasets.sort()
    
    results = []
    
    print("="*110)
    print("SUBDATASET ANNOTATION ANALYSIS")
    print("="*110)
    print(f"{'Subdataset':<15} {'Split':<6} {'Images':<8} {'Total Ann.':<12} {'Pose Ann.':<10} {'Pose %':<10} {'Train:Test Ratio':<15} {'Debug Info':<15}")
    print("-"*110)
    
    total_train_images = 0
    total_test_images = 0
    total_train_annotations = 0
    total_test_annotations = 0
    total_train_pose = 0
    total_test_pose = 0
    
    # Dictionary to store combined stats for each subdataset
    subdataset_totals = {}
    
    for subdataset in subdatasets:
        # Initialize subdataset totals
        subdataset_totals[subdataset] = {
            'images': 0, 'total_annotations': 0, 'pose_annotations': 0,
            'train_images': 0, 'test_images': 0
        }
        
        # Process train file
        train_file = os.path.join(train_folder, f"{subdataset}_train.json")
        train_total, train_pose, train_images = 0, 0, 0
        if os.path.exists(train_file):
            train_total, train_pose, train_images = count_annotations(train_file)
            train_pose_pct = (train_pose / train_total * 100) if train_total > 0 else 0
            
            print(f"{subdataset:<15} {'Train':<6} {train_images:<8} {train_total:<12} {train_pose:<10} {train_pose_pct:<9.3f}% {'':<15} {'':<15}")
            
            results.append({
                'subdataset': subdataset,
                'split': 'train',
                'images': train_images,
                'total_annotations': train_total,
                'pose_annotations': train_pose,
                'pose_percentage': train_pose_pct
            })
            
            total_train_images += train_images
            total_train_annotations += train_total
            total_train_pose += train_pose
            
            # Add to subdataset totals
            subdataset_totals[subdataset]['images'] += train_images
            subdataset_totals[subdataset]['total_annotations'] += train_total
            subdataset_totals[subdataset]['pose_annotations'] += train_pose
            subdataset_totals[subdataset]['train_images'] = train_images
        
        # Process test file
        test_file = os.path.join(test_folder, f"{subdataset}_test.json")
        test_total, test_pose, test_images = 0, 0, 0
        if os.path.exists(test_file):
            test_total, test_pose, test_images = count_annotations(test_file)
            test_pose_pct = (test_pose / test_total * 100) if test_total > 0 else 0
            
            print(f"{subdataset:<15} {'Test':<6} {test_images:<8} {test_total:<12} {test_pose:<10} {test_pose_pct:<9.3f}% {'':<15} {'':<15}")
            
            results.append({
                'subdataset': subdataset,
                'split': 'test',
                'images': test_images,
                'total_annotations': test_total,
                'pose_annotations': test_pose,
                'pose_percentage': test_pose_pct
            })
            
            total_test_images += test_images
            total_test_annotations += test_total
            total_test_pose += test_pose
            
            # Add to subdataset totals
            subdataset_totals[subdataset]['images'] += test_images
            subdataset_totals[subdataset]['total_annotations'] += test_total
            subdataset_totals[subdataset]['pose_annotations'] += test_pose
            subdataset_totals[subdataset]['test_images'] = test_images
        
        # Calculate train:test ratio as proportion
        train_img_count = subdataset_totals[subdataset]['train_images']
        test_img_count = subdataset_totals[subdataset]['test_images']
        ratio_str = calculate_ratio_proportion(train_img_count, test_img_count)
        
        # Show combined totals for this subdataset
        combined_images = subdataset_totals[subdataset]['images']
        combined_total = subdataset_totals[subdataset]['total_annotations']
        combined_pose = subdataset_totals[subdataset]['pose_annotations']
        combined_pose_pct = (combined_pose / combined_total * 100) if combined_total > 0 else 0
        
        print(f"{subdataset:<15} {'TOTAL':<6} {combined_images:<8} {combined_total:<12} {combined_pose:<10} {combined_pose_pct:<9.3f}% {ratio_str:<15} T:{train_img_count},Te:{test_img_count}")
        
        # Debug: Show train/test image breakdown
        if train_img_count == 0 or test_img_count == 0:
            print(f"{'':>15} WARNING: {'No test data!' if test_img_count == 0 else 'No train data!'}")
        
        # Add combined results to results list
        results.append({
            'subdataset': subdataset,
            'split': 'total',
            'images': combined_images,
            'total_annotations': combined_total,
            'pose_annotations': combined_pose,
            'pose_percentage': combined_pose_pct,
            'train_test_ratio': ratio_str
        })
        
        print("-"*100)
    
    # Calculate overall train:test ratio as proportion
    overall_ratio_str = calculate_ratio_proportion(total_train_images, total_test_images)
    
    # Print summary
    print("SUMMARY:")
    print(f"{'TRAIN TOTAL':<15} {'Train':<6} {total_train_images:<8} {total_train_annotations:<12} {total_train_pose:<10} {(total_train_pose/total_train_annotations*100):<9.3f}% {'':<15} {'':<15}")
    print(f"{'TEST TOTAL':<15} {'Test':<6} {total_test_images:<8} {total_test_annotations:<12} {total_test_pose:<10} {(total_test_pose/total_test_annotations*100):<9.3f}% {'':<15} {'':<15}")
    print(f"{'GRAND TOTAL':<15} {'Both':<6} {total_train_images+total_test_images:<8} {total_train_annotations+total_test_annotations:<12} {total_train_pose+total_test_pose:<10} {((total_train_pose+total_test_pose)/(total_train_annotations+total_test_annotations)*100):<9.3f}% {overall_ratio_str:<15} T:{total_train_images},Te:{total_test_images}")
    print("="*110)
    
    # Print dataset totals (train + test combined) for easy reference
    print("\nDATASET TOTALS (Train + Test Combined) WITH TRAIN:TEST RATIOS:")
    print("="*110)
    print(f"{'Dataset':<15} {'Images':<8} {'Total Ann.':<12} {'Pose Ann.':<10} {'Pose %':<10} {'Train:Test Ratio':<15} {'Debug Info':<15}")
    print("-"*110)
    
    for subdataset in sorted(subdataset_totals.keys()):
        totals = subdataset_totals[subdataset]
        combined_pose_pct = (totals['pose_annotations'] / totals['total_annotations'] * 100) if totals['total_annotations'] > 0 else 0
        
        # Calculate ratio for this subdataset as proportion
        ratio_str = calculate_ratio_proportion(totals['train_images'], totals['test_images'])
            
        print(f"{subdataset:<15} {totals['images']:<8} {totals['total_annotations']:<12} {totals['pose_annotations']:<10} {combined_pose_pct:<9.3f}% {ratio_str:<15} T:{totals['train_images']},Te:{totals['test_images']}")
    
    print("="*110)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = "subdataset_annotation_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create a pivot table for better visualization
    pivot_df = df.pivot(index='subdataset', columns='split', values=['images', 'total_annotations', 'pose_annotations'])
    pivot_output_file = "subdataset_annotation_pivot.csv"
    pivot_df.to_csv(pivot_output_file)
    print(f"Pivot table saved to: {pivot_output_file}")
    
    # Create a separate CSV with train/test ratios for each subdataset
    ratio_data = []
    for subdataset in sorted(subdataset_totals.keys()):
        totals = subdataset_totals[subdataset]
        ratio_str = calculate_ratio_proportion(totals['train_images'], totals['test_images'])
        
        ratio_data.append({
            'subdataset': subdataset,
            'train_images': totals['train_images'],
            'test_images': totals['test_images'],
            'total_images': totals['images'],
            'train_test_ratio_proportion': ratio_str,
            'train_percentage': (totals['train_images'] / totals['images'] * 100) if totals['images'] > 0 else 0,
            'test_percentage': (totals['test_images'] / totals['images'] * 100) if totals['images'] > 0 else 0
        })
    
    ratio_df = pd.DataFrame(ratio_data)
    ratio_output_file = "subdataset_train_test_ratios.csv"
    ratio_df.to_csv(ratio_output_file, index=False)
    print(f"Train/test ratios saved to: {ratio_output_file}")
    
    return results


def calculate_dataset_stats():
    """
    Calculate the total images for each dataset and their percentages.
    
    Returns:
        tuple: (sorted_datasets_dict, labels_list, sizes_list, percentages_list, total_images)
    """
    train_folder = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_train_datasets"
    test_folder = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets"
    
    # Get all subdataset names from train folder
    train_files = [f for f in os.listdir(train_folder) if f.endswith('_train.json')]
    subdatasets = [f.replace('_train.json', '') for f in train_files]
    subdatasets.sort()
    
    # Collect data for each subdataset
    dataset_images = {}
    total_all_images = 0
    
    for subdataset in subdatasets:
        total_images = 0
        
        # Count train images
        train_file = os.path.join(train_folder, f"{subdataset}_train.json")
        if os.path.exists(train_file):
            _, _, train_images = count_annotations(train_file)
            total_images += train_images
        
        # Count test images
        test_file = os.path.join(test_folder, f"{subdataset}_test.json")
        if os.path.exists(test_file):
            _, _, test_images = count_annotations(test_file)
            total_images += test_images
        
        if total_images > 0:
            dataset_images[subdataset] = total_images
            total_all_images += total_images
    
    # Sort datasets by image count (descending)
    sorted_datasets = dict(sorted(dataset_images.items(), key=lambda x: x[1], reverse=True))
    
    # Prepare data lists
    labels = list(sorted_datasets.keys())
    sizes = list(sorted_datasets.values())
    percentages = [size/total_all_images*100 for size in sizes]
    
    # Print summary
    print("\n" + "="*80)
    print("DATASET STATISTICS CALCULATED")
    print("="*80)
    print(f"Total datasets: {len(sorted_datasets)}")
    print(f"Total images: {total_all_images:,}")
    print("\nDataset breakdown:")
    for label, size, pct in zip(labels, sizes, percentages):
        # Use more precise formatting for small percentages
        if pct < 0.1:
            pct_str = f"{pct:>7.3f}%"
        elif pct < 1.0:
            pct_str = f"{pct:>7.2f}%"
        else:
            pct_str = f"{pct:>7.1f}%"
        print(f"  {label.upper():<15}: {size:>8,} images ({pct_str})")
    print("="*80)
    
    return sorted_datasets, labels, sizes, percentages, total_all_images


def create_pie_chart(labels, sizes, percentages, total_images, custom_order=None, start_angle=90):
    """
    Create a beautiful pie chart for dataset distribution.
    Uses tiered minimum display widths for small datasets while removing all labels from the chart.
    
    Args:
        labels (list): Dataset names
        sizes (list): Number of images per dataset
        percentages (list): Percentage values
        total_images (int): Total number of images
        custom_order (list, optional): Custom order for datasets. If None, uses the original order.
        start_angle (int): Starting angle for the pie chart (default: 90)
    
    Returns:
        str: Success message with saved file names
    """
    # Apply custom order if provided
    if custom_order is not None:
        # Reorder the data according to custom_order
        ordered_data = []
        for dataset_name in custom_order:
            if dataset_name in labels:
                idx = labels.index(dataset_name)
                ordered_data.append((labels[idx], sizes[idx], percentages[idx]))
        
        # Update the lists with the new order
        labels, sizes, percentages = zip(*ordered_data)
        labels, sizes, percentages = list(labels), list(sizes), list(percentages)
    
    # Create adjusted sizes for visualization using tiered minimum widths
    real_percentages = percentages.copy()  # Keep real percentages for reference
    adjusted_sizes = []
    
    # Define tiered minimum display percentages
    def get_min_display_percent(pct):
        if pct < 0.25:
            return 0.25
        elif pct < 0.5:
            return 0.5
        elif pct < 1.0:
            return 1.0
        else:
            return pct
    
    # Calculate adjusted sizes
    total_adjustment = 0
    small_datasets_info = []
    
    for i, pct in enumerate(percentages):
        min_display_pct = get_min_display_percent(pct)
        
        if pct < min_display_pct:
            # Small dataset: use minimum display percentage
            adjusted_size = total_images * (min_display_pct / 100)
            adjusted_sizes.append(adjusted_size)
            total_adjustment += adjusted_size - sizes[i]
            small_datasets_info.append(f"{labels[i].upper()}({pct:.3f}%→{min_display_pct}%)")
        else:
            adjusted_sizes.append(sizes[i])
    
    # Adjust large datasets proportionally to compensate for the adjustment
    if total_adjustment > 0:
        large_datasets_total = sum(sizes[i] for i, pct in enumerate(percentages) 
                                 if pct >= get_min_display_percent(pct))
        
        if large_datasets_total > 0:
            for i, pct in enumerate(percentages):
                min_display_pct = get_min_display_percent(pct)
                if pct >= min_display_pct:
                    # Reduce large datasets proportionally
                    reduction_factor = total_adjustment / large_datasets_total
                    adjusted_sizes[i] = sizes[i] * (1 - reduction_factor)
    
    # Set up the plot with publication-quality settings
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times', 'serif'],
        'axes.linewidth': 1.2,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Utility: slightly increase saturation/lightness to make tones more vivid while keeping Morandi character
    def adjust_hex_saturation_lightness(hex_color, sat_scale=1.15, light_scale=1.05):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = max(0.0, min(1.0, s * sat_scale))
        l = max(0.0, min(1.0, l * light_scale))
        rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
        return '#%02x%02x%02x' % (int(rr * 255), int(gg * 255), int(bb * 255))

    # Generate evenly spaced hues (HUSL) to avoid similar colors anywhere, then apply Morandi-style adjustment
    num_labels = len(labels)
    base_palette = sns.color_palette("husl", n_colors=num_labels)
    # Convert to hex for adjustment
    base_hex = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for (r, g, b) in base_palette]

    # Make palette vivid yet Morandi-inspired (soft but bright)
    vivid_palette = [adjust_hex_saturation_lightness(c, sat_scale=1.25, light_scale=1.10) for c in base_hex]

    # Interleave indices to increase contrast between neighbors for arbitrary label orders
    left_index = 0
    right_index = num_labels - 1
    interleaved_indices = []
    while left_index <= right_index:
        interleaved_indices.append(left_index)
        if left_index != right_index:
            interleaved_indices.append(right_index)
        left_index += 1
        right_index -= 1
    interleaved_colors = [vivid_palette[i] for i in interleaved_indices]

    # Map colors to labels in their current order using the interleaved palette
    colors_by_label = {}
    for position, dataset_label in enumerate(labels):
        colors_by_label[dataset_label] = interleaved_colors[position % num_labels]

    # Gentle but distinct overrides for key datasets (also slightly more vivid)
    explicit_overrides = {
        'oap': adjust_hex_saturation_lightness('#C48E85', sat_scale=1.35, light_scale=1.12),  # brighter terracotta (red)
        'omc': adjust_hex_saturation_lightness('#8BA6B7', sat_scale=1.35, light_scale=1.12),  # brighter steel blue (blue)
        'ak':  adjust_hex_saturation_lightness('#9BBE8C', sat_scale=1.30, light_scale=1.12),  # soft vivid sage (green)
        'oms': adjust_hex_saturation_lightness('#B7A1C5', sat_scale=1.30, light_scale=1.12),  # soft vivid lavender (purple)
        'ap10k': adjust_hex_saturation_lightness('#E7C26A', sat_scale=1.30, light_scale=1.12), # soft vivid mustard (yellow)
    }
    for key, hex_color in explicit_overrides.items():
        if key in colors_by_label:
            colors_by_label[key] = hex_color

    colors = [colors_by_label[dataset_label] for dataset_label in labels]
    
    # Create the pie chart using adjusted sizes for display (NO LABELS OR PERCENTAGES)
    wedges, texts, autotexts = ax.pie(adjusted_sizes, labels=None, autopct='',
                                     colors=colors, startangle=start_angle,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'},
                                     wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
    
    # Create custom labels with dataset names and REAL percentages for legend only
    legend_labels = []
    
    for i, (label, size, real_pct) in enumerate(zip(labels, sizes, real_percentages)):
        # Use more precise formatting for small percentages
        if real_pct < 0.1:
            pct_str = f"{real_pct:.3f}%"
        elif real_pct < 1.0:
            pct_str = f"{real_pct:.2f}%"
        else:
            pct_str = f"{real_pct:.2f}%"
        
        if label == 'ak':
            label = 'AnimalKingdom'
        elif label == 'oap':
            label = 'OpenApePose'
        elif label == 'mbw':
            label = 'MBW'
        elif label == 'oms':
            label = 'OpenMonkeyStudio'
        elif label == 'deepwild':
            label = 'DeepWild'
        elif label == 'mit':
            label = 'MIT-Marmoset'
        elif label == 'mp':
            label = 'MacaquePose'
        elif label == 'ap10k':
            label = 'AP-10K'
        elif label == 'omc':
            label = 'OpenMonkeyChallenge'
        elif label == 'aptv2':
            label = 'APTv2'
        elif label == 'anipose':
            label = 'Animal-Pose'
        elif label == 'lote':
            label = 'LoTE-Animal'
        legend_labels.append(f'{label}: {size:,} ({pct_str})')
    
    # Add legend outside the pie chart (closer to the pie)
    # ax.legend(wedges, legend_labels, 
    #         #  title="Dataset Distribution", 
    #          loc="center", 
    #          bbox_to_anchor=(0.95, 0, 0.5, 1),  # Moved closer: 1 -> 0.85
    #          fontsize=13,          # Increase label font size: 10 -> 12
    #          title_fontsize=16,    # Increase title font size: 12 -> 14
    #          frameon=True,
    #          fancybox=True,
    #          shadow=True,
    #          labelspacing=1.8,     # Increase vertical spacing between label rows
    #          handletextpad=0.8)    # Increase spacing between color blocks and text
    
    # Set title
    title_text = 'Distribution of Images Across Primate Datasets\n' + f'Total Images: {total_images:,}'
    if small_datasets_info:
        title_text += f'\n(Enhanced visibility for small datasets)'
    # ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    
    # Ensure the pie chart is circular
    ax.axis('equal')
    
    # Remove any extra whitespace
    plt.tight_layout()
    
    # Save the figure in multiple formats for publication
    output_files = []
    
    # High-resolution PNG for papers
    png_file = "dataset_images_distribution.png"
    plt.savefig(png_file, format='png', dpi=500, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    output_files.append(png_file)
    
    # Vector format (PDF) for scalable publication quality
    # pdf_file = "dataset_images_distribution.pdf"
    # plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight',
    #             facecolor='white', edgecolor='none')
    # output_files.append(pdf_file)
    
    # EPS format for some journals
    # eps_file = "dataset_images_distribution.eps"
    # plt.savefig(eps_file, format='eps', dpi=3000, bbox_inches='tight',
    #             facecolor='white', edgecolor='none')
    # output_files.append(eps_file)
    
    # Show the plot
    plt.show()
    
    # Print summary
    print("\n" + "="*80)
    print("PIE CHART GENERATED SUCCESSFULLY")
    print("="*80)
    if small_datasets_info:
        print("Small datasets enhanced with tiered minimum widths:")
        for info in small_datasets_info:
            print(f"  - {info}")
        print("Minimum width rules:")
        print("  - < 0.25% → displayed as 0.25%") 
        print("  - < 0.5% → displayed as 0.5%")
        print("  - < 1.0% → displayed as 1.0%")
    print(f"Files saved:")
    for file in output_files:
        print(f"  - {file}")
    print("="*80)
    
    return output_files


def vis_ratio_subdataset():
    """
    Complete workflow: Calculate dataset statistics and create visualization.
    This function calls both calculate_dataset_stats() and create_pie_chart().
    
    For manual control, you can call these functions separately:
    1. stats, labels, sizes, percentages, total = calculate_dataset_stats()
    2. create_pie_chart(labels, sizes, percentages, total, custom_order=['oms', 'omc', ...])
    """
    # Step 1: Calculate statistics
    sorted_datasets, labels, sizes, percentages, total_images = calculate_dataset_stats()
    print("labels:", labels)
    print("percentages:", percentages)
    custom_order = ['ak', 'oap', 'mbw', 'oms', 'deepwild', 'mit', 'mp', 'ap10k', 'omc', 'aptv2', 'anipose', 'lote']
    # print("labels:", labels)
    # Step 2: Create visualization with small dataset highlighting (rotated 25° anticlockwise)
    output_files = create_pie_chart(labels, sizes, percentages, total_images, 
                                   custom_order=custom_order, 
                                   start_angle=130)  # 90 + 25 = 115 degrees
    
    return sorted_datasets

if __name__ == "__main__":
    # Run the analysis
    print("Running dataset analysis...")
    results = analyze_subdatasets()
    
    # print("\n" + "="*80)
    # print("Analysis complete! Now generating visualization...")
    # print("="*80)
    
    # Generate the pie chart visualization
    dataset_stats = vis_ratio_subdataset()
    
    print("\nBoth analysis and visualization completed successfully!")
    
    print("\n" + "="*80)
    print("MANUAL CONTROL EXAMPLE:")
    print("="*80)
    print("To manually control the pie chart order, use these functions separately:")
    print("1. stats, labels, sizes, percentages, total = calculate_dataset_stats()")
    print("2. create_pie_chart(labels, sizes, percentages, total,")
    print("                   custom_order=['oms', 'omc', 'oap', 'mp', 'aptv2', ...])")
    print("="*80)