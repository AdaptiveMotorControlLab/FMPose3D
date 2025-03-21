import matplotlib.pyplot as plt
import numpy as np
import os

# Store dataset performance results in a structured dictionary
dataset_results = {
    'ak': {
        'original': 39.679,
        # 'pfm': 45.81,
         'pfm': 57.50,
        # 'pfm_V2': 53.88
    },
    'ap10k': {
        'original': 47.442,
        'pfm': 51.64
    },
    'mbw': {
        'original': 65.683,
        'pfm': 81.10
    },
    'mp': {
        'original': 82.75,
        'pfm': 83.089
    },
    'oms': {
        'original': 97.02,
        'pfm': 96.76
    },
    'kinka': {
        'original': 63.76,
        'pfm': 78.18
    },
    'lote': {
        'original': 49.757,
        'pfm': 51.8839,
        # 'pfm_V2': 51.65
    },
    'aptv2': {
        'original': 51.45, # correct the performance
        'pfm': 61.001
    },
    'ap10k': {
        'original': 47.442,
        'pfm': 63.943
    },
    'mit': {
        'original': 80.5,
        'pfm': 86.91
    },
    'deepwild': {
        'original': 74.79,
        'pfm': 90.353
    },
    'oap': {
        'original': 72.793,
        # 'pfm': 71.41
        'pfm': 50.52
    },
    'chimpact': {
        'original': 25.87,
        'pfm': 21.14
    },
    'riken': {
    'original': 26.161,
    'pfm': 6.960,
    # 'pfm_V2': 6.737
    },
    # 'omc': {
    #     'original': 97.83
    # }
}

# 添加样本数量数据
dataset_samples = {
    'oms': 127458,
    'oap': 33644,
    'mbw': 45,
    'lote': 6824,
    'ak': 808,
    'ap10k': 917,
    'aptv2': 10249,
    'chimpact': 37861,
    'kinka': 47,
    'riken': 5878,
    'mp': 15659,
    'mit':3960,
    'deepwild':4676
}

def plot_performance_comparison(results, samples, save_path='plots'):
    """Plot performance comparison between original and PFM models."""
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 首先准备所有数据
    filtered_results = {k: v for k, v in results.items() 
                       if 'original' in v and 'pfm' in v}
    
    datasets = list(filtered_results.keys())
    x = np.arange(len(datasets))  # 在这里定义x
    
    original_scores = [filtered_results[d]['original'] for d in datasets]
    pfm_scores = [filtered_results[d]['pfm'] for d in datasets]
    sample_counts = [samples[d] for d in datasets]
    
    # 检查 pfm_V2 数据
    has_pfm_v2 = any('pfm_V2' in v for v in filtered_results.values())
    if has_pfm_v2:
        pfm_v2_scores = []
        for d in datasets:
            if 'pfm_V2' in filtered_results[d]:
                pfm_v2_scores.append(filtered_results[d]['pfm_V2'])
            else:
                pfm_v2_scores.append(None)
    
    # 2. 创建图形和布局
    plt.style.use('default')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # 3. 绘制样本数量图
    ax1.bar(x, sample_counts, color='gray', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=10, rotation=45, ha='right', fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylabel('Sample Count', fontsize=10)
    
    # 调整子图之间的间距，给更大的标签留出空间
    plt.subplots_adjust(hspace=0.4)  # 增加间距以适应更大的字体
    
    # 添加样本数量标签
    for i, count in enumerate(sample_counts):
        ax1.text(i, count, f'{count:,}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=10,  # 增加字体大小
                 fontweight='bold',  # 添加粗体
                 rotation=45)
    
    # 4. 绘制性能对比图
    bar_width = 0.2 if has_pfm_v2 else 0.3
    
    color1 = '#3498db'
    color2 = '#2ecc71'
    color3 = '#e74c3c'
    
    bars1 = ax2.bar(x - 1.2*bar_width, original_scores, bar_width, 
                    label='Single-Model', color=color1, alpha=0.8,
                    edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x, pfm_scores, bar_width, 
                    label='PFM', color=color2, alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    if has_pfm_v2:
        valid_v2_indices = [i for i, score in enumerate(pfm_v2_scores) if score is not None]
        valid_v2_scores = [pfm_v2_scores[i] for i in valid_v2_indices]
        valid_v2_positions = [x[i] + 1.2*bar_width for i in valid_v2_indices]
        
        bars3 = ax2.bar(valid_v2_positions, valid_v2_scores, bar_width,
                       label='PFM V2', color=color3, alpha=0.8,
                       edgecolor='black', linewidth=1)
    
    # 5. 设置性能图的标签和样式
    ax2.set_xlabel('Datasets', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Performance Score (mAP)', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title('Performance Comparison across Models', 
                  fontsize=14, fontweight='bold', pad=20)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=10, fontweight='bold', rotation=45, ha='right')
    ax2.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=6)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    ax2.legend(fontsize=10, frameon=True, fancybox=True, 
              shadow=True, loc='upper left',
              bbox_to_anchor=(0.02, 0.98))
    
    # 6. 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}',
                    ha='center', 
                    va='bottom', 
                    fontsize=9,  # 增加字体大小（从7增加到9）
                    fontweight='bold', 
                    color='#2c3e50',
                    rotation=45)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    if has_pfm_v2:
        add_value_labels(bars3)
    
    plt.tight_layout()
    
    output_path = os.path.join(save_path, 'performance_comparison_V82.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_performance_comparison(dataset_results, dataset_samples, save_path='visualization_results')
