import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import os

#  make a figure to show the number of each action. 

def vis_actions(actions_stats_path):
    with open(actions_stats_path, 'r') as f:
        actions_stats = json.load(f)
    return actions_stats


def plot_actions_counts(actions_stats_path, out_path=None, figsize=(10, 4.8), dpi=400):
    # load stats
    with open(actions_stats_path, 'r') as f:
        stats = json.load(f)
    counts = stats.get('counts', {})

    # sort actions by name (alphabetical)
    items = sorted(counts.items(), key=lambda kv: kv[0])
    action_names = [k for k, _ in items]
    numbers = [v for _, v in items]

    # plot
    # publication-friendly settings
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['savefig.dpi'] = dpi

    fig, ax = plt.subplots(figsize=figsize)
    bar_positions = range(len(action_names))
    bars = ax.bar(bar_positions, numbers, width=0.5, color='#69B3FF', edgecolor='#2F6FB2', linewidth=0.6)

    # axis formatting
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(action_names, rotation=35, ha='right')
    ax.set_xlabel('action_names', labelpad=8)
    ax.set_ylabel('number', labelpad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # annotate counts on bars if not too crowded
    if len(numbers) <= 25:
        for rect, val in zip(bars, numbers):
            height = rect.get_height()
            ax.annotate(f'{int(val)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, color='#1f1f1f')

    fig.tight_layout()

    # save
    if out_path is None:
        base_dir = os.path.dirname(actions_stats_path) or '.'
        out_path = os.path.join(base_dir, 'test_actions_counts.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    # also save vector version for papers
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    return out_path


def cal_ratio_for_each_actions(actions_stats_path):
    with open(actions_stats_path, 'r') as f:
        stats = json.load(f)
    counts = stats.get('counts', {})
    total = sum(counts.values())
    ratios = {k: v / total for k, v in counts.items()}
    return ratios


def cal_avg_mpjpe(mpjpe_per_action):
    # avg_mpjpe = {k: mpjpe_per_action[k] * ratio[k] for k in mpjpe_per_action.keys()}
    avg_mpjpe = {k: mpjpe_per_action[k] * 1.0/15 for k in mpjpe_per_action.keys()}
    avg_mpjpe = sum(avg_mpjpe.values())
    return avg_mpjpe

mpjpe_per_action_PerturbPE = {
    'Directions': 45.9,
    'Discussion': 50.1,
    'Eating': 41.2,
    'Greeting': 43.2,
    'Phoning': 52.7,
    'Photo': 57.4,
    'Posing': 43.0,
    'Purchases': 38.4,
    'Sitting': 55.4,
    'SittingDown': 61.8,
    'Smoking': 45.8,
    'Waiting': 46.8,
    'WalkDog': 48.5,
    'Walking': 38.9,
    'WalkTogether': 42.8
}

mpjpe_per_action_CFM = {
    'Directions': 46.20,
    'Discussion': 49.75,
    'Eating': 46.34,
    'Greeting': 49.78,
    'Phoning': 51.29,
    'Photo': 57.77,
    'Posing': 47.68,
    'Purchases': 45.45,
    'Sitting': 58.46,
    'SittingDown': 61.79,
    'Smoking': 50.23,
    'Waiting': 47.14,
    'WalkDog': 52.68,
    'Walking': 39.31,
    'WalkTogether': 42.06
}

if __name__ == '__main__':
    
    default_json = 'test_actions_stats.json'
    
    # if os.path.exists(default_json):
    #     saved = plot_actions_counts(default_json)
    #     print(f'Saved figure to {saved}')
    # else:
    #     print(f'Not found: {default_json}')
    # ratios = cal_ratio_for_each_actions(default_json)
    # print(ratios)
    avg_mpjpe = cal_avg_mpjpe(mpjpe_per_action_PerturbPE) # paper:50.8; 47.46, 如果按比例，48.15；
    # print(avg_mpjpe)
    # avg_mpjpe = cal_avg_mpjpe(mpjpe_per_action_CFM)
    print(avg_mpjpe)