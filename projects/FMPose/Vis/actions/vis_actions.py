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

if __name__ == '__main__':
    default_json = 'test_actions_stats.json'
    if os.path.exists(default_json):
        saved = plot_actions_counts(default_json)
        print(f'Saved figure to {saved}')
    else:
        print(f'Not found: {default_json}')