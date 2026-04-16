import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_tracking

def main():
    byte_res = evaluate_tracking('data/simulated_sequence.json', 'outputs/bytetrack_results.json')
    sparse_res = evaluate_tracking('data/simulated_sequence.json', 'outputs/sparsetrack_results.json')
    
    metrics = ['MOTA', 'IDF1']
    byte_vals = [byte_res['MOTA'], byte_res['IDF1']]
    sparse_vals = [sparse_res['MOTA'], sparse_res['IDF1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, byte_vals, width, label='ByteTrack')
    rects2 = ax.bar(x + width/2, sparse_vals, width, label='SparseTrack')
    
    ax.set_ylabel('Scores')
    ax.set_title('Tracking Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('report/images/performance_comparison.png')
    
    # Plot ID Switches
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ['ByteTrack', 'SparseTrack']
    id_sw_vals = [byte_res['ID_Switches'], sparse_res['ID_Switches']]
    
    bars = ax.bar(methods, id_sw_vals, color=['blue', 'orange'])
    ax.set_ylabel('Number of ID Switches')
    ax.set_title('ID Switches Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    fig.tight_layout()
    plt.savefig('report/images/id_switches.png')

if __name__ == '__main__':
    main()
