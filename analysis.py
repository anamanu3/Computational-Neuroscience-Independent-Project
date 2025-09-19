"""
analysis.py - Analyze and plot Wong-Wang simulation results

This script takes CSV files from simulate.py and makes publication-ready plots.
Handles both basic experiments and parameter sweeps.

Usage examples:
    python analysis.py                                    # plot default data file
    python analysis.py --input my_results.csv            # plot specific file
    python analysis.py --savefigs                        # save plots instead of showing
    python analysis.py --input sweep_results.csv --type sweep  # analyze parameter sweep
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# make plots look decent without seaborn
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_basic_csv(path):
    """Load results from a basic psychometric experiment."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'coherence': float(row['coherence']),
                'p_correct': float(row['p_correct']),
                'mean_rt': float(row['mean_rt_ms']),
                'decision_rate': float(row['decision_rate']),
                'rt_std': float(row.get('rt_std', 0))  # might not exist in old files
            })
    
    # sort by coherence
    data.sort(key=lambda x: x['coherence'])
    return data


def load_sweep_csv(path):
    """Load results from parameter sweep experiments."""
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        for row in reader:
            entry = {
                'coherence': float(row['coherence']),
                'p_correct': float(row['p_correct']),
                'mean_rt': float(row['mean_rt_ms']),
                'decision_rate': float(row['decision_rate'])
            }
            
            # figure out what parameter was swept
            if 'threshold' in headers:
                entry['param'] = float(row['threshold'])
                entry['param_name'] = 'threshold'
            elif 'sigma' in headers:
                entry['param'] = float(row['sigma'])
                entry['param_name'] = 'sigma'
            
            data.append(entry)
    
    return data


def plot_basic_results(data, save_path=None):
    """Make standard psychometric + chronometric plots."""
    coherences = [d['coherence'] * 100 for d in data]  # convert to percentage
    accuracies = [d['p_correct'] for d in data]
    rts = [d['mean_rt'] for d in data]
    rt_stds = [d['rt_std'] for d in data]
    decision_rates = [d['decision_rate'] for d in data]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # psychometric curve (top left)
    axes[0,0].plot(coherences, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0,0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[0,0].set_title('Psychometric Function', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Motion Coherence (%)')
    axes[0,0].set_ylabel('Fraction Correct')
    axes[0,0].set_ylim(0.4, 1.05)
    axes[0,0].grid(True, alpha=0.3)
    
    # chronometric curve (top right)
    if any(std > 0 for std in rt_stds):
        axes[0,1].errorbar(coherences, rts, yerr=rt_stds, fmt='o-', 
                          linewidth=2, markersize=8, capsize=5)
    else:
        axes[0,1].plot(coherences, rts, 'o-', linewidth=2, markersize=8)
    
    axes[0,1].set_title('Chronometric Function', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Motion Coherence (%)')
    axes[0,1].set_ylabel('Mean Reaction Time (ms)')
    axes[0,1].grid(True, alpha=0.3)
    
    # decision rate (bottom left)
    axes[1,0].plot(coherences, decision_rates, 'o-', linewidth=2, markersize=8, color='red')
    axes[1,0].set_title('Decision Rate', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Motion Coherence (%)')
    axes[1,0].set_ylabel('Fraction of Decisions Made')
    axes[1,0].set_ylim(0, 1.05)
    axes[1,0].grid(True, alpha=0.3)
    
    # speed-accuracy tradeoff (bottom right)
    # only use trials where decisions were made
    valid_idx = [i for i, dr in enumerate(decision_rates) if dr > 0.5]
    if valid_idx:
        acc_subset = [accuracies[i] for i in valid_idx]
        rt_subset = [rts[i] for i in valid_idx]
        coh_subset = [coherences[i] for i in valid_idx]
        
        scatter = axes[1,1].scatter(rt_subset, acc_subset, c=coh_subset, 
                                   s=100, cmap='viridis', alpha=0.7)
        axes[1,1].set_title('Speed-Accuracy Tradeoff', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Mean Reaction Time (ms)')
        axes[1,1].set_ylabel('Fraction Correct')
        cbar = plt.colorbar(scatter, ax=axes[1,1])
        cbar.set_label('Coherence (%)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_parameter_sweep(data, save_path=None):
    """Plot results from parameter sweep experiments."""
    param_name = data[0]['param_name']
    param_values = sorted(list(set(d['param'] for d in data)))
    coherences = sorted(list(set(d['coherence'] for d in data)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # plot accuracy for each parameter value
    for param_val in param_values:
        subset = [d for d in data if d['param'] == param_val]
        subset.sort(key=lambda x: x['coherence'])
        
        coh = [d['coherence'] * 100 for d in subset]
        acc = [d['p_correct'] for d in subset]
        
        axes[0].plot(coh, acc, 'o-', label=f'{param_name}={param_val}', 
                    linewidth=2, markersize=6)
    
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_title(f'Psychometric Function vs {param_name.title()}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Motion Coherence (%)')
    axes[0].set_ylabel('Fraction Correct')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # plot reaction times
    for param_val in param_values:
        subset = [d for d in data if d['param'] == param_val]
        subset.sort(key=lambda x: x['coherence'])
        
        coh = [d['coherence'] * 100 for d in subset]
        rt = [d['mean_rt'] for d in subset]
        
        axes[1].plot(coh, rt, 'o-', label=f'{param_name}={param_val}', 
                    linewidth=2, markersize=6)
    
    axes[1].set_title(f'Chronometric Function vs {param_name.title()}', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Motion Coherence (%)')
    axes[1].set_ylabel('Mean Reaction Time (ms)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter sweep figure to {save_path}")
    else:
        plt.show()
    
    return fig


def print_summary(data):
    """Print a nice summary of the results."""
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    if isinstance(data, list) and 'param_name' in data[0]:
        # parameter sweep
        param_name = data[0]['param_name']
        param_values = sorted(list(set(d['param'] for d in data)))
        coherences = sorted(list(set(d['coherence'] for d in data)))
        
        print(f"Parameter sweep: {param_name}")
        print(f"Parameter values: {param_values}")
        print(f"Coherences tested: {[c*100 for c in coherences]}")
        print(f"Total conditions: {len(param_values) * len(coherences)}")
        
    else:
        # basic experiment
        coherences = [d['coherence'] * 100 for d in data]
        max_acc = max(d['p_correct'] for d in data)
        
        # find fastest RT from trials that actually made decisions
        valid_rts = [d['mean_rt'] for d in data if d['decision_rate'] > 0.5 and not np.isnan(d['mean_rt'])]
        min_rt = min(valid_rts) if valid_rts else float('nan')
        
        print(f"Coherences tested: {coherences}%")
        print(f"Best accuracy: {max_acc:.1%}")
        if not np.isnan(min_rt):
            print(f"Fastest mean RT: {min_rt:.0f}ms")
        else:
            print("No valid reaction times found")
        
        print("\nDetailed results:")
        print("Coherence | Accuracy | Mean RT | Dec. Rate")
        print("-" * 42)
        for d in data:
            print(f"{d['coherence']*100:8.1f}% | {d['p_correct']:7.1%} | "
                  f"{d['mean_rt']:6.0f}ms | {d['decision_rate']:8.1%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Wong-Wang simulation results")
    
    parser.add_argument("--input", type=str, default="data/results.csv",
                       help="CSV file to analyze")
    parser.add_argument("--savefigs", action="store_true",
                       help="save plots instead of showing them")
    parser.add_argument("--type", choices=["basic", "sweep"], default="basic",
                       help="type of analysis (basic or parameter sweep)")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="directory to save figures")
    
    args = parser.parse_args()
    
    # check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        print("Make sure you've run simulate.py first to generate data.")
        return
    
    print(f"Loading data from {input_path}...")
    
    # load appropriate data format
    if args.type == "sweep":
        data = load_sweep_csv(input_path)
        if not data:
            print("No data found! Check your CSV file.")
            return
            
        print_summary(data)
        
        # make plots
        if args.savefigs:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "parameter_sweep.png"
        else:
            save_path = None
            
        plot_parameter_sweep(data, save_path)
        
    else:
        data = load_basic_csv(input_path)
        if not data:
            print("No data found! Check your CSV file.")
            return
            
        print_summary(data)
        
        # make plots
        if args.savefigs:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / "psychometric_analysis.png"
        else:
            save_path = None
            
        plot_basic_results(data, save_path)


if __name__ == "__main__":
    main()
    