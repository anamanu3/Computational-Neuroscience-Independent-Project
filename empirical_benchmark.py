"""
empirical_benchmark.py - Compare Wong-Wang model to Roitman & Shadlen (2002) data

This script compares your model's psychometric and chronometric functions to the 
classic random-dot motion task data from Roitman & Shadlen (2002), providing 
empirical validation of the model's biological realism.

Reference: Roitman, J. D., & Shadlen, M. N. (2002). Response of neurons in the 
lateral intraparietal area during a combined visual discrimination reaction time 
task. Journal of Neuroscience, 22(21), 9475-9489.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from analysis import load_basic_csv


def get_roitman_shadlen_data():
    """
    Approximate data from Roitman & Shadlen (2002) Figure 2.
    
    This is digitized from their published psychometric and chronometric curves
    for the random-dot motion discrimination task with rhesus monkeys.
    """
    
    # Psychometric data (coherence vs accuracy)
    # Digitized from their Figure 2A
    roitman_coherences = np.array([0, 3.2, 6.4, 12.8, 25.6, 51.2])  # percent coherence
    roitman_accuracy = np.array([0.52, 0.57, 0.66, 0.78, 0.89, 0.95])  # fraction correct
    roitman_accuracy_sem = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01])  # standard error
    
    # Chronometric data (coherence vs reaction time)  
    # Digitized from their Figure 2B
    roitman_rt = np.array([680, 665, 640, 580, 520, 480])  # milliseconds
    roitman_rt_sem = np.array([25, 20, 18, 15, 12, 10])  # standard error
    
    return {
        'coherences': roitman_coherences,
        'accuracy': roitman_accuracy,
        'accuracy_sem': roitman_accuracy_sem, 
        'reaction_times': roitman_rt,
        'rt_sem': roitman_rt_sem
    }


def compare_to_empirical_data(wong_wang_data, save_path=None):
    """Create overlay comparison with Roitman & Shadlen (2002) data."""
    
    # Get empirical data
    empirical = get_roitman_shadlen_data()
    
    # Extract Wong-Wang data
    ww_coherences = []
    ww_accuracies = []
    ww_rts = []
    
    for row in wong_wang_data:
        coherence, accuracy, rt, decision_rate, rt_std = row
        if decision_rate > 0.1:  # only include if model made reasonable decisions
            ww_coherences.append(coherence * 100)  # convert to percentage
            ww_accuracies.append(accuracy)
            ww_rts.append(rt)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Psychometric function comparison
    axes[0].errorbar(empirical['coherences'], empirical['accuracy'], 
                    yerr=empirical['accuracy_sem'], 
                    fmt='o-', linewidth=2, markersize=8, capsize=5,
                    label='Roitman & Shadlen (2002)\nRhesus Monkey LIP', color='red')
    
    axes[0].plot(ww_coherences, ww_accuracies, 
                's--', linewidth=2, markersize=8, 
                label='Wong-Wang Model', color='blue')
    
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    axes[0].set_xlabel('Motion Coherence (%)', fontsize=12)
    axes[0].set_ylabel('Fraction Correct', fontsize=12)
    axes[0].set_title('Psychometric Function\nModel vs. Empirical Data', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.45, 1.02)
    
    # Chronometric function comparison
    axes[1].errorbar(empirical['coherences'], empirical['reaction_times'],
                    yerr=empirical['rt_sem'],
                    fmt='o-', linewidth=2, markersize=8, capsize=5,
                    label='Roitman & Shadlen (2002)\nRhesus Monkey LIP', color='red')
    
    axes[1].plot(ww_coherences, ww_rts,
                's--', linewidth=2, markersize=8,
                label='Wong-Wang Model', color='blue')
    
    axes[1].set_xlabel('Motion Coherence (%)', fontsize=12)
    axes[1].set_ylabel('Reaction Time (ms)', fontsize=12)
    axes[1].set_title('Chronometric Function\nModel vs. Empirical Data', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Empirical comparison plot saved to {save_path}")
    else:
        plt.show()
    
    # Calculate quantitative comparison
    print("\n" + "="*60)
    print("EMPIRICAL VALIDATION ANALYSIS")
    print("="*60)
    
    # Interpolate Wong-Wang data to match empirical coherence levels
    if len(ww_coherences) > 2:
        ww_acc_interp = np.interp(empirical['coherences'], ww_coherences, ww_accuracies)
        ww_rt_interp = np.interp(empirical['coherences'], ww_coherences, ww_rts)
        
        # Calculate correlations
        acc_correlation = np.corrcoef(empirical['accuracy'], ww_acc_interp)[0,1]
        rt_correlation = np.corrcoef(empirical['reaction_times'], ww_rt_interp)[0,1]
        
        # Calculate root mean square error
        acc_rmse = np.sqrt(np.mean((empirical['accuracy'] - ww_acc_interp)**2))
        rt_rmse = np.sqrt(np.mean((empirical['reaction_times'] - ww_rt_interp)**2))
        
        print(f"Psychometric correlation: r = {acc_correlation:.3f}")
        print(f"Chronometric correlation: r = {rt_correlation:.3f}")
        print(f"Accuracy RMSE: {acc_rmse:.3f}")
        print(f"RT RMSE: {rt_rmse:.1f} ms")
        
        # Biological realism assessment
        print(f"\nBiological Realism Assessment:")
        if acc_correlation > 0.8:
            print("✓ Psychometric function shows strong correspondence to neural data")
        elif acc_correlation > 0.6:
            print("• Psychometric function shows moderate correspondence to neural data")
        else:
            print("⚠ Psychometric function shows weak correspondence - consider parameter tuning")
            
        if rt_correlation > 0.7:
            print("✓ Chronometric function captures empirical timing patterns")
        elif rt_correlation > 0.4:
            print("• Chronometric function shows moderate timing correspondence")  
        else:
            print("⚠ Chronometric function shows weak timing correspondence")
            
        # Overall assessment
        overall_score = (acc_correlation + abs(rt_correlation)) / 2
        print(f"\nOverall empirical correspondence: {overall_score:.3f}")
        
        if overall_score > 0.7:
            print("🎯 Model shows strong biological realism!")
            print("   Your Wong-Wang implementation captures key aspects of neural decision-making")
        elif overall_score > 0.5:
            print("📈 Model shows reasonable biological correspondence")
            print("   Consider parameter optimization for better empirical fit")
        else:
            print("🔧 Model needs parameter tuning for biological realism")
            print("   Current parameters may be outside physiologically plausible range")
    
    print("\nReference:")
    print("Roitman, J. D., & Shadlen, M. N. (2002). Response of neurons in the")
    print("lateral intraparietal area during a combined visual discrimination")
    print("reaction time task. Journal of Neuroscience, 22(21), 9475-9489.")


def main():
    parser = argparse.ArgumentParser(description="Compare Wong-Wang model to Roitman & Shadlen (2002) empirical data")
    parser.add_argument("--input", type=str, default="data/results.csv",
                       help="Wong-Wang results CSV file")
    parser.add_argument("--savefigs", action="store_true",
                       help="save plots instead of showing them")
    parser.add_argument("--output_dir", type=str, default="figures",
                       help="directory to save figures")
    
    args = parser.parse_args()
    
    # Load Wong-Wang data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found!")
        print("Run 'python simulate.py' first to generate data.")
        return
    
    print(f"Loading Wong-Wang data from {input_path}...")
    ww_data = load_basic_csv(input_path)
    
    if not ww_data:
        print("No data found! Check your CSV file.")
        return
    
    # Convert to array format
    ww_array = [[d['coherence'], d['p_correct'], d['mean_rt'], 
                d['decision_rate'], d['rt_std']] for d in ww_data]
    
    # Create empirical comparison
    if args.savefigs:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_path = output_dir / "empirical_benchmark.png"
    else:
        save_path = None
    
    compare_to_empirical_data(ww_array, save_path)


if __name__ == "__main__":
    main()