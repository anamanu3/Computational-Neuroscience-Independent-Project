"""
mechanistic_analysis.py - Explore how key network parameters affect decision-making

This script demonstrates the mechanistic principles of the Wong-Wang model by showing
how inhibition (w_I), background drive (I0), and noise (sigma) affect the balance
between slow reverberation and inhibitory control that underlies decision-making.

The "slow reverberation" refers to the NMDA-mediated recurrent excitation that 
allows evidence integration, while inhibition provides the competitive dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from network import WongWangNetwork, sweep_psychometric


def explore_inhibition(base_network, coherences, n_trials=100, thresh=15):
    """Explore how global inhibition (w_I) affects decision dynamics."""
    
    w_I_values = [0.2, 0.5, 1.0]  # low, medium, high inhibition
    w_I_labels = ['Low Inhibition\n(w_I=0.2)', 'Medium Inhibition\n(w_I=0.5)', 'High Inhibition\n(w_I=1.0)']
    
    results = []
    
    for w_I in w_I_values:
        print(f"  Testing w_I = {w_I}")
        net = WongWangNetwork(w_I=w_I, w_plus=base_network.w_plus, 
                            JA_ext=base_network.JA_ext, sigma=base_network.sigma,
                            a=base_network.a, b=base_network.b)
        
        res = sweep_psychometric(net, coherences=coherences, n_trials=n_trials, thresh=thresh)
        results.append((w_I, res, w_I_labels[w_I_values.index(w_I)]))
    
    return results


def explore_background_drive(base_network, coherences, n_trials=100, thresh=15):
    """Explore how background drive (I0) affects decision dynamics."""
    
    I0_values = [0.25, 0.325, 0.4]  # low, medium, high background
    I0_labels = ['Low Background\n(I0=0.25)', 'Medium Background\n(I0=0.325)', 'High Background\n(I0=0.4)']
    
    results = []
    
    for I0 in I0_values:
        print(f"  Testing I0 = {I0}")
        net = WongWangNetwork(I0=I0, w_I=base_network.w_I,
                            JA_ext=base_network.JA_ext, sigma=base_network.sigma,
                            a=base_network.a, b=base_network.b)
        
        res = sweep_psychometric(net, coherences=coherences, n_trials=n_trials, thresh=thresh)
        results.append((I0, res, I0_labels[I0_values.index(I0)]))
    
    return results


def explore_noise(base_network, coherences, n_trials=100, thresh=15):
    """Explore how neural noise (sigma) affects decision dynamics."""
    
    sigma_values = [0.005, 0.01, 0.02]  # low, medium, high noise
    sigma_labels = ['Low Noise\n(σ=0.005)', 'Medium Noise\n(σ=0.01)', 'High Noise\n(σ=0.02)']
    
    results = []
    
    for sigma in sigma_values:
        print(f"  Testing sigma = {sigma}")
        net = WongWangNetwork(sigma=sigma, w_I=base_network.w_I,
                            JA_ext=base_network.JA_ext, I0=base_network.I0,
                            a=base_network.a, b=base_network.b)
        
        res = sweep_psychometric(net, coherences=coherences, n_trials=n_trials, thresh=thresh)
        results.append((sigma, res, sigma_labels[sigma_values.index(sigma)]))
    
    return results


def plot_mechanistic_effects(inhibition_results, background_results, noise_results, save_path=None):
    """Create three-panel figure showing parameter effects."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    def plot_parameter_effect(results, axes_row, param_name):
        """Helper function to plot one parameter's effects."""
        
        for i, (param_val, res, label) in enumerate(results):
            coherences = res[:, 0] * 100  # convert to percentage
            accuracies = res[:, 1]
            rts = res[:, 2]
            
            # Filter out conditions with very low decision rates
            valid_idx = res[:, 3] > 0.05
            if not any(valid_idx):
                continue
                
            coherences_valid = coherences[valid_idx]
            accuracies_valid = accuracies[valid_idx]
            rts_valid = rts[valid_idx]
            
            # Plot psychometric curves
            axes_row[0].plot(coherences_valid, accuracies_valid, 'o-', 
                           color=colors[i], linewidth=2, markersize=6, label=label)
            
            # Plot chronometric curves
            axes_row[1].plot(coherences_valid, rts_valid, 'o-',
                           color=colors[i], linewidth=2, markersize=6, label=label)
        
        # Format psychometric plot
        axes_row[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes_row[0].set_xlabel('Motion Coherence (%)')
        axes_row[0].set_ylabel('Fraction Correct')
        axes_row[0].set_title(f'{param_name} Effects:\nPsychometric Function')
        axes_row[0].legend(fontsize=9)
        axes_row[0].grid(True, alpha=0.3)
        axes_row[0].set_ylim(0, 1.05)
        
        # Format chronometric plot
        axes_row[1].set_xlabel('Motion Coherence (%)')
        axes_row[1].set_ylabel('Mean RT (ms)')
        axes_row[1].set_title(f'{param_name} Effects:\nChronometric Function')
        axes_row[1].legend(fontsize=9)
        axes_row[1].grid(True, alpha=0.3)
    
    # Plot effects for each parameter
    plot_parameter_effect(inhibition_results, [axes[0,0], axes[1,0]], 'Inhibition (w_I)')
    plot_parameter_effect(background_results, [axes[0,1], axes[1,1]], 'Background Drive (I0)')
    plot_parameter_effect(noise_results, [axes[0,2], axes[1,2]], 'Neural Noise (σ)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mechanistic analysis plot saved to {save_path}")
    else:
        plt.show()
    
    return fig


def print_mechanistic_insights(inhibition_results, background_results, noise_results):
    """Print interpretation of the mechanistic effects."""
    
    print("\n" + "="*70)
    print("MECHANISTIC INSIGHTS: Slow Reverberation vs. Inhibitory Control")
    print("="*70)
    
    print("\n1. INHIBITION EFFECTS (w_I):")
    print("   • Low inhibition → Faster decisions, potential instability")
    print("   • High inhibition → Slower decisions, more conservative")
    print("   • Inhibition balances excitatory reverberation for stable competition")
    
    print("\n2. BACKGROUND DRIVE EFFECTS (I0):")
    print("   • Low drive → Weak baseline activity, poor sensitivity")
    print("   • High drive → Strong baseline, faster evidence accumulation")
    print("   • Background drive sets the operating point for decision circuits")
    
    print("\n3. NEURAL NOISE EFFECTS (σ):")
    print("   • Low noise → Deterministic but potentially stuck decisions")
    print("   • High noise → Variable decisions, explores decision space")
    print("   • Noise enables decision-making when evidence is weak")
    
    print("\nKEY PRINCIPLE:")
    print("The Wong-Wang model implements 'slow reverberation' through NMDA-mediated")
    print("recurrent excitation (τ_NMDA ~100ms), allowing evidence integration over")
    print("hundreds of milliseconds. Global inhibition (w_I) provides the competitive")
    print("dynamics that force winner-take-all decisions. The balance between these")
    print("mechanisms determines the speed-accuracy tradeoff.")
    
    print("\nBIOLOGICAL RELEVANCE:")
    print("These parameters map to real neural mechanisms:")
    print("• w_I ≈ GABAergic interneuron strength in LIP/PFC")
    print("• I0 ≈ Baseline dopamine/acetylcholine modulation")  
    print("• σ ≈ Ion channel noise and synaptic variability")


def main():
    print("Exploring mechanistic effects of Wong-Wang parameters...")
    print("This demonstrates how inhibition, background drive, and noise")
    print("affect the balance between slow reverberation and competition.")
    
    # Use moderate parameters as baseline
    base_net = WongWangNetwork()
    coherences = (0.0, 0.128, 0.256, 0.512)
    
    print("\nTesting inhibition effects...")
    inhibition_results = explore_inhibition(base_net, coherences)
    
    print("\nTesting background drive effects...")
    background_results = explore_background_drive(base_net, coherences)
    
    print("\nTesting noise effects...")
    noise_results = explore_noise(base_net, coherences)
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'mechanistic_analysis.png'
    
    # Generate plots
    plot_mechanistic_effects(inhibition_results, background_results, noise_results, save_path)
    
    # Print interpretations
    print_mechanistic_insights(inhibition_results, background_results, noise_results)


if __name__ == "__main__":
    main()