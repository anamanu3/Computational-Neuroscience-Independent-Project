"""
simulate.py - Run batches of Wong-Wang trials and save results

This script lets you run large-scale experiments with the Wong-Wang network.
You can sweep across different coherences, thresholds, or network parameters
and save everything to CSV for later analysis.

Usage examples:
    python simulate.py                           # basic run with defaults
    python simulate.py --trials 500 --thresh 60 # more trials, higher threshold  
    python simulate.py --coherences "0,0.1,0.2,0.5" --out my_results.csv
    python simulate.py --param_sweep             # sweep multiple thresholds
"""

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from network import WongWangNetwork, sweep_psychometric


def save_metadata(network, args, output_file, runtime_seconds):
    """Save experiment metadata for reproducibility."""
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": runtime_seconds,
        "parameters": {
            "n_trials": args.trials,
            "threshold": args.thresh,
            "dt": args.dt,
            "coherences": [float(x.strip()) for x in args.coherences.split(",")],
            "seed": getattr(args, 'seed', None)
        },
        "network_parameters": {
            "N_E1": network.N_E1,
            "N_E2": network.N_E2, 
            "N_I": network.N_I,
            "w_plus": network.w_plus,
            "w_minus": network.w_minus,
            "w_I": network.w_I,
            "tau_s": network.tau_s,
            "I0": network.I0,
            "JA_ext": network.JA_ext,
            "mu0": network.mu0,
            "a": network.a,
            "b": network.b,
            "d": network.d,
            "sigma": network.sigma
        }
    }
    
    # save metadata alongside CSV
    metadata_path = Path(output_file).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")


def run_basic_experiment(coherences, n_trials, thresh, dt, output_file, seed=None):
    """Run a single psychometric experiment and save to CSV."""
    print(f"Running experiment with {n_trials} trials per coherence...")
    print(f"Coherences: {coherences}")
    print(f"Threshold: {thresh} Hz, dt: {dt} ms")
    if seed is not None:
        print(f"Seed: {seed}")
    
    net = WongWangNetwork()
    start_time = time.time()
    
    results = sweep_psychometric(net, coherences=tuple(coherences), 
                               n_trials=n_trials, dt=dt, thresh=thresh, seed=seed)
    
    elapsed = time.time() - start_time
    print(f"Simulation took {elapsed:.1f} seconds")
    
    # save to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["coherence", "p_correct", "mean_rt_ms", "decision_rate", "rt_std"])
        for row in results:
            writer.writerow(row.tolist())
    
    print(f"Results saved to {output_path}")
    return results, net, elapsed


def run_parameter_sweep(coherences, n_trials, dt, output_file):
    """Run experiments across multiple thresholds to see how it affects behavior."""
    print("Running parameter sweep across thresholds...")
    
    thresholds = [20, 30, 40, 50, 60, 70, 80]
    all_results = []
    
    for thresh in thresholds:
        print(f"  Running threshold = {thresh} Hz...")
        net = WongWangNetwork()
        results = sweep_psychometric(net, coherences=tuple(coherences),
                                   n_trials=n_trials, dt=dt, thresh=thresh)
        
        # add threshold column to results
        for row in results:
            new_row = [thresh] + row.tolist()
            all_results.append(new_row)
    
    # save combined results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "coherence", "p_correct", "mean_rt_ms", "decision_rate", "rt_std"])
        for row in all_results:
            writer.writerow(row)
    
    print(f"Parameter sweep results saved to {output_path}")
    return all_results


def run_noise_sweep(coherences, n_trials, thresh, dt, output_file):
    """See how different noise levels affect the network."""
    print("Running noise level sweep...")
    
    noise_levels = [0.005, 0.01, 0.02, 0.04, 0.08]
    all_results = []
    
    for sigma in noise_levels:
        print(f"  Running sigma = {sigma}...")
        net = WongWangNetwork(sigma=sigma)
        results = sweep_psychometric(net, coherences=tuple(coherences),
                                   n_trials=n_trials, dt=dt, thresh=thresh)
        
        # add noise level to results
        for row in results:
            new_row = [sigma] + row.tolist()
            all_results.append(new_row)
    
    # save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "coherence", "p_correct", "mean_rt_ms", "decision_rate", "rt_std"])
        for row in all_results:
            writer.writerow(row)
    
    print(f"Noise sweep results saved to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Wong-Wang network experiments")
    
    # basic experiment parameters
    parser.add_argument("--trials", type=int, default=200, 
                       help="trials per coherence (default: 200)")
    parser.add_argument("--thresh", type=float, default=60.0, 
                       help="decision threshold in Hz (default: 60)")
    parser.add_argument("--dt", type=float, default=0.5, 
                       help="simulation time step in ms (default: 0.5)")
    parser.add_argument("--coherences", type=str, default="0,0.064,0.128,0.256,0.512",
                       help="comma-separated coherence values")
    parser.add_argument("--out", type=str, default="data/results.csv",
                       help="output CSV file path")
    parser.add_argument("--seed", type=int, default=None,
                       help="random seed for reproducibility")
    
    # experiment types
    parser.add_argument("--param_sweep", action="store_true",
                       help="sweep across multiple thresholds")
    parser.add_argument("--noise_sweep", action="store_true", 
                       help="sweep across noise levels")
    
    args = parser.parse_args()
    
    # parse coherences
    coherences = [float(x.strip()) for x in args.coherences.split(",")]
    
    # run the appropriate experiment
    if args.param_sweep:
        run_parameter_sweep(coherences, args.trials, args.dt, args.out)
    elif args.noise_sweep:
        run_noise_sweep(coherences, args.trials, args.thresh, args.dt, args.out)
    else:
        results, network, elapsed = run_basic_experiment(coherences, args.trials, args.thresh, 
                                                        args.dt, args.out, args.seed)
        
        # save metadata for reproducibility
        save_metadata(network, args, args.out, elapsed)
        
        # print a quick summary
        print("\nQuick summary:")
        print("Coherence | Accuracy | Mean RT")
        print("-" * 30)
        for row in results:
            coh, acc, rt = row[0], row[1], row[2]
            print(f"{coh:8.3f} | {acc:7.2%} | {rt:6.0f}ms")


if __name__ == "__main__":
    main()