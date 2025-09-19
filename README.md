# 🧠 Wong-Wang Decision Network: A Computational Neuroscience Implementation

<div align="center">

![Network Dynamics](https://img.shields.io/badge/Model-Wong%20&%20Wang%202006-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

*A biophysically realistic neural network model of perceptual decision-making*

**[Quick Start](#-quick-start-60-seconds)** • **[Documentation](#-file-documentation)** • **[Results](#-example-results)** • **[Science](#-scientific-background)**

</div>

---

## 🎯 Overview

This project implements the seminal **Wong & Wang (2006)** neural network model of decision-making in the brain. The model simulates how competing populations of neurons in areas like the lateral intraparietal cortex integrate sensory evidence over time to make perceptual decisions.

### What Makes This Special

- **Biophysically Realistic**: Uses proper NMDA/GABA time constants and transfer functions
- **Empirically Validated**: Compared against classic Roitman & Shadlen (2002) monkey data  
- **Mechanistically Interpretable**: Shows how inhibition balances slow reverberation
- **Reproducible**: Full metadata logging and parameter tracking
- **Publication Ready**: Analysis pipeline suitable for computational neuroscience research

---

## ⚡ Quick Start (60 Seconds)

```bash
# 1. Clone and install
git clone https://github.com/anamanu/Computational-Neuroscience-Independent-Project.git
cd Computational-Neuroscience-Independent-Project
pip install -r requirements.txt

# 2. Run simulation with full metadata logging
python simulate.py --trials 200 --seed 42

# 3. Generate analysis plots
python analysis.py

# 4. Explore mechanistic effects
python mechanistic_analysis.py

# 5. Compare to drift diffusion model
python simple_ddm_comparison.py

# 6. Validate against empirical data
python empirical_benchmark.py

# 7. Run tests
python test.py
```

---

## 📁 File Documentation

### Core Implementation

<table>
<tr>
<td><strong>📄 network.py</strong></td>
<td>
<strong>The heart of the model</strong><br>
• <code>WongWangNetwork</code> class with biologically realistic parameters<br>
• Proper Wong-Wang transfer function: <code>r = (aI - b) / (1 - d(aI - b))</code><br>
• Single trial simulation with NMDA-mediated slow dynamics<br>
• Psychometric curve generation across coherence levels<br>
• Noise scaling that respects integration timestep
</td>
</tr>
</table>

### Experimental Pipeline

<table>
<tr>
<td><strong>⚙️ simulate.py</strong></td>
<td>
<strong>High-throughput experiment runner</strong><br>
• Command-line interface for parameter sweeps<br>
• Automatic metadata logging (parameters, seeds, runtime)<br>
• Multiple experiment types: basic, threshold sweeps, noise sweeps<br>
• CSV output with JSON metadata for full reproducibility<br>
• Progress tracking and performance monitoring
</td>
</tr>
</table>

<table>
<tr>
<td><strong>📊 analysis.py</strong></td>
<td>
<strong>Publication-quality visualization</strong><br>
• Four-panel analysis: psychometric, chronometric, decision rate, speed-accuracy<br>
• Error bars and statistical summaries<br>
• Parameter sweep visualization with overlaid conditions<br>
• Automatic data loading and format detection<br>
• High-resolution figure export
</td>
</tr>
</table>

### Scientific Validation

<table>
<tr>
<td><strong>🔬 empirical_benchmark.py</strong></td>
<td>
<strong>Compare against real neural data</strong><br>
• Overlay with Roitman & Shadlen (2002) monkey LIP recordings<br>
• Quantitative fit assessment (correlations, RMSE)<br>
• Biological realism scoring<br>
• Reference to foundational decision-making literature<br>
• Statistical validation of model correspondence
</td>
</tr>
</table>

<table>
<tr>
<td><strong>⚖️ simple_ddm_comparison.py</strong></td>
<td>
<strong>Mathematical model equivalence</strong><br>
• Compare neural network to drift diffusion model<br>
• Demonstrate that biophysical detail → behavioral simplicity<br>
• Side-by-side psychometric and chronometric comparisons<br>
• Correlation analysis between model frameworks<br>
• Proof that complex mechanisms can exhibit simple dynamics
</td>
</tr>
</table>

<table>
<tr>
<td><strong>🔧 mechanistic_analysis.py</strong></td>
<td>
<strong>Parameter space exploration</strong><br>
• Three-way analysis: inhibition (w_I), background drive (I0), noise (σ)<br>
• Six-panel visualization showing mechanistic effects<br>
• "Slow reverberation vs inhibition" principle demonstration<br>
• Biological interpretation of parameter effects<br>
• Links computational parameters to neural mechanisms
</td>
</tr>
</table>

### Quality Assurance

<table>
<tr>
<td><strong>🧪 test.py</strong></td>
<td>
<strong>Comprehensive model validation</strong><br>
• Six test categories: mechanics, coherence effects, curve shape<br>
• Decision rate validation and parameter effect testing<br>
• Reaction time bounds checking<br>
• Automated pass/fail reporting with diagnostics<br>
• Ensures model behaves like real decision-making system
</td>
</tr>
</table>

<table>
<tr>
<td><strong>🐞 debug_data.py</strong></td>
<td>
<strong>Development diagnostic tool</strong><br>
• Quick data inspection and validation<br>
• Parameter range checking<br>
• Decision rate diagnostics<br>
• Troubleshooting helper for model tuning
</td>
</tr>
</table>

### Configuration

<table>
<tr>
<td><strong>📋 requirements.txt</strong></td>
<td>
<strong>Pinned dependencies</strong><br>
• <code>numpy==1.24.3</code> - Numerical computation<br>
• <code>matplotlib==3.7.1</code> - Plotting and visualization<br>
• <code>scipy==1.10.1</code> - Optimization and statistics
</td>
</tr>
</table>

---

## 🎨 Example Results

### Behavioral Curves
The model produces classic decision-making patterns:
- **Psychometric functions**: Accuracy increases with motion coherence
- **Chronometric functions**: Reaction times show inverted-U with coherence  
- **Speed-accuracy tradeoffs**: Higher thresholds → slower, more accurate decisions

### Mechanistic Insights
Parameter manipulations reveal neural mechanisms:
- **Inhibition (w_I)**: Controls competition strength and decision speed
- **Background drive (I0)**: Sets operating point and sensitivity
- **Neural noise (σ)**: Enables decisions when evidence is weak

### Model Validation
Multiple validation approaches confirm biological realism:
- **Empirical correspondence**: Matches Roitman & Shadlen (2002) monkey data
- **DDM equivalence**: Shows drift-diffusion-like behavioral dynamics
- **Parameter sensitivity**: Realistic responses to neural manipulations

---

## 🔬 Scientific Background

### The Wong-Wang Model

The Wong & Wang (2006) model is a foundational framework in computational neuroscience that bridges neural mechanisms and behavioral decisions. Key innovations:

1. **Slow Reverberation**: NMDA-mediated recurrent excitation (τ ≈ 100ms) allows evidence integration over hundreds of milliseconds

2. **Competitive Dynamics**: Global inhibition creates winner-take-all competition between decision alternatives

3. **Biophysical Realism**: Parameters map directly to neural mechanisms (ion channels, neurotransmitter kinetics, connectivity)

### Biological Mapping

The model's abstract components correspond to real neural circuits:

| Model Component | Biological Substrate |
|-----------------|---------------------|
| E1, E2 populations | LIP neurons selective for left/right motion |
| Global inhibition | GABAergic interneuron networks |
| NMDA dynamics | Slow glutamate receptor kinetics |
| Background drive | Dopaminergic/cholinergic modulation |
| Neural noise | Ion channel stochasticity |

### Research Applications

This implementation enables investigation of:
- **Decision-making disorders** (ADHD, schizophrenia, Parkinson's)
- **Cognitive aging** effects on neural competition
- **Pharmacological interventions** targeting specific mechanisms
- **Individual differences** in decision-making style
- **Learning and adaptation** in uncertain environments

---

## 📈 Advanced Usage

### Parameter Sweeps
```bash
# Explore threshold effects
python simulate.py --param_sweep --trials 500

# Test noise sensitivity  
python simulate.py --noise_sweep --trials 300

# Custom coherence range
python simulate.py --coherences "0,0.1,0.2,0.5" --thresh 15
```

### Analysis Options
```bash
# Parameter sweep analysis
python analysis.py --type sweep --input sweep_results.csv

# High-resolution figures
python analysis.py --savefigs --output_dir publication_figures

# Custom input file
python analysis.py --input my_experiment.csv
```

### Reproducibility
Every simulation generates paired files:
- `results.csv` - Behavioral data
- `results.json` - Complete metadata (parameters, seeds, runtime)

This ensures every result can be exactly reproduced.

---

## 📚 References

### Primary Literature
- **Wong, K. F., & Wang, X. J.** (2006). A recurrent network mechanism of time integration in perceptual decisions. *Journal of Neuroscience*, 26(4), 1314-1328.

### Empirical Foundation  
- **Roitman, J. D., & Shadlen, M. N.** (2002). Response of neurons in the lateral intraparietal area during a combined visual discrimination reaction time task. *Journal of Neuroscience*, 22(21), 9475-9489.

### Theoretical Framework
- **Ratcliff, R., & McKoon, G.** (2008). The diffusion decision model: theory and data for two-choice decision tasks. *Neural Computation*, 20(4), 873-922.

---

## 🤝 Contributing

This project demonstrates publication-ready computational neuroscience methodology. Areas for extension:

- **Learning mechanisms**: Synaptic plasticity and adaptation
- **Multi-alternative decisions**: Beyond two-choice tasks  
- **Hierarchical processing**: Multiple cortical areas
- **Clinical applications**: Disease modeling and intervention testing

---

## 📜 License

MIT License - See LICENSE file for details.

---

<div align="center">

**Built with computational neuroscience rigor**

*Bridging neural mechanisms and behavioral dynamics*

</div>
