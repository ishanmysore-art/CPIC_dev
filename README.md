# Compressed Predictive Information Coding 
This repo is used to publish the code for Compressed Predictive Information Coding (CPIC) methodology.

## Preinstall Pacakages
CPIC requests the pre-installation of DCA package for initialization. Please refer to https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/index.html.

## Code

### Main Code
1. Main code:
   1. CPIC code: <code>src/cpic/CPIC.py</code>
   2. utility code: <code>src/cpic/utils.py/</code>

### Synthetic Experiments

1. Lorenz experiment code: <code>synthetic/synthetic_*.py</code>
   1. Synthetic data generation using <code>synthetic_generator.py</code>.
   2. CPIC model on synthetic data using <code>synthetic_experiment.py</code>.
   3. Other models on synthetic data using <code>synthetic_competitors.py</code>.
   4. Per/Post analysis includes <code>synthetic_plot.py</code>, <code>synthetic_summarization.py</code>, <code>synthetic_summary.py</code>, <code>synthetic_visualization.py</code>.
2. Synthetic experiments to understand the Prediction information in CPIC setting:
   1. Synthetic data generation with <code>synthetic/data_generator.py</code>.
   2. PI analysis using <code>synthetic/PI_analysis.py</code>.


### Real Data Experiments
1. Real data experiments code: <code>analysis/real_data_*.py</code>
   1. Real data experiments with CPIC for four datasets including M1, HC, Temp, MS using <code>real_data_experiment_standard.py</code>, <code>real_data_experiment_standard_beta.py</code>. Note that beta refers to varying weight option.
   2. Real data experiments with other models using <code>real_data_competitors.py</code>.
   3. Real data experiments post analysis: <code>real_data_summary_standard.py</code>, <code>real_data_summary_standard_beta.py</code>.

## Encoder types
CPIC supports multiple encoder architectures via <code>encoder_params["encoder_type"]</code>. All encoders map input (T x D) to output (T x M). Example configurations:

- **Linear**: <code>{"encoder_type": "linear", "deterministic": False}</code>
- **MLP**: <code>{"encoder_type": "mlp", "n_layers": 1, "activation": "relu", "deterministic": False}</code>
- **2x MLP**: <code>{"encoder_type": "mlp2", "deterministic": False}</code>
- **Conv (spatial only)**: <code>{"encoder_type": "conv_spatial"}</code> or <code>"conv"</code>, with optional <code>n_layers</code>, <code>conv_kernel_size</code>, <code>conv_stride</code>, <code>conv_padding</code>
- **Conv (spatiotemporal)**: <code>{"encoder_type": "conv_spatiotemporal", "conv_kernel_size": 3, "conv_stride": 1}</code>
- **1D temporal Conv**: <code>{"encoder_type": "conv1d_temporal", "kernel_size_1d": 3, "n_layers": 1}</code>
- **Attention**: <code>{"encoder_type": "attention", "num_heads": 4, "num_layers": 2, "dropout": 0.1}</code>

Pass <code>encoder_params</code> when constructing CPIC and when calling <code>fit()</code> the encoder is built from the registry.

## Configuration
Synthetic experiment configurations for CPIC are available in <code>synthetic/config/\*</code>. Real data experiment configurations for CPIC are available in <code>analysis/config/\*</code>.

## Figures
Figures for synthetic experiments in the paper are available in <code>synthetic/fig/\*</code>. Figures for real data experiments in the paper are available in <code>analysis/fig/\*</code>.