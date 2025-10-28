# Python Implementation Details for SpectraMorph

This document provides a comprehensive guide for using the Python scripts included in the SpectraMorph framework. Whether you're working in an IDE or running experiments from the command line, this guide will help you navigate the available options and choose the best configuration for your use case.

---

## Overview of Python Scripts

| Script / Module       | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `run.py`              | Interactive script for running SpectraMorph within an IDE (e.g., VS Code, PyCharm). Ideal for debugging, experimentation, or educational purposes. |
| `main.py`             | Command-line interface for reproducible experiments using a config file. Recommended for batch runs, scripting, and performance tracking. |
| `spectramorph_helpers.py` | Core training loop, inference routines, and MLP model architecture. |
| `generate_synthetic_inputs.py` | Creates synthetic low-resolution HSI and high-resolution MSI inputs using custom or predefined PSFs and SRFs. |
| `utils.py`            | Utility functions for spectral/spatial degradation, PSF/SRF models, normalization, and noise injection. |
| `compute_metrics.py`  | Computes evaluation metrics for quantitative benchmarking. |
| `__init__.py`         | Makes SpectraMorph usable as a Python package by exposing key functions for import. |

---

## Using `run.py` (for IDE users)

If you're working in an IDE like VS Code or PyCharm, use `run.py` for a hands-on, interactive workflow.

### User-Defined Parameters in `run.py`

| Parameter | Description |
|----------|-------------|
| `synthetic` | If `True`, SpectraMorph generates synthetic HR MSI and LR HSI from a raw HSI input. Set to `False` to use real-world data. |
| `provide_psf` | Set to `True` to load a custom PSF from a `.mat` file (`psf_file`). |
| `provide_srf` | Set to `True` to load a custom SRF from a `.mat` file (`srf_file`). Required if using real-world MSI data with arbitrary band counts. |

---

### Input Paths

| Parameter | Description |
|----------|-------------|
| `mat_file` | Path to your raw hyperspectral `.mat` file (only used if `synthetic=True`). |
| `mat_key` | Name of the variable inside the `.mat` file. This is case-sensitive. |
| `psf_file` / `srf_file` | Paths to custom PSF and SRF files, if `provide_psf` or `provide_srf` is `True`. |
| `hr_msi_file`, `lr_hsi_file` | For real-world mode: paths to HR MSI and LR HSI data (as `.mat` or `.tif`). |

---

### Synthetic Generation Parameters

| Parameter | Description |
|----------|-------------|
| `psf_type` | Type of PSF to use. Options: `'gaussian'`, `'kolmogorov'`, `'airy'`, `'moffat'`, `'sinc'`, `'lorentzian2'`, `'hermite'`, `'parabolic'`, `'gabor'`, `'delta'`. |
| `sigma` | Spread of the PSF kernel. Controls the blurring severity. |
| `kernel_size` | Kernel will be of size `(2 × kernel_size + 1)`. |
| `downsample_ratio` | Factor by which spatial resolution is reduced. |
| `snr_spatial`, `snr_spectral` | Signal-to-noise ratios (in dB) for LR HSI and HR MSI respectively. |
| `num_msi_bands` | Number of spectral bands in HR MSI. Must be one of `[1, 3, 4, 8, 16]` unless a custom SRF is provided. |
| `fwhm_factor` | Used to approximate Gaussian SRF from spectral edge bounds. |
| `seed` | Random seed for reproducibility. |

---

### Real-World Input Parameters

| Parameter | Description |
|----------|-------------|
| `num_bands_msi_non_synthetic` | If not using a custom SRF, this tells the framework how many bands are in the HR MSI. Must be one of `[1, 3, 4, 8, 16]`. |

---

### Model Training & Inference Parameters

| Parameter | Description |
|----------|-------------|
| `init_lr`, `max_lr`, `final_lr` | Initial, maximum, and final learning rate parameters for the One Cycle LR scheduler. |
| `epochs` | Total number of training epochs. |
| `hidden_size` | Number of neurons in each hidden layer of the Latent Estimation Network. |
| `training_batch_size` | Number of pixels per batch during training. If `None`, trains on the full image. |
| `inference_batch_size` | Number of pixels per batch during inference. If `None`, runs full-image inference. |
| `num_endmembers` | Number of endmembers to be extracted from the LR HSI. |
| `spec_prior` | Boolean flag indicating use of the Coarse Spectral Prior. If `None`, CSP is not used, else if `True`, CSP is used. |
| `prior_downsample` | The downsampling ratio for generating the downsampled LR HSI for training with the CSP. |
| `verbose` | If set to 1, prints shapes of inputs and outputs. Set to 0 for silent mode. |
| `figure_format` | The file format to save the visualizations of the endmember signatures and Abundance Like Latent Estimates. Options: `jpg` or `png`. |
| `output_file_type` | Format for saving output: `'numpy'`, `'h5'`, or `'matlab'`. Use `'h5'` for large files. |

---

## Using `main.py` (for CLI users)

To run experiments via terminal using a config file:

1. Navigate to the project root.
2. Use a template config from:  
   `SpectraMorph_python/Config_file_templates/`
3. Execute:

```bash
python main.py --config SpectraMorph_python/Config_file_templates/your_config.yaml
```

YAML is recommended over JSON because it supports comments and is easier to read/write. Nonetheless, main.py has support for YAML and JSON config files as well as pure CLI execution.

## Performance & Training Tips

### Inference in Batches = Seamless Results

SpectraMorph uses a **spectral-only MLP** (MSItoHSI_MLP) that treats each pixel independently and does **not rely on spatial context**.  
As a result:

- You can perform inference in **mini-batches** without needing to split the image into spatial tiles.
- There are **no boundary artifacts, stitching errors, or padding effects**, even when inference is done patch-wise.

> This structure allows efficient processing of large HSIs on memory-constrained devices while maintaining pixel-level fidelity.

---

### Choosing the Right Training Batch Size

Unlike many HSI super-resolution methods that train on large high-resolution images, SpectraMorph trains **directly on the low-resolution HSI domain**. This has two important consequences:

#### 1. Small GPUs Can Still Train Large HSIs

- Because the LR HSI is much smaller in size, even full-image training is feasible on modest GPUs (e.g., 4–6 GB VRAM).
- This makes SpectraMorph suitable for laptop or lightweight server environments.

#### 2. Very Small Batch Sizes Hurt Performance

- When batch size is too small, each batch contains **limited spectral diversity**.
- The spectral MLP learns from only a narrow distribution of spectral signatures at a time, increasing risk of **overfitting** or **failure to generalize**.

> **Recommendation**: Use the **largest training batch size your GPU can support** to maximize spectral variability within each batch. This leads to better generalization and improved reconstruction quality.

---

## Learning Rate Schedulers & Optimization Settings

SpectraMorph is compatible with a variety of learning rate (LR) schedulers and optimizer settings. While most reasonable LR schedules will lead to convergence, **careful tuning of LR parameters is essential to achieve state-of-the-art performance**.

The following learning rate schedulers are supported (although the framework can be used with any LR scheduler of your choice, custom implementation will be required from the user's side):

| Scheduler         | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `"one_cycle"`     | Cyclical schedule with warm-up and decay. Good balance between speed and stability. |
| `"flat"`          | Constant learning rate throughout training. Recommended only for ablation or debugging. |

> **Important:** The optimal learning rate schedule and parameters **vary by dataset and experimental setting** (e.g., PSF type, SNR, band count, etc.).

---

### Reference for Exact Settings

To see the **exact learning rate schedules and hyperparameters** used in each of the experiments reported in our paper, please refer to:

[`SpectraMorph_Implementation_Jupyter_Notebooks`](SpectraMorph_Implementation_Jupyter_Notebooks)

These notebooks contain:

- The specific LR scheduler used per experiment
- Exact LR values
- Epoch counts
- Complete configuration for reproducibility

---

## Intermediate Latent Structure Visualization File Format

The intermediate structures: Endmember Signatures and Abundance Like Latent Estimates (ALLE) are saved for visualization. SpectraMorph supports two output formats for saving these: `jpg` and `png`. You can control the format using the `figure_format` parameter.

---

## Output File Formats

SpectraMorph supports three output formats for saving the final high-resolution hyperspectral image:

| Format     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| `'numpy'`  | Saves as `.npy`. Fast and ideal for NumPy-based pipelines.                  |
| `'matlab'` | Saves as `.mat`. Useful for MATLAB-based post-processing or analysis.       |
| `'h5'`     | Saves as `.h5` (HDF5). **Recommended for large datasets** due to efficient I/O and compression. |

You can control the format using the `output_file_type` parameter (`'numpy'`, `'matlab'`, or `'h5'`, case-sensitive).

---

## Questions?

If you have any questions or run into issues:

- Review the [README](README.md)
- Check our pre executed jupyter notebooks that implement the method
- Open an issue on the GitHub repository
- Refer to our [paper on arXiv](https://arxiv.org/pdf/2510.20814v1)
- Email the authors if neither of the above options work for you
