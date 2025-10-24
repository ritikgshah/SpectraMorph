# SpectraMorph: Structured Latent Learning for Self-Supervised Hyperspectral Super-Resolution

**Authors:** Ritik Shah ([rgshah@umass.edu](mailto:rgshah@umass.edu)), Marco Duarte ([mduarte@ecs.umass.edu](mailto:mduarte@ecs.umass.edu))

📄 [Extended version on arXiv](https://arxiv.org/pdf/2510.20814v1)

---

## 🔍 Overview

Welcome to the official implementation of **SpectraMorph**. This repository accompanies our research paper and includes code, notebooks, datasets, precomputed results, and environment files for experimentation and analysis.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)
- [Datasets](#-datasets)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)

---

## 📁 Repository Structure

### 🔸 CSV Files with Metrics

- Organized into subfolders by synthetic HSI dataset:
  - *Botswana*
  - *Kennedy Space Center*
  - *Pavia Center*
  - *Pavia University*
  - *Washington DC Mall*
- Each subfolder contains:
  - CSVs with image quality metrics across various experimental configurations (e.g., different PSFs, downsampling ratios, and MSI band counts).
  - An Excel summary: `All SOTA methods comparison.xlsx` aggregates all individual method metrics and comparison tables

---

### 🔸 Comparison Implementations

Benchmarks against state-of-the-art supervised and unsupervised baselines.

- **Supervised**  
  - Pre-executed Jupyter notebooks for: `FeINFN`, `FusFormer`, `GuidedNet`, `MIMO-SST`
  - Applied to both synthetic datasets and the University of Houston dataset.

- **Unsupervised**  
  - Pre-executed Jupyter notebooks for: `MIAE`, `C2FF`, `SSSR`, `SDP`, `SpectraLift`
  - Applied to both synthetic datasets and the University of Houston dataset.

---

### 🔸 SpectraMorph Implementation
> Note: The Latent Estimation Network (LEN) mentioned in the paper is the MLP part of MSItoHSI_MLP model class in the code on this repository. 

- **SpectraMorph Implementation** (`SpectraMorph_Implementation_Jupyter_Notebooks`):  
  Pre-executed notebooks demonstrating SpectraMorph on both synthetic and University of Houston datasets.

- **Ablation Study** (`SpectraMorph_Ablation_Study_Jupyter_Notebooks`):  
  Pre-executed notebooks evaluating contributions of different model components and architectural choices.

- **Python Scripts** (`SpectraMorph_python`):  
  Full Python-based implementation for use in IDEs or via command line.  
  Detailed instructions available in [Python Implementation Details](py_implementation.md).

- **Visualization Notebooks** (`Super_Resolved_Images_Comparison_Jupyter_Notebooks`):  
  Pre-executed notebooks for visual comparison of high-resolution HSI outputs of all methods on each dataset. Includes:
  - Spectral plots
  - Image visualizations
  > Note: Botswana output is split across two notebooks due to data size.

---

## ⚙️ Environment Setup

Use either `pip` or `conda` to recreate the project environment:
> Note: This environment contains all the packages necessary to execute all the jupyter notebooks and python files in this repository, including the comparison methods.

### Using pip:
```bash
pip install -r requirements.txt
```
### Using Conda:
```bash
conda env create -f spectramorph-env.yaml
conda activate spectramorph-env
```
In rare cases, the Conda YAML file may fail to recreate the environment correctly. If so, use requirements.txt. In the case where both the yaml as well as requirements.txt fail to create the environment using the commands given above, please follow the instructions in the [Manual Environment Setup Guide](ENVIRONMENT_SETUP.md) to be able to create a conda environment with every dependency you need to execute any file within this repository.

## 🗂️ Datasets

The repository includes `.mat` files for all synthetic hyperspectral image (HSI) datasets used in our experiments:

- Washington DC Mall  
- Kennedy Space Center  
- Pavia University  
- Pavia Center  
- Botswana  

### 📥 University of Houston Dataset

The full University of Houston dataset can be downloaded from the official source:

🔗 [2018 IEEE GRSS Data Fusion Challenge](https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/)

---

## ✅ Reproducibility

To ensure full transparency and ease of experimentation:

- All notebooks are **pre-executed**, so results can be viewed immediately without re-running code.
- Environment files are provided to replicate the exact Python environment used in our experiments.
- Python scripts are included along with detailed instructions to ensure that researchers can use our method with ease.

---

## 🛡️ Contribution Guidelines
This is a public, read-only repository. If you'd like to suggest changes:
- Please **fork** the repo
- Make your edits
- Submit a **pull request** for review

Direct pushes are disabled. Thanks for respecting this workflow!

## 📖 Citation

If you use this repository or build upon our work, please cite the following paper:

```bibtex
@misc{shah2025spectramorphstructuredlatentlearning,
      title={SpectraMorph: Structured Latent Learning for Self-Supervised Hyperspectral Super-Resolution}, 
      author={Ritik Shah and Marco F Duarte},
      year={2025},
      eprint={2510.20814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.20814}, 
}
```

## 📜 License

This repository is licensed under the **Apache-2.0 License**.

You are free to use, modify, and distribute this code with proper attribution.

For full details, see the [LICENSE](./LICENSE) file in the repository.





