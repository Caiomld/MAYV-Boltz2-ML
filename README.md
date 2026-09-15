# Hybrid Boltz-2 + Machine Learning for Large-Scale Virtual Screening

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML%20Pipeline-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Model-ChemBERTa-orange.svg)](https://huggingface.co/)
[![Boltz-2](https://img.shields.io/badge/Structure%20Model-Boltz--2-purple.svg)](https://github.com/jwohlwend/boltz)
[![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-green.svg)](https://www.rdkit.org/)

</p>

<p align="center">
<i>
From hundreds of reference compounds to millions of molecules:
a hybrid Boltz-2 + Machine Learning pipeline for scalable virtual screening.
</i>
</p>

---

## Overview

Virtual screening offers a powerful strategy for reducing the chemical search space in early-stage drug discovery. However, increasing library size introduces a fundamental trade-off between **computational scalability** and **prediction fidelity**.

This project investigates a hybrid strategy in which **Boltz-2** is used as the high-fidelity reference model, while a lightweight **Machine Learning surrogate model** is trained to reproduce its affinity-related predictions directly from molecular SMILES.

The central idea is simple:

> **Use an inexpensive ML model to screen millions of molecules, and reserve Boltz-2 for the most promising candidates.**

The project is applied to the **capsid protein C of Mayaro virus (MAYV)**, with the objective of identifying candidate molecules capable of interfering with the protein–protein interaction involving protein M.

---

# The Pipeline

The project is organized into two complementary approaches.

### Approach I — Model Development

A reference dataset of approximately **320 compounds** from the Essential Fragment Library is used to investigate how molecular representations affect the ability of ML models to reproduce Boltz-2 predictions.

Several strategies are compared under the same data split:

* RDKit molecular descriptors
* pretrained ChemBERTa embeddings
* fine-tuned ChemBERTa
* a baseline based on features derived from Boltz-2

The best-performing representation is subsequently optimized with **Optuna** and used to define the surrogate model for large-scale screening.

### Approach II — Large-Scale Screening

The selected representation/model is transferred to a larger reference dataset of approximately **8,960 compounds**.

After training and evaluation, the surrogate model is frozen and used to process a much larger chemical library.

The screening strategy is:

```text
Millions of compounds
        │
        ▼
ChemBERTa + ML surrogate
        │
        ▼
Ranked candidates
        │
        ▼
Top 10,000
        │
        ▼
Candidate refinement / diversity
        │
        ▼
Top 1,000
        │
        ▼
Boltz-2
        │
        ▼
Structural validation
```

The objective is therefore not to replace Boltz-2, but to **reduce the number of molecules that require expensive structural prediction**.

---

# Biological Target

## Mayaro Virus Capsid Protein C

The molecular target considered in this project is the **Mayaro virus capsid protein C**, using the structural system associated with **PDB 7KO8**.

The computational objective is to identify compounds that may interfere with the interaction between capsid protein C and protein M.

The target system and the motivation for targeting the Mayaro virus are described in the associated research report.

---

# Why Boltz-2 + Machine Learning?

Large-scale molecular screening creates a practical computational bottleneck.

A structural model such as Boltz-2 provides information about protein–ligand interactions, but applying a computationally expensive model to an entire multi-million-compound library is impractical in many workflows.

This project therefore separates the problem into two stages:

### High-throughput stage

**SMILES → ChemBERTa → ML surrogate → affinity-related score**

### High-fidelity stage

**Protein + ligand → Boltz-2 → structural/affinity validation**

This creates a hierarchical screening strategy in which computational resources are concentrated on increasingly promising candidates.

---

# Machine Learning Strategy

## Molecular representation

Molecules are represented from their SMILES strings using **ChemBERTa**, a transformer-based model trained on chemical language representations.

The implementation uses:

```text
DeepChem/ChemBERTa-77M-MTR
```

with a maximum SMILES sequence length of 128 tokens.

The resulting molecular representation is used as input to a fully connected neural network.

---

## Prediction target

The primary target in the large-scale screening workflow is:

```text
affinity_probability_binary
```

A secondary Boltz-2 affinity output is also investigated:

```text
affinity_pred_value
```

The probability-like output is emphasized for screening because the primary goal of the surrogate model is **candidate prioritization** rather than exact reproduction of an absolute binding-affinity scale.

---

# Approach I — Model Development

## Reference dataset

The first stage uses approximately **320 molecules** from the Essential Fragment Library.

The Boltz-2 results are associated with the corresponding molecular SMILES and converted into a machine-learning dataset.

### Compared representations

```text
                ~320 compounds
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
        RDKit       ChemBERTa    Fine-tuned
      descriptors   embeddings    ChemBERTa
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                     ML model
                       │
                       ▼
            performance comparison
```

A Boltz-2-derived feature baseline is also included as a positive-control strategy.

---

## Model selection

All candidate strategies are evaluated using the same train/validation/test partition.

The evaluation includes:

* MAE
* RMSE
* R²
* Pearson correlation
* Spearman correlation

The winning representation is then optimized using **Optuna**, allowing the neural-network architecture and training hyperparameters to be selected systematically.

---

# Approach II — Scalability

After identifying the most suitable molecular representation, the workflow moves to a larger reference dataset of approximately **8,960 compounds**.

The resulting model acts as a **target-specific surrogate for Boltz-2**.

The final model is then frozen before application to previously unseen molecules.

---

## Large-scale library

The current large-scale experiment uses Enamine Hit Locator Library.

containing approximately:

**460 thousand compounds**

The library is processed incrementally rather than loading the entire dataset into memory.

Conceptually:

```text
460.160 compounds
        │
        ├── chunk
        ├── chunk
        ├── chunk
        ├── ...
        └── chunk
             │
             ▼
        ChemBERTa
             │
             ▼
       ML surrogate
             │
             ▼
      predicted score
             │
             ▼
       ranked library
```

This allows large libraries to be screened while maintaining controlled memory usage.

---

# Scalability Benchmark

A major objective of Approach II is not only predictive performance, but **computational efficiency**.

The pipeline measures:

### Machine Learning

* ChemBERTa inference time
* neural-network inference time
* total pipeline time
* molecules per second

### Boltz-2

* measured runtime on benchmark subsets
* throughput
* comparative runtime

### Derived metrics

The project reports:

```text
Speedup = Boltz-2 runtime / ML runtime
```

and evaluates how the computational cost scales with library size.

The benchmark is designed to distinguish between **measured Boltz-2 runtimes** and extrapolated estimates.

---

## The key scientific question

The purpose of the ML model is not simply to obtain a high R².

For virtual screening, the more important question is:

> **Can the model retain the compounds that Boltz-2 would rank highly?**

Therefore, large-scale evaluation focuses on ranking-oriented quantities such as:

* Spearman correlation
* rank preservation
* Recall@K
* enrichment
* top-K overlap
* computational speedup

---

# Screening Funnel

The final screening strategy is designed around progressive reduction of the chemical search space:

```text
                    LARGE LIBRARY
                         │
                         ▼
                ML PRE-SCREENING
                         │
                         ▼
                    TOP 10,000
                         │
                         ▼
              Candidate refinement
                         │
                         ▼
                     TOP 1,000
                         │
                         ▼
                      BOLTZ-2
                         │
                         ▼
               FINAL CANDIDATES
```

The Approach II notebook explicitly produces Top-10,000 and Top-1,000 candidate tables for subsequent analysis.

---

# Repository Structure

```text
.
├── ML_boltz2_PARTE1.ipynb
├── ML_boltz2_PARTE2.ipynb
│
│
├── results/
│   ├── approach_I/
│   ├── approach_II/
│
├── README.md
```
---

# Reproducibility

## Environment

The project uses Python together with:

* PyTorch
* Hugging Face Transformers
* ChemBERTa
* RDKit
* scikit-learn
* SciPy
* Optuna
* pandas
* NumPy
* Matplotlib

Boltz-2 inference is performed separately in the computational environment used for the structural calculations.

---

## Hardware

Boltz-2 calculations in the original study were performed using NVIDIA A100 40 GB GPU resources on LNBio (CNPEM) infrastructure.

The Machine Learning screening stage can be executed independently from the Boltz-2 environment, allowing the inexpensive screening layer to be run on local hardware or accelerated computing resources depending on library size.

---

# Data Availability

The repository does **not** redistribute proprietary or restricted commercial compound libraries.

Users interested in reproducing the complete screening experiments should obtain the corresponding Enamine libraries directly from the provider.

The repository instead provides:

* data-processing scripts;
* expected input formats;
* selected derived tables;
* model artifacts where appropriate;
* benchmark results;
* analysis code;
* reproducibility instructions.

---

# Limitations

This workflow should be interpreted as a **target-specific surrogate modeling strategy**, not as a general-purpose replacement for structural protein–ligand prediction.

The ML model learns the relationship between molecular representations and Boltz-2 outputs for the specific target system used in this study.

Important limitations include:

* dependence on the reference Boltz-2 dataset;
* possible distribution shift between reference and screening libraries;
* uncertainty in extrapolation to chemically novel molecules;
* surrogate-model approximation error;
* ranking errors among chemically similar candidates;
* the fact that ML scores are not experimental binding measurements.

Consequently, ML predictions are intended for **prioritization**, while high-ranking candidates should undergo subsequent structural and experimental validation.

# References

Key references include:

1. Passaro, S. et al. **Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction.** 2025.
2. Chithrananda, S., Grand, G. & Ramsundar, B. **ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction.** 2020.
3. Wohlwend, J. & Boltz Team. **Boltz: Official repository for the Boltz biomolecular interaction models.**
4. RCSB Protein Data Bank. **PDB 7KO8 — Mayaro virus.**
5. Enamine Ltd. **Essential Fragment Library.**
6. Enamine Ltd. **Protein Mimetics Library.**

A complete reference list is maintained in the project documentation.

---

# Authors

**Caio Matheus Leão Dantas**
Ilum – Escola de Ciência
Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)

**Amauri Donadon Leal Junior**
Laboratório Nacional de Biociências — LNBio / CNPEM

**Leandro Oliveira Bortot**
Laboratório Nacional de Biociências — LNBio / CNPEM

</i>
</p>
