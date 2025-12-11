# 📦 DUASM Assembly Dataset and Evaluation Scripts

Deep Unsupervised Assembly Supplier Matching (DUASM)

This repository provides the assembly-level embeddings, evaluation scripts, and case-specific test files used in the DUASM study.
The full voxel dataset and all synthetic manufacturing metrics (cost, time, tolerance, quantity) used in the Journal of Manufacturing Systems paper are publicly archived at Georgia Tech’s institutional repository:

### 🔗 **Dataset Archive (SMARTech)**

**[https://repository.gatech.edu/entities/publication/d2cae494-2cc7-427c-bccc-8ea950491bd3](https://repository.gatech.edu/entities/publication/d2cae494-2cc7-427c-bccc-8ea950491bd3)**
This archive contains:

* Raw 3D voxel files used for component-level autoencoder training
* Assembly-level CSVs for Stage 1–3 experiments (edges, metrics, suppliers)
* Synthetic manufacturing attributes generated using DFA rules
* Additional supplementary materials used in the DUASM paper

This GitHub repository includes all evaluation scripts and embedding files, but **pretrained models are not included** due to GitHub size limits.

---

## 1. Overview

DUASM (Deep Unsupervised Assembly Supplier Matching) is a multimodal embedding framework that encodes assembly-level geometry, topology, and manufacturing metrics into a unified latent space for supplier selection and retrieval tasks.

This repository includes:

* Case-specific assembly embeddings (Case 1, Case 2, Case 3)
* DUASM test-time evaluation scripts
* PCA visualization tool
* Example metrics masking files
* Supplier-label CSVs used for retrieval evaluation

Pretrained `.pth` checkpoint files are **not** included because GitHub restricts file sizes to 100 MB.

---

## 2. Repository Structure

```
DUASM-Assembly-dataset/
│
├── Assembly_AE/
│   ├── Case1/
│   │   ├── CAE_test_extracted_embeddings.csv
│   │   ├── CAE_train_extracted_embeddings.csv
│   │   └── DUASM/
│   │       └── trained_model/
│   │           ├── GAE_model_epoch_300.pth        (excluded if >100MB)
│   │           ├── test_assembly_edges.csv
│   │           ├── test_assembly_metrics_cost_time.csv
│   │           └── train_assembly_edges.csv
│   │
│   ├── Case2/
│   │   ├── CAE_test_extracted_embeddings.csv
│   │   └── DUASM/
│   │       └── trained_model/
│   │           ├── test_assembly_edges.csv
│   │           ├── test_assembly_metrics_cost_time.csv
│   │           └── ...
│   │
│   └── Case3/
│       └── DUASM/
│           └── trained_model/
│               ├── test_assembly_edges.csv
│               ├── train_assembly_edges.csv
│               └── ...
│
├── Validation/
│   ├── PCA_visualization.py
│   └── Example Outputs/
│
└── README.md
```

---

## 3. Pretrained Models (Not Included)

GitHub enforces a 100 MB file size limit, so large model files such as:

* `encoder_700.pth` (~178 MB)
* DUASM GAE checkpoints >100MB

are **not included in this repository**.

If pretrained models are required for academic reproduction, please contact the author.

---

## 4. Assembly-level Test Scripts

### DUASM Test Runner (Example)

Each case directory under `Assembly_AE/CaseX/DUASM/` contains a test script such as:

```
Case1_test.py
Case2_test.py
Case3_test.py
```

These scripts:

1. Load assembly embeddings
2. Load edges and manufacturing metrics
3. Load the DUASM GAE model
4. Perform threshold-based nearest-neighbor supplier retrieval
5. Output accuracy, F1-score, and confusion matrices

All paths inside the scripts have been updated to use the repository’s new folder layout.

---

## 5. PCA Visualization Tool

Located at:

```
JMS_Code/Validation/PCA_visualization.py
```

This script performs:

* Dimensionality reduction (PCA) on DUASM latent embeddings
* Supplier-wise scatter visualization
* Masking-condition-based comparative analysis

Paths are configurable and currently point to the Case2 folder structure.

---

## 6. Data Included vs. Data Hosted Externally

| Data Type                           | Location         | Notes                           |
| ----------------------------------- | ---------------- | ------------------------------- |
| Assembly embeddings (CSV)           | GitHub           | Small files                     |
| Edge files, metrics, labels         | GitHub           | Used for evaluation             |
| PCA and evaluation scripts          | GitHub           | Updated with correct paths      |
| **Voxel dataset (binvox)**          | **SMARTech**     | Full dataset archive            |
| **Synthetic manufacturing metrics** | **SMARTech**     | Cost, time, tolerance, quantity |
| **Pretrained DUASM/CAE models**     | **Not included** | Too large for GitHub            |

---

## 7. Citation

If you use this dataset or code in academic work, please cite:

**Park, S., Melkote, S. (2025). Deep Unsupervised Learning-Based Supplier Selection and Ranking for Assembly Manufacturing. Journal of Manufacturing Systems.**

And the dataset DOI from SMARTech (see archive page).

---

## 8. Contact

For pretrained models or academic collaboration:
**Suyoung Park**
Georgia Institute of Technology
[spark3026@gatech.edu](mailto:spark3026@gatech.edu)
