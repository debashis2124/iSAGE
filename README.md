# iSAGE

**iSAGE** is a lightweight neuro-symbolic framework for clinical AI under resource scarcity.  
The project studies how explicit symbolic knowledge affects prediction when labeled data, patient observations, neural capacity, or knowledge quality become limited.

The main goal is not to show that symbolic knowledge always improves prediction. Instead, iSAGE evaluates when symbolic knowledge helps, when its effect becomes limited, and when incorrect knowledge can become harmful.

## Overview

iSAGE combines a lightweight neural model with explicit symbolic knowledge.

The neural component learns predictive patterns from patient data, while the symbolic component represents domain knowledge through task-specific rules. Symbolic evidence is used only when relevant rules are active and is combined with the neural output through conditional fusion.

The framework studies four main resource dimensions:

- Labeled-data availability
- Observation availability
- Computational capacity
- Symbolic knowledge reliability

iSAGE also compares symbolic knowledge introduced during neural model fitting with symbolic knowledge added after neural training.

## Main Research Questions

The experiments investigate the following questions:

1. Can symbolic knowledge support clinical prediction when labeled training data are limited?
2. Can symbolic reasoning improve robustness when patient observations are incomplete?
3. Can symbolic knowledge support lightweight neural models under limited computational capacity?
4. How do missing or incorrect symbolic rules affect neuro-symbolic prediction?

## Datasets

The experiments use three binary healthcare prediction datasets:

### Breast Cancer
Breast Cancer Classification Dataset  
Kaggle:  
https://www.kaggle.com/datasets/boiniabhiram/breast-cancer-classifiacation

### Cardiovascular Disease
Cardiovascular Disease Dataset  
Kaggle:  
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

### Diabetes
Diabetes Prediction Dataset  
Kaggle:  
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Please download the datasets directly from the original sources and place them in the appropriate local data directory before running the experiments.

## Methods Compared

The experiments include the following configurations:

- **S-NN**: neural-only baseline using the matched lightweight neural architecture.
- **Rule-Only**: symbolic prediction without neural evidence.
- **iSAGE-Train**: rule-derived symbolic information is included in the neural input during model fitting.
- **iSAGE-Fusion**: the neural model is trained first and symbolic probability is conditionally fused with the neural prediction afterward.

## Resource Scarcity Experiments

iSAGE is evaluated under several controlled resource conditions.

### Training-Data Scarcity
The available labeled training data are progressively reduced to study whether symbolic knowledge becomes more useful when neural supervision is limited.

### Observation Scarcity
Clinical observations are progressively removed. Missing features affect both the neural input and the availability of symbolic rules.

### Computational Constraints
The neural model is evaluated under restricted computational capacity using lightweight model configurations.

### Knowledge Reliability
Symbolic rules are deliberately removed or reversed.

- **Rule removal** represents missing knowledge.
- **Rule reversal** represents incorrect knowledge.

This experiment identifies when symbolic knowledge stops providing a predictive advantage.

### Joint Resource Degradation
Multiple resource limitations are applied together to evaluate how well each model preserves its full-resource performance.

## Evaluation Metrics

The experiments consider:

- AUROC
- Macro-F1
- Balanced Accuracy
- Sensitivity
- Specificity
- Brier Score
- Expected Calibration Error
- Rule-Violation Rate
- Symbolic Coverage
- Performance Retention
- Resource Robustness Score

Statistical comparisons are performed across independent random seeds.

## Main Findings

The experiments show that the value of symbolic knowledge depends on the dataset and resource condition. Under strong training-data scarcity, iSAGE-Fusion provides useful gains for Breast Cancer and Cardiovascular prediction. The Diabetes experiments show that symbolic fusion is not always beneficial, especially when the symbolic knowledge does not sufficiently complement the neural model. The knowledge-reliability experiments also show that incorrect symbolic knowledge can be more harmful than missing knowledge. Rule removal mainly reduces symbolic support, while reversed rules can push predictions in the wrong direction. The comparison between training-time and post-training symbolic integration shows that knowledge timing is task dependent. Post-training fusion is more favorable for Breast Cancer and Cardiovascular prediction under severe scarcity, while training-time integration is more competitive for Diabetes in several conditions.

## Repository Structure

A typical project structure is:
```bash
iSAGE/
├── data/
├── results/
│   ├── figures/
│   └── tables/
├── src/
├── run_complete.py
├── requirements.txt
└── README.md
```
## How to Run

Follow the steps below to run the iSAGE experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/debashis2124/iSAGE.git
cd iSAGE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pip install -r requirements.txt
python run_iSAGE.py


One thing to verify before adding this: if your actual main Python file is not `iSAGE.py`, replace that filename with the exact script name used in the repository. If you need more details, contact: debashis.das@ieee.org
