# ADNI MRI Longitudinal Dynamics EDA

This analysis focuses on the structure, retention patterns, and longitudinal behavior of MRI imaging data within the ADNI dataset.

The goal is to understand how imaging data evolves over time, how patient participation varies, and how these patterns impact downstream modeling.

---

## Overview

MRI imaging in ADNI is collected longitudinally, but patient participation is highly uneven.

This EDA examines:

- Scan frequency per patient  
- Follow-up duration  
- Dropout patterns  
- Differences between stable vs progressing patients  
- Alignment between imaging and tabular data  

---

## Dataset Summary

- Total patients: 2,922  
- Total MRI scans: 10,868  
- Avg scans per patient: 3.72  
- Median scans per patient: 3  
- Max scans: 16  
- Avg follow-up duration: 2.65 years  
- Max follow-up duration: ~20 years  

:contentReference[oaicite:0]{index=0}

---

## Key Findings

### 1. High Early Attrition

- ~38% of patients have only a single baseline scan  
- These patients contribute **cross-sectional data only**  
- Represents a major limitation for longitudinal modeling  

---

### 2. Typical Longitudinal Window

- Most patients who continue participation remain for **1–3 years**  
- The majority complete **1–5 scans**  

---

### 3. Long-Term “Super-Participants”

- A small subset of patients are tracked for up to **20 years**  
- These patients have up to **16 scans**  
- They form a **high-value subset for time-series modeling**  

---

### 4. Retention Differs by Disease Progression

Progressing patients are tracked far more consistently:

- **MCI → AD:** ~91% have ≥4 scans  
- **CN → AD:** ~95% have ≥4 scans  

Stable patients drop out significantly:

- **CN → CN:** ~62% drop by 4th scan  
- **MCI → MCI:** ~50% drop by 4th scan  

:contentReference[oaicite:1]{index=1}

---

### 5. Imaging vs Tabular Data Gap

There is a mismatch between imaging and tabular availability:

- Many patients with MRI data lack complete tabular records  
- This is especially pronounced in stable cohorts (CN → CN)

---

## Implications for Modeling

### ⚠️ Longitudinal Bias
- Models trained on MRI sequences will be biased toward **progressing patients**

### ⚠️ Data Imbalance
- Large volume of single-scan patients vs smaller longitudinal cohort

### ⚠️ Multimodal Constraints
- Combining MRI + tabular data reduces usable sample size

### ✅ Opportunity
- Long-term participants provide **high-quality time-series data**

---

## Role in Overall Project

This EDA provides critical insight into:

- MRI data availability  
- longitudinal cohort structure  
- dropout behavior  
- modeling feasibility for imaging-based approaches  

It complements:

- multimodal integration EDA  
- feature deep dive EDA  
- data merge pipeline  

---

## Files
