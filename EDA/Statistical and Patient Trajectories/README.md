# ADNI Statistical & Patient Trajectory EDA

This analysis focuses on statistical feature evaluation and longitudinal patient-level behavior within the ADNI dataset.

The goal is to identify which features significantly differ across diagnostic groups and to visualize how these features evolve over time at both the population and individual levels.

---

## Overview

This notebook combines:

- Statistical hypothesis testing (ANOVA)
- Distribution analysis across diagnosis groups
- Longitudinal patient trajectory visualization
- Population-level patient count distributions

It is designed to support **feature selection and modeling readiness**.

---

## Key Components

### 1. Statistical Feature Testing

A batch ANOVA pipeline is applied across all numerical features to evaluate their relationship with diagnosis:

- Compares feature distributions across diagnostic groups
- Outputs F-statistics and p-values
- Ranks features by statistical significance

```python
run_batch_anova(dataframe, target_col)
