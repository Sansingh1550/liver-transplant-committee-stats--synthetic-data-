**Liver Transplant Committee — Statistical Analysis (Synthetic Data)**
Statistical evaluation pipeline for a game-theoretic multi-agent LLM
committee simulating a liver transplant selection committee, developed
as part of a research project at the Bhat Liver Lab, University of Toronto.

Code for evaluating the committee's listing decisions against ground truth
outcomes and screening for demographic disparities in its errors:

- **PPV / NPV** with Wilson score 95% confidence intervals
- **McNemar's test** — a paired comparison testing whether two
  decision-aggregation methods (a normal-form game vs. a cooperative
  normal-form game) differ significantly on the same patients
- **Paired bootstrap confidence intervals** — an effect-size estimate
  to accompany McNemar's significance test
- **Subgroup fairness screening** — two-proportion z-tests comparing
  false positive / false negative rates for each demographic subgroup
  against the rest of the cohort, for each decision method
- **Benjamini-Hochberg (FDR) correction** — controls the false-positive
  rate across the many simultaneous subgroup tests being run

## Why the data is synthetic

The original analysis was run on data from the Scientific Registry of
Transplant Recipients (SRTR), which is restricted under a data use
agreement and cannot be shared or reproduced publicly. All data in this
repo is randomly generated to match the shape of the real dataset, so the
statistical methodology can be run and inspected end-to-end without any
real patient data.

**Results produced by this repo carry no clinical meaning** — they reflect
patterns (or lack thereof) in randomly generated numbers only.
