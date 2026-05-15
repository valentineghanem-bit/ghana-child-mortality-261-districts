# Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)  
**Reporting standard:** STROBE  
**Date:** May 2026

> Ghanem VG. *Subnational spatial distribution and ensemble ML predictors of neonatal and under-five mortality across Ghana's 261 health districts.* 2026.

---

## Overview

This study characterises the subnational spatial distribution of neonatal mortality (NMR) and under-five mortality (U5MR) across Ghana's 261 health districts, identifies high-risk spatial clusters using global and local spatial autocorrelation, and develops ensemble machine learning models to predict district-level mortality risk. The analysis integrates 2022 Ghana DHS regional data, 2021 Census district estimates, and Ghana Statistical Service geospatial boundaries to generate actionable district-level risk maps.

---

## Key Findings

| Metric | Value |
|--------|-------|
| U5MR range | 20.0–72.0 per 1,000 live births (3.6× north–south gradient) |
| NMR range | 8.0–23.0 per 1,000 live births (median: 17.0 [IQR: 12.5–18.0]) |
| High-risk districts | 72 (27.6%) — U5MR ≥48 per 1,000 LB |
| Global Moran's I (U5MR) | 0.7783 (z=19.322, p<0.001, KNN-4) |
| Stacked ensemble CV AUC | 1.00 (ecological artifact) |
| Top RF predictor (Gini) | Early breastfeeding initiation (0.224) |

---

## Repository Structure

```
ghana-child-mortality-261-districts/
├── data/
├── scripts/
│   ├── 01_spatial_analysis.py
│   ├── 02_ml_pipeline.py
│   └── 03_descriptive_stats.py
├── dashboard/
│   └── Ghana_ChildMortality_261District_Dashboard.html
├── poster/
├── tests/
├── outputs/
├── requirements.txt
└── CITATION.cff
```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts.git
cd ghana-child-mortality-261-districts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python scripts/03_descriptive_stats.py
python scripts/01_spatial_analysis.py
python scripts/02_ml_pipeline.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Open the interactive dashboard

Open `dashboard/Ghana_ChildMortality_261District_Dashboard.html` in any modern browser. No server required.

---

## Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| 2022 Ghana DHS | Regional U5MR, NMR, breastfeeding, ANC coverage | 2022 | dhsprogram.com (registration) |
| GSS 2021 Census | District population, socioeconomic indicators | 2021 | Ghana Statistical Service |
| Ghana Statistical Service GeoJSON | District boundaries (261 districts) | 2021 | statsghana.gov.gh |

---

## Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Global Moran's I (KNN-4, 999 permutations) | esda / libpysal | Spatial autocorrelation of U5MR and NMR |
| LISA | esda | Local spatial cluster detection |
| Bivariate Moran's I | esda | Co-clustering of NMR and U5MR |
| Random Forest | scikit-learn | Predictor importance (Gini impurity) |
| Gradient Boosting | scikit-learn | Mortality risk prediction |
| Stacked Ensemble (logistic meta-learner) | scikit-learn | Final ensemble classification |
| 10-fold stratified CV | scikit-learn | Model validation |

---

## Reproducibility

- Random seed: 42 throughout  
- Reporting: STROBE  
- All random seeds set explicitly (`random_state=42`)  
- Spatial weights: KNN k=4 for global Moran's I; 999 permutations for all significance tests

---

## Ethical Statement

This study received Ghana Health Service Ethics Review Board approval. All data used are de-identified secondary data from the 2022 Ghana DHS and 2021 Census. No primary data collection from individual participants was conducted.

---

## Citation

```bibtex
@misc{ghanem2026childmortality,
  author = {Ghanem, Valentine Golden},
  title  = {Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contact

Valentine Golden Ghanem  
Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
valentineghanem@gmail.com  
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
