# Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts

[![CI](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** May 2026
**Status:** Manuscript in preparation

---

## 1. Abstract

This study characterises the subnational spatial distribution of neonatal mortality (NMR) and under-five mortality (U5MR) across Ghana's 261 health districts, identifies high-risk spatial clusters using global and local spatial autocorrelation, and develops ensemble machine learning models to predict district-level mortality risk. The analysis integrates 2022 Ghana DHS regional data, 2021 Census district estimates, and Ghana Statistical Service geospatial boundaries to generate actionable district-level risk maps. A stacked ensemble (Random Forest + Gradient Boosting with logistic meta-learner) under 10-fold stratified cross-validation is used for risk classification.

---

## 2. Research Question & Aims

- **Primary:** Quantify the subnational distribution of NMR and U5MR across Ghana's 261 health districts.
- **Secondary:** (a) Detect spatial clusters using Global Moran's I, LISA, and bivariate Moran's I (NMR × U5MR); (b) rank predictors using Random Forest Gini importance; (c) build a stacked ensemble classifier with logistic meta-learner; (d) produce district-level risk maps for programmatic targeting.

---

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Global Moran's I (KNN-4, 999 permutations) | esda / libpysal | Spatial autocorrelation of U5MR and NMR |
| LISA | esda | Local spatial cluster detection |
| Bivariate Moran's I | esda | NMR × U5MR co-clustering |
| Random Forest | scikit-learn | Predictor importance (Gini impurity) |
| Gradient Boosting | scikit-learn | Mortality risk prediction |
| Stacked ensemble (logistic meta-learner) | scikit-learn | Final ensemble classification |
| 10-fold stratified CV | scikit-learn | Model validation |
| Spatial diagnostics (R) | spdep / spatialreg | OLS / SLM / SEM model selection |

---

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| Ghana DHS 2022 | Regional U5MR, NMR, breastfeeding, ANC coverage | 2022 | [dhsprogram.com](https://dhsprogram.com) (registration) |
| GSS 2021 Census | District population, socioeconomic indicators | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| Ghana Statistical Service GeoJSON | District boundaries (261 districts) | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |

> DHS data accessed under standard DHS Programme Data Use Agreement. No individual participant data redistributed.

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| U5MR range | 20.0–72.0 per 1,000 live births (3.6× north–south gradient) |
| NMR range | 8.0–23.0 per 1,000 live births (median: 17.0 [IQR: 12.5–18.0]) |
| High-risk districts | 72 (27.6%) — U5MR ≥48 per 1,000 LB |
| Global Moran's I (U5MR) | 0.7783 (z=19.322, p<0.001, KNN-4) |
| Stacked ensemble CV AUC | 1.00 (ecological aggregation artefact — see discussion) |
| Top RF predictor (Gini) | Early breastfeeding initiation (0.224) |
| Districts analysed | 261 |

---

## 6. Repository Structure

```
ghana-child-mortality-261-districts/
├── scripts/
│   ├── 01_spatial_analysis.py
│   ├── 02_ml_pipeline.py
│   ├── 03_descriptive_stats.py
│   ├── spatial_utils.py            # Reusable spatial analysis utilities
│   └── spatial_diagnostics.R       # R: spatial autocorrelation diagnostics
├── app.py                          # Plotly Dash interactive application
├── analysis.R                      # R: spatial regression + model selection
├── dashboard/
│   └── Ghana_ChildMortality_261District_Dashboard.html
├── poster/
│   └── Ghana_ChildMortality_261District_Poster.html
├── data/
├── outputs/
├── tests/
├── requirements.txt
└── CITATION.cff
```

---

## 7. Reproducibility

### 7.1 Requirements

- Python 3.12 (pinned in `requirements.txt`)
- R 4.3+ with packages: spdep, spatialreg, dplyr (see `analysis.R` header)
- Random seed: 42 throughout
- Estimated runtime: ~5–8 minutes on a standard laptop
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install

```bash
git clone https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts.git
cd ghana-child-mortality-261-districts
pip install -r requirements.txt
```

### 7.3 Run the analytical pipeline

```bash
python scripts/03_descriptive_stats.py
python scripts/01_spatial_analysis.py
python scripts/02_ml_pipeline.py
```

### 7.4 Run the test suite

```bash
pytest tests/ -v
```

### 7.5 Launch the interactive Dash application

```bash
python app.py
# Visit http://127.0.0.1:8050
```

### 7.6 Open the static HTML dashboard

```bash
# macOS
open dashboard/Ghana_ChildMortality_261District_Dashboard.html
# Windows
start dashboard/Ghana_ChildMortality_261District_Dashboard.html
# Linux
xdg-open dashboard/Ghana_ChildMortality_261District_Dashboard.html
```

---

## 8. Outputs

| Output | Description |
|--------|-------------|
| `data/processed/` | Master CSV, spatial weights, risk scores |
| `outputs/figures/` | Publication-quality PNG figures (300 DPI) |
| `dashboard/` | Self-contained interactive HTML dashboard |
| `poster/` | A0 conference poster (HTML, print-ready) |

## 8a. Downloadable Artefacts (HTML)

Both the interactive dashboard and the conference poster are committed as self-contained HTML files — no server, no build step required.

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|---------------|--------------|---------------------------|
| Interactive dashboard | [View](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-child-mortality-261-districts/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) |
| Conference poster | [View](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/poster/Ghana_ChildMortality_261District_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/poster/Ghana_ChildMortality_261District_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-child-mortality-261-districts/main/poster/Ghana_ChildMortality_261District_Poster.html) |

> **Tip:** The dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study analyses publicly released aggregate data from the Ghana Demographic and Health Survey 2022 (ICF International) and the Ghana Statistical Service 2021 Population and Housing Census. No individual participant data were accessed. All inputs are de-identified district and regional summary statistics. Ethical review was not required for analysis of publicly available aggregate statistics; DHS data were accessed under the standard DHS Programme Data Use Agreement.

---

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts.* GitHub. https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts

**BibTeX:**
```bibtex
@misc{ghanem2026childmortality,
  author = {Ghanem, Valentine Golden},
  title  = {Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts}
}
```

A machine-readable citation is provided in `CITATION.cff`.

---

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details.
Outputs and figures: **CC BY 4.0**.

---

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana
Email: valentineghanem@gmail.com
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

The author thanks the DHS Programme and ICF International for the 2022 Ghana DHS, and the Ghana Statistical Service for the 2021 Census district files and boundary GeoJSON. Spatial analysis relied on esda, libpysal, and spdep. Ensemble modelling used scikit-learn. The North–South mortality gradient documented here highlights a persistent structural disparity that pre-dates the survey data and demands sustained programmatic attention.
