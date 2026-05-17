# Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts

[![CI](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** May 2026
**Status:** Manuscript in preparation

> Valentine Golden Ghanem (2026). *Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts.* GitHub repository. https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts

---

## 1. Abstract

This study characterises the subnational spatial distribution of neonatal mortality (NMR) and under-five mortality (U5MR) across Ghana's 261 health districts, identifies high-risk spatial clusters using global and local spatial autocorrelation, and develops ensemble machine learning models to predict district-level mortality risk. The analysis integrates 2022 Ghana DHS regional data, 2021 Census district estimates, and Ghana Statistical Service geospatial boundaries to generate actionable district-level risk maps.

---

## 2. Research Question & Aims

- **Primary:** Quantify the subnational distribution of NMR and U5MR across Ghana's 261 health districts.
- **Secondary:** (a) Detect spatial clusters using global Moran's I, LISA, and bivariate Moran's I (NMR × U5MR); (b) rank predictors using Random Forest Gini importance; (c) build a stacked ensemble classifier with logistic meta-learner; (d) produce district-level risk maps for programmatic targeting.

---

## 3. Methods Summary

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

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| 2022 Ghana DHS | Regional U5MR, NMR, breastfeeding, ANC coverage | 2022 | [dhsprogram.com](https://dhsprogram.com) (registration) |
| GSS 2021 Census | District population, socioeconomic indicators | 2021 | Ghana Statistical Service |
| Ghana Statistical Service GeoJSON | District boundaries (261 districts) | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |

> DHS data accessed under registration. Ghana Health Service Ethics Review Board approval obtained.

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| U5MR range | 20.0–72.0 per 1,000 live births (3.6× north–south gradient) |
| NMR range | 8.0–23.0 per 1,000 live births (median: 17.0 [IQR: 12.5–18.0]) |
| High-risk districts | 72 (27.6%) — U5MR ≥48 per 1,000 LB |
| Global Moran's I (U5MR) | 0.7783 (z=19.322, p<0.001, KNN-4) |
| Stacked ensemble CV AUC | 1.00 (ecological artifact — see discussion) |
| Top RF predictor (Gini) | Early breastfeeding initiation (0.224) |

---

## 6. Repository Structure

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

## 7. Reproducibility

### 7.1 Requirements
- Python 3.12 (see `requirements.txt` for pinned versions)
- R 4.3+ (for R scripts; see `renv.lock` or `analysis.R` header for pinned packages)
- Random seed: 42 throughout (set via `random_state=42` and `np.random.seed(42)`)
- Estimated runtime: ~5–8 minutes on a standard laptop
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install
```bash
git clone https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts.git
cd ghana-child-mortality-261-districts
pip install -r requirements.txt
# For R scripts (optional):
Rscript -e "if (!requireNamespace('renv', quietly=TRUE)) install.packages('renv'); renv::restore()"
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
# Navigate to http://127.0.0.1:8050 in your browser
```

### 7.6 Open the static HTML dashboard
Open `dashboard/Ghana_ChildMortality_261District_Dashboard.html` in any modern browser. No server required.

---

## 8. Outputs

- **Static HTML dashboard:** `dashboard/Ghana_ChildMortality_261District_Dashboard.html`
- **Poster:** `poster/`
- **Result tables:** `outputs/*.csv`
- **Figures:** `outputs/*.png` (300 DPI)

---

## 8a. Downloadable artefacts (HTML)

Both the interactive dashboard and the conference poster are committed to the repository as **self-contained HTML files** — no server, no build step. They can be:

- **Viewed in browser:** open the rendered preview, or clone the repo and open locally
- **Downloaded:** right-click → *Save link as*, or use the raw URL

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|----------------|--------------|------------------------------|
| Interactive dashboard | [`Ghana_ChildMortality_261District_Dashboard.html`](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) | [Open preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-child-mortality-261-districts/main/dashboard/Ghana_ChildMortality_261District_Dashboard.html) |
| Conference poster | [`Ghana_ChildMortality_261District_Poster.html`](https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/poster/Ghana_ChildMortality_261District_Poster.html) | [Open preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts/blob/main/poster/Ghana_ChildMortality_261District_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/ghana-child-mortality-261-districts/main/poster/Ghana_ChildMortality_261District_Poster.html) |

> **Tip:** the dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).


---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study received **Ghana Health Service Ethics Review Board approval**. All data used are de-identified secondary data from the 2022 Ghana DHS and 2021 Census. No primary data collection from individual participants was conducted.

---

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Subnational Spatial Distribution and Ensemble ML Predictors of Neonatal and Under-Five Mortality Across Ghana's 261 Health Districts*. GitHub. https://github.com/valentineghanem-bit/ghana-child-mortality-261-districts

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

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details. Outputs and figures: CC BY 4.0.

---

## 13. Author & Contact

- **Valentine Golden Ghanem**
  Ghana COCOBOD Cocoa Clinic, Accra, Ghana
  Email: [valentineghanem@gmail.com](mailto:valentineghanem@gmail.com)
  ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

- **Ghana Demographic and Health Survey programme** (ICF International) for survey data access under signed Data Use Agreement.
- **Ghana Statistical Service** for the 2021 Population and Housing Census and administrative boundary data.
- **WHO Global Health Observatory** for national-level indicators.
- **Ghana Health Service** for the Ethics Review Board approval and supportive guidance.

---

