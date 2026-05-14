"""
03_descriptive_stats.py
Descriptive Statistics & Table 1 — Child Mortality Ghana 261 Districts
Author: Valentine Golden Ghanem | AIPOCH v6.0 | 2026-05-13
Outputs: outputs/table1.csv, outputs/descriptive_summary.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, 'data')
OUT   = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'Ghana_ChildMortality_261District_MasterDataset.csv'))

CONTINUOUS = {
    'Neonatal Mortality Rate (per 1,000 LB)': 'nmr_per1000lb',
    'Under-Five Mortality Rate (per 1,000 LB)': 'u5mr_per1000lb',
    'Infant Mortality Rate (per 1,000 LB)': 'imr_per1000lb',
    'Postneonatal Mortality Rate (per 1,000 LB)': 'pnmr_per1000lb',
    'Poverty Incidence (%)': 'poverty_rate_pct',
    'Poverty Intensity (%)': 'poverty_intensity_pct',
    'Illiteracy Rate (%)': 'illiteracy_rate_pct',
    'Uninsurance Rate (%)': 'uninsurance_rate_pct',
    'Youth Dependency Ratio': 'youth_dependency_ratio',
    'Employment Rate (%)': 'employment_rate_pct',
    'BCG Coverage (%)': 'bcg_coverage_pct',
    'DPT3 Coverage (%)': 'dpt3_coverage_pct',
    'Measles Coverage (%)': 'measles_coverage_pct',
    'Fully Vaccinated (%)': 'fully_vaccinated_pct',
    'Diarrhoea Prevalence (%)': 'diarrhea_prevalence_pct',
    'ORS Use (%)': 'ors_use_pct',
    'Early Breastfeeding Initiation (%)': 'early_breastfeeding_pct',
    'Minimum Dietary Diversity (%)': 'dietary_diversity_pct',
    'Child Anaemia — Any (%)': 'child_anemia_any_pct',
    'Child Anaemia — Moderate/Severe (%)': 'child_anemia_moderate_pct',
    'Improved Water Access (%)': 'improved_water_pct',
    'Open Defecation (%)': 'open_defecation_pct',
    'Improved Sanitation (%)': 'improved_sanitation_pct',
    'Women — No Education (%)': 'women_no_edu_pct',
    'Women — Secondary+ Education (%)': 'women_secondary_plus_pct',
    'Female Literacy (%)': 'female_literacy_pct',
    'Total Population': 'total_pop',
}

rows = []
summary_dict = {}

for label, col in CONTINUOUS.items():
    if col not in df.columns:
        continue
    vals = df[col].dropna()
    n = len(vals)
    if n == 0:
        continue

    mean_v  = vals.mean()
    sd_v    = vals.std()
    med_v   = vals.median()
    iqr_v   = vals.quantile(0.75) - vals.quantile(0.25)
    min_v   = vals.min()
    max_v   = vals.max()
    sk_v    = float(stats.skew(vals))
    _, pnorm = stats.shapiro(vals[:50] if len(vals) > 50 else vals)

    rows.append({
        'Variable': label,
        'N': int(n),
        'Mean (SD)': f"{mean_v:.1f} ({sd_v:.1f})",
        'Median [IQR]': f"{med_v:.1f} [{vals.quantile(0.25):.1f}–{vals.quantile(0.75):.1f}]",
        'Range': f"{min_v:.1f}–{max_v:.1f}",
        'Skewness': f"{sk_v:.2f}",
        'Normality (Shapiro p)': f"{pnorm:.4f}",
        'Preferred summary': 'Median [IQR]' if abs(sk_v) > 1 or pnorm < 0.05 else 'Mean (SD)',
    })
    summary_dict[col] = {
        'n': int(n), 'mean': round(float(mean_v), 3), 'sd': round(float(sd_v), 3),
        'median': round(float(med_v), 3), 'q25': round(float(vals.quantile(0.25)), 3),
        'q75': round(float(vals.quantile(0.75)), 3),
        'min': round(float(min_v), 3), 'max': round(float(max_v), 3),
        'skewness': round(sk_v, 3), 'shapiro_p': round(float(pnorm), 6),
    }

# Table 1 stratified by U5MR high risk
t1_rows = []
for label, col in CONTINUOUS.items():
    if col not in df.columns:
        continue
    high = df[df['u5mr_high_risk'] == 1][col].dropna()
    low  = df[df['u5mr_high_risk'] == 0][col].dropna()
    if len(high) == 0 or len(low) == 0:
        continue
    _, p_test = stats.mannwhitneyu(high, low, alternative='two-sided')
    t1_rows.append({
        'Variable': label,
        'High-Risk Districts (n=72)': f"{high.median():.1f} [{high.quantile(0.25):.1f}–{high.quantile(0.75):.1f}]",
        'Lower-Risk Districts (n=189)': f"{low.median():.1f} [{low.quantile(0.25):.1f}–{low.quantile(0.75):.1f}]",
        'p-value (Mann-Whitney U)': f"{p_test:.4f}" if p_test >= 0.001 else '<0.001',
    })

t1_df = pd.DataFrame(t1_rows)
t1_df.to_csv(os.path.join(OUT, 'table1.csv'), index=False)

full_df = pd.DataFrame(rows)
full_df.to_csv(os.path.join(OUT, 'descriptive_full.csv'), index=False)

with open(os.path.join(OUT, 'descriptive_summary.json'), 'w') as f:
    json.dump(summary_dict, f, indent=2)

print(f"Table 1 saved: {len(t1_rows)} variables")
print(f"Descriptive summary: {len(summary_dict)} variables")

# Print key statistics
print("\n── KEY CANONICAL VALUES ──────────────────────────────────────────")
for col in ['nmr_per1000lb', 'imr_per1000lb', 'u5mr_per1000lb']:
    s = summary_dict.get(col, {})
    print(f"{col}: median={s.get('median','?')} [IQR {s.get('q25','?')}–{s.get('q75','?')}], "
          f"range {s.get('min','?')}–{s.get('max','?')}")
