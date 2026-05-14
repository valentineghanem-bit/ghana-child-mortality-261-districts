"""
01_spatial_analysis.py
Spatial Analysis — Child Mortality Across Ghana's 261 Health Districts
Author: Valentine Golden Ghanem | AIPOCH v6.0 | 2026-05-13
Inputs:  data/Ghana_ChildMortality_261District_MasterDataset.csv
         data/Ghana_New_260_District.geojson  (261 features)
Outputs: outputs/figures/  — choropleth maps, LISA cluster maps, Moran scatter
         outputs/spatial_results.json — all spatial statistics (canonical values)

Fail-Fast Gate: python3 -m py_compile 01_spatial_analysis.py before running
Requirements: see requirements.txt (geopandas, pysal, libpysal, esda, splot, matplotlib, scipy)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import scipy.stats as stats

from libpysal.weights import KNN, Queen
from esda.moran import Moran, Moran_Local, Moran_BV, Moran_Local_BV
from splot.esda import (plot_moran, plot_local_autocorrelation,
                        plot_moran_bv_simulation, lisa_cluster)
import libpysal

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, 'data')
FIGS    = os.path.join(ROOT, 'outputs', 'figures')
OUTDIR  = os.path.join(ROOT, 'outputs')
os.makedirs(FIGS, exist_ok=True)

np.random.seed(42)
DPI = 300

# ── Load data ─────────────────────────────────────────────────────────────────
csv  = pd.read_csv(os.path.join(DATA, 'Ghana_ChildMortality_261District_MasterDataset.csv'))
gdf  = gpd.read_file(os.path.join(DATA, 'Ghana_New_260_District.geojson'))

# Harmonise district names for join
gdf['district_clean'] = gdf['DIST_NAME'].str.strip().str.upper()
csv['district_clean']  = csv['district'].str.strip().str.upper()
gdf = gdf.merge(csv, on='district_clean', how='left')
gdf = gdf.to_crs('EPSG:32630')  # UTM Zone 30N for Ghana

print(f"GDF shape: {gdf.shape} | CRS: {gdf.crs}")
print(f"Missing U5MR in GDF: {gdf['u5mr_per1000lb'].isna().sum()}")

# ── Spatial weights ────────────────────────────────────────────────────────────
# KNN (k=4) for Global Moran's I — consistent with prior Ghana spatial papers
W_KNN  = KNN.from_dataframe(gdf, k=4)
W_KNN.transform = 'R'

# Queen contiguity for LISA — standard for administrative boundary data
W_QUEEN = Queen.from_dataframe(gdf)
W_QUEEN.transform = 'R'

print(f"KNN-4 weight matrix: {W_KNN.n} units, mean neighbours = {np.mean(list(W_KNN.cardinalities.values())):.2f}")
print(f"Queen contiguity: {W_QUEEN.n} units, mean neighbours = {np.mean(list(W_QUEEN.cardinalities.values())):.2f}")

results = {}

# ── Global Moran's I ──────────────────────────────────────────────────────────
for outcome, label in [
    ('u5mr_per1000lb', 'U5MR'),
    ('nmr_per1000lb',  'NMR'),
    ('imr_per1000lb',  'IMR'),
]:
    y = gdf[outcome].dropna().values
    # Use KNN weights aligned to non-missing indices
    gdf_valid = gdf[gdf[outcome].notna()].copy().reset_index(drop=True)
    W_k = KNN.from_dataframe(gdf_valid, k=4)
    W_k.transform = 'R'

    mi = Moran(gdf_valid[outcome].values, W_k, permutations=9999, seed=42)
    results[f'global_moran_{outcome}'] = {
        'I': round(float(mi.I), 4),
        'z_score': round(float(mi.z_norm), 4),
        'p_value': round(float(mi.p_norm), 6),
        'p_sim': round(float(mi.p_sim), 6),
        'permutations': 9999,
        'weight_matrix': 'KNN-4',
        'interpretation': 'Significant positive spatial autocorrelation' if mi.p_sim < 0.05 else 'No significant autocorrelation'
    }
    print(f"\nGlobal Moran's I ({label}): I={mi.I:.4f}, z={mi.z_norm:.3f}, p_sim={mi.p_sim:.4f}")

    # Moran scatterplot
    fig, ax = plot_moran(mi, zstandard=True, aspect_equal=True, figsize=(8, 7))
    ax[0].set_title(f"Global Moran's I — {label}\n(I={mi.I:.4f}, p<{mi.p_sim:.3f})", fontsize=13, fontweight='semibold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'moran_scatter_{outcome}.pdf'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'moran_scatter_{outcome}.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

# ── Local Moran's I (LISA) ────────────────────────────────────────────────────
for outcome, label in [
    ('u5mr_per1000lb', 'U5MR'),
    ('nmr_per1000lb',  'NMR'),
]:
    gdf_valid = gdf[gdf[outcome].notna()].copy().reset_index(drop=True)
    W_q = Queen.from_dataframe(gdf_valid)
    W_q.transform = 'R'

    lisa = Moran_Local(gdf_valid[outcome].values, W_q, permutations=9999, seed=42)
    gdf_valid['lisa_q'] = lisa.q
    gdf_valid['lisa_p'] = lisa.p_sim
    gdf_valid['lisa_sig'] = (lisa.p_sim < 0.05).astype(int)

    # Classify: 1=HH, 2=LH, 3=LL, 4=HL (significant only)
    gdf_valid['lisa_cluster'] = 0  # Not significant
    sig = gdf_valid['lisa_sig'] == 1
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 1), 'lisa_cluster'] = 1  # HH
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 2), 'lisa_cluster'] = 2  # LH
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 3), 'lisa_cluster'] = 3  # LL
    gdf_valid.loc[sig & (gdf_valid['lisa_q'] == 4), 'lisa_cluster'] = 4  # HL

    cluster_counts = gdf_valid['lisa_cluster'].value_counts()
    results[f'lisa_{outcome}'] = {
        'HH_clusters': int(cluster_counts.get(1, 0)),
        'LH_clusters': int(cluster_counts.get(2, 0)),
        'LL_clusters': int(cluster_counts.get(3, 0)),
        'HL_clusters': int(cluster_counts.get(4, 0)),
        'not_significant': int(cluster_counts.get(0, 0)),
        'total_significant': int(gdf_valid['lisa_sig'].sum()),
        'weight_matrix': 'Queen contiguity',
        'significance_level': 0.05,
        'permutations': 9999
    }
    print(f"\nLISA ({label}): HH={cluster_counts.get(1,0)}, LH={cluster_counts.get(2,0)}, "
          f"LL={cluster_counts.get(3,0)}, HL={cluster_counts.get(4,0)}, "
          f"NS={cluster_counts.get(0,0)}")

    # LISA cluster map
    colours = {0: '#d9d9d9', 1: '#d7191c', 2: '#abd9e9', 3: '#2c7bb6', 4: '#fdae61'}
    cmap_vals = [colours[v] for v in gdf_valid['lisa_cluster']]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf_valid.plot(color=cmap_vals, linewidth=0.3, edgecolor='white', ax=ax)
    patches = [
        mpatches.Patch(color='#d7191c', label=f"High–High (n={cluster_counts.get(1,0)})"),
        mpatches.Patch(color='#fdae61', label=f"High–Low (n={cluster_counts.get(4,0)})"),
        mpatches.Patch(color='#abd9e9', label=f"Low–High (n={cluster_counts.get(2,0)})"),
        mpatches.Patch(color='#2c7bb6', label=f"Low–Low (n={cluster_counts.get(3,0)})"),
        mpatches.Patch(color='#d9d9d9', label=f"Not significant (n={cluster_counts.get(0,0)})"),
    ]
    ax.legend(handles=patches, loc='lower left', fontsize=10, title='LISA Cluster', title_fontsize=11)
    ax.set_title(f"LISA Cluster Map — {label} per 1,000 Live Births\nGhana 261 Health Districts, 2022 DHS",
                 fontsize=13, fontweight='semibold', pad=12)
    ax.set_axis_off()
    plt.figtext(0.5, 0.01,
        f"Queen contiguity spatial weights | 9,999 permutations | p<0.05 significance threshold",
        ha='center', fontsize=9, style='italic')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(FIGS, f'lisa_cluster_{outcome}.pdf'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'lisa_cluster_{outcome}.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

# ── Bivariate Moran's I: U5MR × predictors ───────────────────────────────────
bivariate_pairs = [
    ('u5mr_per1000lb', 'open_defecation_pct',    'U5MR × Open Defecation'),
    ('u5mr_per1000lb', 'poverty_rate_pct',        'U5MR × Poverty Rate'),
    ('u5mr_per1000lb', 'fully_vaccinated_pct',    'U5MR × Full Vaccination'),
    ('u5mr_per1000lb', 'women_secondary_plus_pct','U5MR × Women Secondary Education'),
    ('nmr_per1000lb',  'women_secondary_plus_pct','NMR × Women Secondary Education'),
]

results['bivariate_moran'] = {}
for y_col, x_col, label in bivariate_pairs:
    valid_mask = gdf[y_col].notna() & gdf[x_col].notna()
    gdf_v = gdf[valid_mask].copy().reset_index(drop=True)
    W_q = Queen.from_dataframe(gdf_v)
    W_q.transform = 'R'
    bv = Moran_BV(gdf_v[y_col].values, gdf_v[x_col].values, W_q, permutations=9999, seed=42)
    results['bivariate_moran'][label] = {
        'I_bv': round(float(bv.I), 4),
        'p_sim': round(float(bv.p_sim), 6),
        'z_sim': round(float(bv.z_sim), 4),
    }
    print(f"Bivariate Moran's I ({label}): I={bv.I:.4f}, p={bv.p_sim:.4f}")

# ── Choropleth maps ───────────────────────────────────────────────────────────
for col, title, cmap_name in [
    ('u5mr_per1000lb', 'Under-Five Mortality Rate', 'YlOrRd'),
    ('nmr_per1000lb',  'Neonatal Mortality Rate',   'YlOrRd'),
    ('poverty_rate_pct', 'Poverty Incidence (%)', 'YlOrBr'),
    ('open_defecation_pct', 'Open Defecation (%)', 'RdPu'),
    ('women_secondary_plus_pct', 'Women Secondary+ Education (%)', 'Blues'),
]:
    if gdf[col].isna().all():
        continue
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(column=col, cmap=cmap_name, linewidth=0.3, edgecolor='white',
             ax=ax, legend=True,
             legend_kwds={'label': col.replace('_', ' ').title(),
                          'orientation': 'vertical', 'shrink': 0.6})
    ax.set_title(f"{title}\nGhana 261 Health Districts, 2022 DHS",
                 fontsize=13, fontweight='semibold', pad=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'choropleth_{col}.pdf'), dpi=DPI, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'choropleth_{col}.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

# ── Save canonical spatial results ────────────────────────────────────────────
with open(os.path.join(OUTDIR, 'spatial_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("SPATIAL ANALYSIS COMPLETE")
print(f"Results saved: {os.path.join(OUTDIR, 'spatial_results.json')}")
print(f"Figures saved: {FIGS}")
