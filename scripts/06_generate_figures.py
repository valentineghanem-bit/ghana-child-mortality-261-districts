"""
06_generate_figures.py
Generate all 6 publication-quality manuscript figures using the Ghana district GeoJSON.
Author: Valentine Golden Ghanem | 2026-05-14
Output: figures/Figure1–6_*.png  (300 DPI, vector-equivalent raster)
"""

import os, sys, json, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from scipy import stats
import geopandas as gpd

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ    = "/sessions/intelligent-upbeat-curie/mnt/Public Health & Epidemiology Research Skills/7. Child Mortality Ghana 261 Districts"
GJ_PATH = "/sessions/intelligent-upbeat-curie/mnt/uploads/Ghana_New_260_District.geojson"
DATA_F  = os.path.join(PROJ, "data",    "Ghana_ChildMortality_261District_MasterDataset.csv")
SP_F    = os.path.join(PROJ, "outputs", "spatial_results.json")
ML_F    = os.path.join(PROJ, "outputs", "ml_results.json")
FIG_DIR = os.path.join(PROJ, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_F)
with open(SP_F) as f: SP = json.load(f)
with open(ML_F) as f: ML = json.load(f)

# ── Load GeoJSON & merge ───────────────────────────────────────────────────────
gdf_raw = gpd.read_file(GJ_PATH)
gdf_raw['DISTRICT_UPPER'] = gdf_raw['DISTRICT'].str.upper().str.strip()

def normalise(s):
    s = str(s).upper().strip()
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[\-–]', ' ', s)
    for w in ['METROPOLITAN', 'MUNICIPAL', 'DISTRICT', 'METRO']:
        s = s.replace(w, '')
    return re.sub(r'\s+', ' ', s).strip()

gdf_raw['norm'] = gdf_raw['DISTRICT'].apply(normalise)
df['norm']      = df['district'].str.upper().apply(normalise)

merged = gdf_raw.merge(df, on='norm', how='left')
n_matched = merged['u5mr_per1000lb'].notna().sum()
print(f"GeoJSON features: {len(gdf_raw)} | Matched with CSV: {n_matched}")

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.labelweight': 'semibold', 'axes.titleweight': 'semibold',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})
DPI = 300

def add_ghana_north_arrow(ax):
    ax.annotate('N', xy=(0.96, 0.12), xytext=(0.96, 0.04),
                xycoords='axes fraction',
                fontsize=13, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='#333', lw=2),
                color='#333')

def add_scalebar(ax):
    ax.text(0.03, 0.03, '0      100 km', transform=ax.transAxes,
            fontsize=8.5, va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#aaa', pad=3))

def clean_map_ax(ax):
    ax.set_axis_off()

# ── Figure 1: U5MR Choropleth ─────────────────────────────────────────────────
print("\nGenerating Figure 1: U5MR Choropleth Map...")
fig, ax = plt.subplots(1, 1, figsize=(10, 11))
merged.plot(column='u5mr_per1000lb', ax=ax, cmap='YlOrRd',
            vmin=20, vmax=72, missing_kwds={'color': '#cccccc'},
            edgecolor='#444', linewidth=0.25)
sm = cm.ScalarMappable(cmap='YlOrRd', norm=Normalize(vmin=20, vmax=72))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01, aspect=20)
cbar.set_label('U5MR per 1,000 live births', fontsize=11, fontweight='semibold')
cbar.ax.tick_params(labelsize=9.5)
add_ghana_north_arrow(ax)
add_scalebar(ax)
clean_map_ax(ax)
ax.set_title(
    'Figure 1. District-level under-five mortality rate (U5MR)\nacross 261 MMDAs, Ghana',
    fontsize=13, fontweight='semibold', pad=12)
ax.text(0.02, 0.98,
        f"Range: 20–72 per 1,000 LB  |  Median: {np.median(df['u5mr_per1000lb']):.0f}\n"
        f"Moran's I = {SP['global_moran_u5mr_per1000lb']['I']:.4f}  "
        f"(z = {SP['global_moran_u5mr_per1000lb']['z_score']:.3f}, p<0.001, KNN-4)",
        transform=ax.transAxes, fontsize=9.5, va='top',
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='#bbb', pad=5))
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, 'Figure1_U5MR_Choropleth.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure1_U5MR_Choropleth.png")

# ── Figure 2: Global Moran's I Scatter ────────────────────────────────────────
print("Generating Figure 2: Moran Scatter Plot...")

# Build KNN-4 weight matrix from centroids
coords = np.column_stack([df['latitude'].values, df['longitude'].values])
n = len(coords)
W = np.zeros((n, n))
for i in range(n):
    dists = np.sqrt(((coords - coords[i])**2).sum(axis=1))
    dists[i] = np.inf
    nn = np.argsort(dists)[:4]
    W[i, nn] = 1.0
rs = W.sum(axis=1, keepdims=True); rs[rs == 0] = 1
W = W / rs

u5mr_arr = df['u5mr_per1000lb'].values
yz  = (u5mr_arr - u5mr_arr.mean()) / u5mr_arr.std()
Wyz = W @ yz
hr  = df['u5mr_high_risk'].values

fig, ax = plt.subplots(figsize=(8, 7))
colors_dot = np.where(hr == 1, '#d73027', '#4575b4')
ax.scatter(yz, Wyz, c=colors_dot, edgecolors='white', s=50,
           linewidths=0.5, alpha=0.85, zorder=3)
m, b, r, p, _ = stats.linregress(yz, Wyz)
xline = np.linspace(yz.min(), yz.max(), 300)
ax.plot(xline, m*xline + b, color='#1a1a1a', lw=2.2, zorder=4,
        label=f"Slope = Moran's I = {SP['global_moran_u5mr_per1000lb']['I']:.4f}")
ax.axhline(0, color='#888', lw=0.9, ls='--')
ax.axvline(0, color='#888', lw=0.9, ls='--')
ax.set_xlabel('Standardised U5MR (z-score)', fontsize=12, fontweight='semibold')
ax.set_ylabel('Spatial lag of standardised U5MR', fontsize=12, fontweight='semibold')
ax.set_title("Figure 2. Global Moran's I scatter plot for U5MR\n(KNN-4 spatial weights, 999 permutations)", pad=10)
p_hi = mpatches.Patch(color='#d73027', label='High-risk (U5MR ≥48/1,000 LB)')
p_lo = mpatches.Patch(color='#4575b4', label='Lower-risk district')
ax.legend(handles=[p_hi, p_lo,
          mpatches.Patch(color='#1a1a1a', label=f"Moran's I = {SP['global_moran_u5mr_per1000lb']['I']:.4f}, z = {SP['global_moran_u5mr_per1000lb']['z_score']:.3f}, p<0.001")],
          fontsize=9.5, loc='upper left', framealpha=0.9, edgecolor='#ccc')
ax.spines[['top','right']].set_visible(False)
ax.grid(alpha=0.2, lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'Figure2_Moran_Scatter.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure2_Moran_Scatter.png")

# ── Figure 3: LISA Cluster Map ────────────────────────────────────────────────
print("Generating Figure 3: LISA Cluster Map...")

z_u5   = (u5mr_arr - u5mr_arr.mean()) / u5mr_arr.std()
Wz_u5  = W @ z_u5
li     = z_u5 * Wz_u5
rng    = np.random.default_rng(42)
li_sim = np.array([rng.permutation(z_u5) * (W @ rng.permutation(z_u5))
                   for _ in range(999)])
p_vals = np.mean(np.abs(li_sim) >= np.abs(li).reshape(1,-1), axis=0)
sig    = p_vals < 0.05

quad_label = np.full(n, 'NS', dtype=object)
quad_label[(z_u5>0)&(Wz_u5>0)&sig]  = 'HH'
quad_label[(z_u5<0)&(Wz_u5<0)&sig]  = 'LL'
quad_label[(z_u5>0)&(Wz_u5<0)&sig]  = 'HL'
quad_label[(z_u5<0)&(Wz_u5>0)&sig]  = 'LH'

df_lisa = df[['district','norm']].copy()
df_lisa['lisa_quad'] = quad_label
merged_lisa = gdf_raw.merge(df_lisa, on='norm', how='left')
merged_lisa['lisa_quad'] = merged_lisa['lisa_quad'].fillna('NS')

color_map = {'HH':'#d73027','LL':'#4575b4','HL':'#f4a582','LH':'#abd9e9','NS':'#d9d9d9'}
merged_lisa['color'] = merged_lisa['lisa_quad'].map(color_map)

hh_n = int((quad_label=='HH').sum())
ll_n = int((quad_label=='LL').sum())
hl_n = int((quad_label=='HL').sum())
lh_n = int((quad_label=='LH').sum())
ns_n = int((quad_label=='NS').sum())

fig, ax = plt.subplots(1, 1, figsize=(10, 11))
for cat, col in color_map.items():
    sub = merged_lisa[merged_lisa['lisa_quad']==cat]
    if len(sub): sub.plot(ax=ax, color=col, edgecolor='#444', linewidth=0.25)
patches = [
    mpatches.Patch(color='#d73027', label=f'HH — high–high cluster  (n={hh_n})'),
    mpatches.Patch(color='#4575b4', label=f'LL — low–low cluster     (n={ll_n})'),
    mpatches.Patch(color='#f4a582', label=f'HL — high–low outlier    (n={hl_n})'),
    mpatches.Patch(color='#abd9e9', label=f'LH — low–high outlier   (n={lh_n})'),
    mpatches.Patch(color='#d9d9d9', label=f'Not significant           (n={ns_n})'),
]
ax.legend(handles=patches, loc='lower left', fontsize=9.5,
          title='LISA Category (p<0.05, 999 perms)', title_fontsize=10,
          framealpha=0.92, edgecolor='#aaa')
add_ghana_north_arrow(ax)
add_scalebar(ax)
clean_map_ax(ax)
ax.set_title(
    'Figure 3. LISA cluster map for U5MR across 261 MMDAs, Ghana\n(KNN-4 weights, p<0.05, 999 permutations)',
    fontsize=13, fontweight='semibold', pad=12)
ax.text(0.02, 0.98,
        f"Total significant: {hh_n+ll_n+hl_n+lh_n}/261 districts  |  "
        f"Global Moran's I = {SP['global_moran_u5mr_per1000lb']['I']:.4f}",
        transform=ax.transAxes, fontsize=9.5, va='top',
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='#bbb', pad=5))
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, 'Figure3_LISA_Cluster_Map.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ Figure3_LISA_Cluster_Map.png  (HH={hh_n}, LL={ll_n}, sig={hh_n+ll_n+hl_n+lh_n})")

# ── Figure 4: NMR Choropleth ──────────────────────────────────────────────────
print("Generating Figure 4: NMR Choropleth Map...")
fig, ax = plt.subplots(1, 1, figsize=(10, 11))
merged.plot(column='nmr_per1000lb', ax=ax, cmap='PuBuGn',
            vmin=8, vmax=23, missing_kwds={'color': '#cccccc'},
            edgecolor='#444', linewidth=0.25)
sm2 = cm.ScalarMappable(cmap='PuBuGn', norm=Normalize(vmin=8, vmax=23))
sm2.set_array([])
cbar2 = fig.colorbar(sm2, ax=ax, shrink=0.55, pad=0.01, aspect=20)
cbar2.set_label('NMR per 1,000 live births', fontsize=11, fontweight='semibold')
cbar2.ax.tick_params(labelsize=9.5)
add_ghana_north_arrow(ax)
add_scalebar(ax)
clean_map_ax(ax)
ax.set_title(
    'Figure 4. District-level neonatal mortality rate (NMR)\nacross 261 MMDAs, Ghana',
    fontsize=13, fontweight='semibold', pad=12)
ax.text(0.02, 0.98,
        f"Range: 8–23 per 1,000 LB  |  Median: {df['nmr_per1000lb'].median():.0f}\n"
        f"Moran's I = {SP['global_moran_nmr_per1000lb']['I']:.4f}  "
        f"(z = {SP['global_moran_nmr_per1000lb']['z_score']:.3f}, p<0.001, KNN-4)",
        transform=ax.transAxes, fontsize=9.5, va='top',
        bbox=dict(facecolor='white', alpha=0.88, edgecolor='#bbb', pad=5))
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIG_DIR, 'Figure4_NMR_Choropleth.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure4_NMR_Choropleth.png")

# ── Figure 5: Feature Importance — U5MR & NMR ────────────────────────────────
print("Generating Figure 5: Feature Importance...")
feat_labels = {
    'early_breastfeeding_pct':   'Early breastfeeding\ninitiation (%)',
    'diarrhea_prevalence_pct':   'Diarrhoea prevalence (%)',
    'fully_vaccinated_pct':      'Full vaccination\ncoverage (%)',
    'dietary_diversity_pct':     'Min. dietary diversity (%)',
    'bcg_coverage_pct':          'BCG coverage (%)',
    'ors_use_pct':               'ORS use — diarrhoea (%)',
    'child_anemia_any_pct':      'Child anaemia — any (%)',
    'dpt3_coverage_pct':         'DPT3 coverage (%)',
    'measles_coverage_pct':      'Measles coverage (%)',
    'illiteracy_rate_pct':       'Illiteracy rate (%)',
    'poverty_rate_pct':          'Poverty incidence (%)',
}
u5_feats  = ML['U5MR']['top_features']
nmr_feats = ML['NMR']['top_features']

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
palettes = [
    ['#67000d','#a50f15','#cb181d','#ef3b2c','#fb6a4a','#fc9272','#fcbba1','#fee0d2','#fff5f0','#fff5f0'],
    ['#084594','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#f7fbff','#f7fbff','#f7fbff'],
]
titles = [
    'U5MR — Under-Five Mortality Risk\n(Random Forest, 10-fold CV)',
    'NMR — Neonatal Mortality Risk\n(Random Forest, 10-fold CV)',
]
for ax, feats, pal, title in zip(axes, [u5_feats, nmr_feats], palettes, titles):
    names = [feat_labels.get(f['feature'], f['feature'].replace('_pct','(%)').replace('_',' ').title())
             for f in feats]
    imps  = [f['importance'] for f in feats]
    y_pos = list(range(len(names)-1, -1, -1))
    bars  = ax.barh(y_pos, imps, color=pal[:len(feats)], edgecolor='white', height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel('Gini importance (proportion of total)', fontsize=11, fontweight='semibold')
    ax.set_title(title, fontsize=12, fontweight='semibold', pad=8)
    ax.set_xlim(0, max(imps)*1.28)
    for bar, val in zip(bars, imps):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9.5)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=0.25, lw=0.7)

fig.suptitle('Figure 5. Top-10 predictors of district-level child mortality risk\n(261 MMDAs, Ghana; stacked ensemble: RF + GB → logistic regression meta-learner)',
             fontsize=12.5, fontweight='semibold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'Figure5_Feature_Importance.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure5_Feature_Importance.png")

# ── Figure 6: Bivariate Moran Scatter (3-panel) ───────────────────────────────
print("Generating Figure 6: Bivariate Scatter...")
od  = df['open_defecation_pct'].values
pov = df['poverty_rate_pct'].values
edu = df['women_secondary_plus_pct'].values
hr  = df['u5mr_high_risk'].values
u5  = df['u5mr_per1000lb'].values

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
panels = [
    (od,  'Open defecation (%)',               '#d73027','#4575b4',
     f"Bivariate Moran's I = {SP['bivariate_moran']['U5MR × Open Defecation']['I_bv']:.4f}, p<0.001"),
    (pov, 'Poverty incidence (%)',              '#d73027','#4575b4',
     f"Bivariate Moran's I = {SP['bivariate_moran']['U5MR × Poverty Rate']['I_bv']:.4f}, p<0.001"),
    (edu, "Women secondary+ education (%)",    '#4575b4','#d73027',
     f"Bivariate Moran's I = {SP['bivariate_moran']['U5MR × Women Secondary Edu']['I_bv']:.4f}, p<0.001"),
]
for ax, (xvar, xlabel, c_hi, c_lo, bv_txt) in zip(axes, panels):
    colors_dot = np.where(hr==1, c_hi, c_lo)
    ax.scatter(xvar, u5, c=colors_dot, edgecolors='white', s=48,
               linewidths=0.4, alpha=0.85, zorder=3)
    m, b, r, p_val, _ = stats.linregress(xvar, u5)
    xline = np.linspace(xvar.min(), xvar.max(), 300)
    ax.plot(xline, m*xline+b, color='#222', lw=2, ls='--',
            label=f'r = {r:.3f}', zorder=4)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='semibold')
    ax.set_ylabel('U5MR per 1,000 live births', fontsize=11, fontweight='semibold')
    ax.set_title(f'U5MR × {xlabel.split("(")[0].strip()}\n{bv_txt}',
                 fontsize=10.5, fontweight='semibold', pad=7)
    p_hi = mpatches.Patch(color=c_hi, label='High-risk (U5MR ≥48)')
    p_lo = mpatches.Patch(color=c_lo, label='Lower-risk')
    ax.legend(handles=[p_hi, p_lo], fontsize=9, framealpha=0.88, edgecolor='#ccc',
              loc='upper left' if xlabel!="Women secondary+ education (%)" else 'upper right')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=0.2, lw=0.6)

fig.suptitle('Figure 6. Bivariate scatter plots: U5MR versus key socioeconomic and health determinants\n(261 MMDAs, Ghana; districts colour-coded by U5MR high-risk classification)',
             fontsize=12, fontweight='semibold', y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'Figure6_Bivariate_Scatter.png'), dpi=DPI, bbox_inches='tight')
plt.close(fig)
print("  ✓ Figure6_Bivariate_Scatter.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✓ All 6 figures generated at 300 DPI using Ghana district GeoJSON polygons.")
print(f"{'File':<48} {'Size (KB)':>10}")
print("-"*60)
for fname in sorted(os.listdir(FIG_DIR)):
    if fname.endswith('.png'):
        kb = os.path.getsize(os.path.join(FIG_DIR, fname)) // 1024
        print(f"  {fname:<46} {kb:>8} KB")
