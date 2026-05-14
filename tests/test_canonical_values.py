"""
test_canonical_values.py
Pytest canonical assertions — Child Mortality Ghana 261 Districts
Author: Valentine Golden Ghanem | AIPOCH v6.0 | 2026-05-13
Run: pytest tests/ -v
EX-024 pattern: ≥10 canonical assertions before QA.
"""

import os
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'Ghana_ChildMortality_261District_MasterDataset.csv')


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(DATA_PATH)


# ── Dataset structure ─────────────────────────────────────────────────────────
def test_district_count(df):
    """Ghana has 261 MMDAs per 2019 demarcation."""
    assert len(df) == 261, f"Expected 261 districts, got {len(df)}"


def test_region_count(df):
    """Ghana has 16 administrative regions (post-2022 restructuring)."""
    assert df['region'].nunique() == 16, f"Expected 16 regions, got {df['region'].nunique()}"


def test_column_count(df):
    """Master dataset must have at least 44 columns."""
    assert df.shape[1] >= 44, f"Expected ≥44 columns, got {df.shape[1]}"


# ── Mortality outcomes — canonical values ─────────────────────────────────────
def test_u5mr_range(df):
    """U5MR must span 20–72 per 1,000 LB (Greater Accra–Oti gradient)."""
    assert df['u5mr_per1000lb'].min() == pytest.approx(20.0, abs=0.5)
    assert df['u5mr_per1000lb'].max() == pytest.approx(72.0, abs=0.5)


def test_nmr_range(df):
    """NMR must span 8–23 per 1,000 LB."""
    assert df['nmr_per1000lb'].min() == pytest.approx(8.0, abs=0.5)
    assert df['nmr_per1000lb'].max() == pytest.approx(23.0, abs=0.5)


def test_imr_range(df):
    """IMR must span 16–46 per 1,000 LB."""
    assert df['imr_per1000lb'].min() == pytest.approx(16.0, abs=0.5)
    assert df['imr_per1000lb'].max() == pytest.approx(46.0, abs=0.5)


def test_u5mr_median(df):
    """U5MR median must be approximately 45.0 per 1,000 LB."""
    assert df['u5mr_per1000lb'].median() == pytest.approx(45.0, abs=1.0)


def test_nmr_median(df):
    """NMR median must be approximately 17.0 per 1,000 LB."""
    assert df['nmr_per1000lb'].median() == pytest.approx(17.0, abs=1.0)


# ── High-risk classification ──────────────────────────────────────────────────
def test_u5mr_high_risk_count(df):
    """U5MR top-quartile high-risk: 72 districts (27.6%)."""
    assert df['u5mr_high_risk'].sum() == 72, f"Expected 72 high-risk districts, got {df['u5mr_high_risk'].sum()}"


def test_high_risk_threshold(df):
    """U5MR high-risk threshold must be ≥48 per 1,000 LB (top quartile)."""
    threshold = df['u5mr_per1000lb'].quantile(0.75)
    assert threshold == pytest.approx(48.0, abs=1.0), f"Threshold: {threshold}"


def test_no_missing_mortality(df):
    """Zero missing values for all three primary mortality outcomes."""
    for col in ['nmr_per1000lb', 'imr_per1000lb', 'u5mr_per1000lb']:
        n_missing = df[col].isna().sum()
        assert n_missing == 0, f"{col} has {n_missing} missing values"


# ── Population ────────────────────────────────────────────────────────────────
def test_total_population(df):
    """Total population across 261 districts must be ~30.8M (Ghana Census 2021)."""
    total = df['total_pop'].sum()
    assert 29_000_000 < total < 33_000_000, f"Population {total:,.0f} out of expected range"


def test_population_0_14(df):
    """Children 0–14 proxy population must be ~10.9M."""
    total_014 = df['pop_0_14'].sum()
    assert 9_000_000 < total_014 < 13_000_000, f"Pop 0-14: {total_014:,.0f}"


# ── Regional extremes ─────────────────────────────────────────────────────────
def test_highest_u5mr_region(df):
    """Oti region must have the highest U5MR (72.0 per 1,000 LB)."""
    reg_u5 = df.groupby('region')['u5mr_per1000lb'].first()
    assert reg_u5.idxmax() == 'Oti', f"Highest U5MR region: {reg_u5.idxmax()}"


def test_lowest_u5mr_region(df):
    """Greater Accra must have the lowest U5MR (20.0 per 1,000 LB)."""
    reg_u5 = df.groupby('region')['u5mr_per1000lb'].first()
    assert reg_u5.idxmin() == 'Greater Accra', f"Lowest U5MR region: {reg_u5.idxmin()}"


def test_north_south_gradient(df):
    """U5MR gradient: Oti/Savannah > Greater Accra by ≥3×."""
    reg_u5 = df.groupby('region')['u5mr_per1000lb'].first()
    ratio = reg_u5['Oti'] / reg_u5['Greater Accra']
    assert ratio >= 3.0, f"North-South gradient ratio: {ratio:.2f}"


# ── Predictor ranges ──────────────────────────────────────────────────────────
def test_bcg_coverage_high(df):
    """BCG coverage national median must be ≥90% (high-performing programme)."""
    assert df['bcg_coverage_pct'].median() >= 90.0


def test_open_defecation_variation(df):
    """Open defecation must show large inter-district variation (range >50pp)."""
    od_range = df['open_defecation_pct'].max() - df['open_defecation_pct'].min()
    assert od_range > 50.0, f"OD range: {od_range:.1f}pp"


def test_binary_outcome_valid(df):
    """u5mr_high_risk must be binary (0/1 only)."""
    assert set(df['u5mr_high_risk'].unique()).issubset({0, 1})
    assert set(df['nmr_high_risk'].unique()).issubset({0, 1})


def test_coordinates_valid(df):
    """Latitude must be within Ghana bounds (4.5°N–11.2°N)."""
    lats = df['latitude'].dropna()
    assert lats.between(4.0, 12.0).all(), f"Lat range: {lats.min():.2f}–{lats.max():.2f}"


def test_poverty_plausible(df):
    """Poverty incidence must be bounded 0–100%."""
    pov = df['poverty_rate_pct'].dropna()
    assert pov.between(0, 100).all()


def test_ashanti_district_count(df):
    """Ashanti region must contain 43 districts (largest region)."""
    n = (df['region'] == 'Ashanti').sum()
    assert n == 43, f"Ashanti districts: {n}"
