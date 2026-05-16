# analysis.R — Ghana Child Mortality 261 Districts
# Spatial regression (SLM/SEM) + negative binomial GLM
# Author: Valentine Golden Ghanem | ORCID: 0009-0002-8332-0220
# Usage: Rscript analysis.R
suppressPackageStartupMessages({
  library(spdep)
  library(spatialreg)
  library(MASS)      # glm.nb
  library(ggplot2)
  library(dplyr)
  library(readr)
})
set.seed(42)

cat("── Loading data ──────────────────────────────────────────────────────\n")
df <- read_csv("data/Ghana_ChildMortality_261District_MasterDataset.csv",
               show_col_types = FALSE)
cat(sprintf("Loaded: %d districts × %d variables\n", nrow(df), ncol(df)))

# ── 1. Spatial weights (KNN k=4) ─────────────────────────────────────────────
cat("\n── Spatial weights (KNN k=4) ─────────────────────────────────────────\n")
coords <- cbind(df$longitude, df$latitude)
knn4   <- knearneigh(coords, k = 4)
nb4    <- knn2nb(knn4)
W      <- nb2listw(nb4, style = "W")
cat("Spatial weights object created.\n")

# ── 2. Global Moran's I (U5MR) ───────────────────────────────────────────────
cat("\n── Global Moran's I (U5MR) ───────────────────────────────────────────\n")
mi <- moran.test(df$u5mr_per1000lb, W, randomisation = TRUE)
cat(sprintf("  Moran I = %.4f  (z = %.3f, p = %.4f)\n",
            mi$estimate[1], mi$statistic, mi$p.value))

mi_nmr <- moran.test(df$nmr_per1000lb, W, randomisation = TRUE)
cat(sprintf("  NMR Moran I = %.4f  (z = %.3f, p = %.4f)\n",
            mi_nmr$estimate[1], mi_nmr$statistic, mi_nmr$p.value))

# ── 3. LM diagnostics for model selection ────────────────────────────────────
cat("\n── LM diagnostics ────────────────────────────────────────────────────\n")
formula_ols <- u5mr_per1000lb ~ early_breastfeeding_pct + diarrhea_prevalence_pct +
               fully_vaccinated_pct + poverty_rate_pct + improved_water_pct +
               illiteracy_rate_pct
ols <- lm(formula_ols, data = df)
lm_tests <- lm.RStests(ols, W, test = "all")
print(lm_tests)

# ── 4. Spatial Lag Model (SLM) ───────────────────────────────────────────────
cat("\n── Spatial Lag Model (SLM) ───────────────────────────────────────────\n")
slm <- lagsarlm(formula_ols, data = df, listw = W)
cat(summary(slm)$Coef |> as.data.frame() |> format(digits = 4))
cat(sprintf("\n  rho = %.4f (p = %.4f)  Log-Lik = %.2f  AIC = %.2f\n",
            slm$rho, slm$rho.se, logLik(slm), AIC(slm)))

# ── 5. Spatial Error Model (SEM) ─────────────────────────────────────────────
cat("\n── Spatial Error Model (SEM) ─────────────────────────────────────────\n")
sem <- errorsarlm(formula_ols, data = df, listw = W)
cat(sprintf("  lambda = %.4f  AIC = %.2f\n", sem$lambda, AIC(sem)))

# ── 6. Negative Binomial GLM (district-level count analogue) ─────────────────
cat("\n── Negative Binomial GLM ─────────────────────────────────────────────\n")
nb_mod <- glm.nb(round(u5mr_per1000lb) ~ early_breastfeeding_pct +
                   diarrhea_prevalence_pct + fully_vaccinated_pct +
                   poverty_rate_pct + improved_water_pct, data = df)
cat(sprintf("  AIC = %.2f  Theta = %.4f\n", AIC(nb_mod), nb_mod$theta))
print(coef(summary(nb_mod)))

# ── 7. Model comparison ───────────────────────────────────────────────────────
cat("\n── Model AIC comparison ──────────────────────────────────────────────\n")
comparison <- data.frame(
  Model = c("OLS", "SLM", "SEM"),
  AIC   = round(c(AIC(ols), AIC(slm), AIC(sem)), 2)
)
print(comparison)
cat("\nSpatial regression complete.\n")
