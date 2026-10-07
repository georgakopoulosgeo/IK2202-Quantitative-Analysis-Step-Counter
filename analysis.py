# II2202 Quantitative exercise - iPhone step counter apps
# Run: python analysis.py  (step-counter-survey.csv must be in the same folder)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(1)
BLUE = "#0072B2"     # colorblind friendly colors
ORANGE = "#D55E00"

# 1. Load data
df = pd.read_csv("step-counter-survey.csv")
print(df.shape)
print(df.isna().sum())

# 2. Preprocessing
df["guide"] = np.where(df["listed_in"].str.contains("cute"), "cute", "simple")
df["widget"] = df["widget_per_description"].map({"yes": 1, "no": 0})
df["log_ratings"] = np.log10(df["ratings_count"] + 1)    # +1 because some apps have 0 ratings
df["age_rank"] = df["app_store_id"].rank()               # smaller id = older app

# 3. Descriptive statistics
print(df[["ratings_count", "log_ratings", "rating_average"]].describe())
print("skewness ratings:", round(df["ratings_count"].skew(), 2))
print("skewness log ratings:", round(df["log_ratings"].skew(), 2))

# 4. H1: older apps have more ratings
rho, p = stats.spearmanr(df["age_rank"], df["ratings_count"])
print(f"H1 Spearman: rho = {rho:.2f}, p = {p:.2g}")
fit = stats.linregress(df["age_rank"], df["log_ratings"])
print(f"Linear fit: slope = {fit.slope:.2f}, R^2 = {fit.rvalue ** 2:.2f}, p = {fit.pvalue:.2g}")

# 5. H2: simple apps mention widgets more often than cute apps
table = pd.crosstab(df["guide"], df["widget"])
print(table)
print(f"H2 Fisher exact: p = {stats.fisher_exact(table)[1]:.2f}")

# 6. H3: no difference in ratings between the two guides
simple = df[df["guide"] == "simple"]["ratings_count"]
cute = df[df["guide"] == "cute"]["ratings_count"]
print(f"H3 medians: simple {simple.median():.0f}, cute {cute.median():.0f}")
print(f"H3 Mann-Whitney: p = {stats.mannwhitneyu(simple, cute)[1]:.2f}")

# 7. How much data do we need? Simulate the widget test with more apps
p_simple = df[df["guide"] == "simple"]["widget"].mean()
p_cute = df[df["guide"] == "cute"]["widget"].mean()
sizes = [5, 10, 15, 20, 25, 30, 40]
power = []
for n in sizes:
    hits = 0
    for i in range(1000):
        a = np.random.binomial(n, p_simple)
        b = np.random.binomial(n, p_cute)
        if stats.fisher_exact([[a, n - a], [b, n - b]])[1] < 0.05:
            hits += 1
    power.append(hits / 10)   # in percent
    print(f"{n} apps per group -> power {hits / 10:.0f}%")

# 8. Figures
# Figure 1: distribution of ratings
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(df["ratings_count"] / 1000, bins=12, color="lightgray", edgecolor="black")
ax[0].set_xlabel("ratings (thousands)")
ax[0].set_ylabel("number of apps")
ax[1].hist(df["log_ratings"], bins=10, color="lightgray", edgecolor="black")
ax[1].set_xlabel("log10(ratings + 1)")
plt.tight_layout()
plt.savefig("fig1_distribution.png", dpi=300)

# Figure 2: age vs ratings
plt.figure(figsize=(6, 3.8))
for g, color, marker in [("simple", BLUE, "o"), ("cute", ORANGE, "^")]:
    part = df[df["guide"] == g]
    plt.scatter(part["age_rank"], part["log_ratings"], color=color, marker=marker, edgecolor="black", label=g)
x = np.arange(1, 20)
plt.plot(x, fit.intercept + fit.slope * x, "k--", label="linear fit")
plt.xlabel("age rank (1 = oldest app)")
plt.ylabel("log10(ratings + 1)")
plt.legend()
plt.savefig("fig2_age_vs_ratings.png", dpi=300, bbox_inches="tight")

# Figure 3: widget mentions by guide
plt.figure(figsize=(4.5, 3.5))
share = [p_simple * 100, p_cute * 100]
plt.bar("simple", share[0], color=BLUE, edgecolor="black")
plt.bar("cute", share[1], color=ORANGE, hatch="///", edgecolor="black")
plt.ylabel("% of apps mentioning a widget")
plt.savefig("fig3_widgets.png", dpi=300, bbox_inches="tight")

# Figure 4: power vs number of apps
plt.figure(figsize=(6, 3.5))
plt.plot(sizes, power, "o-", color=BLUE)
plt.axhline(80, color="gray", linestyle="--", label="80% power")
plt.axvline(10, color="gray", linestyle=":", label="our data")
plt.xlabel("apps per group")
plt.ylabel("power (%)")
plt.legend()
plt.savefig("fig4_power.png", dpi=300, bbox_inches="tight")
print("done")