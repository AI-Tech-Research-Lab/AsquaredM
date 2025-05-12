"""
Two-sample Kolmogorov–Smirnov test (Python / SciPy)
---------------------------------------------------

Copy–paste this script into your analysis notebook or a .py file.
Replace the placeholder arrays `sample_a` and `sample_b` with 
your actual data vectors.

When you run it, you’ll get a one-liner ready to drop into the
Methods/Results section of your paper.
"""

import numpy as np
from scipy import stats

# -------------------------------------------------------------------
# 1) LOAD OR DEFINE YOUR TWO SAMPLES
# -------------------------------------------------------------------
# -- Example dummy data (delete these two lines and load your own) --
sample_a = np.random.lognormal(mean=0.0, sigma=1.0, size=120)
sample_b = np.random.lognormal(mean=0.5, sigma=1.0, size=135)
# -------------------------------------------------------------------

# 2) RUN THE *TWO-SIDED* KS TEST
#    mode='auto' chooses the exact test when both n ≤ 10 000,
#    otherwise the well-known asymptotic approximation.
#
stat_result = stats.ks_2samp(
    sample_a,
    sample_b,
    alternative="two-sided",
    mode="auto"
)

D_stat   = stat_result.statistic       # KS distance
p_value  = stat_result.pvalue          # (exact or asymptotic) p-value
n_a      = len(sample_a)
n_b      = len(sample_b)

# 3) FORMAT A SENTENCE FOR THE PAPER AND PRINT IT
sentence = (
    "Kolmogorov–Smirnov two-sample test indicated that the two "
    f"distributions differ (D = {D_stat:.3f}, n₁ = {n_a}, n₂ = {n_b}, "
    f"p = {p_value:.4g}, two-sided)."
)

print(sentence)