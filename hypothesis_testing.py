import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, norm, chi2

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('data/data.csv')

# Create separate Data Frames for working and non-working students
working = df[df['has_work'] == True]['gpa'].dropna()
not_working = df[df['has_work'] == False]['gpa'].dropna()

print("="*90)
print(" " * 20 + "HYPOTHESIS TESTING & CONFIDENCE INTERVALS")
print("="*90)

# 2. Test hypothesis and construct CI for the mean of a population
print("\n" + "="*90)
print("REQUIREMENT 2: HYPOTHESIS TEST & CI FOR POPULATION MEAN (ALL STUDENTS)")
print("="*90)

# Population: All students' GPA
all_gpa = df['gpa'].dropna() #Get all the gpa of students
n_all = len(all_gpa) #Get n
mean_all = all_gpa.mean() #Get mean using .mean()
std_all = all_gpa.std(ddof=1)  # Sample standard deviation
se_all = std_all / np.sqrt(n_all)

print(f"\nSample Size: {n_all}")
print(f"Sample Mean: {mean_all:.4f}")
print(f"Sample Std Dev: {std_all:.4f}")
print(f"Standard Error: {se_all:.4f}")

# Hypothesis Test: H0: μ = 3.0 vs H1: μ ≠ 3.0 (Two sided)
mu_0 = 3.0
alpha = 0.05

t_stat = (mean_all - mu_0) / se_all
df_test = n_all - 1
p_value = 2 * (1 - t.cdf(abs(t_stat), df_test))
t_critical = t.ppf(1 - alpha/2, df_test)

print(f"\n--- Hypothesis Test ---")
print(f"H0: μ = {mu_0} (Population mean GPA is {mu_0})")
print(f"H1: μ ≠ {mu_0} (Population mean GPA is not {mu_0})")
print(f"Significance Level (α): {alpha}")
print(f"\nTest Statistic (t): {t_stat:.4f}")
print(f"Degrees of Freedom: {df_test}")
print(f"P-value: {p_value:.4f}")
print(f"Critical Value (±): {t_critical:.4f}")

if p_value < alpha:
    print(f"\nREJECT H0 (p-value {p_value:.4f} < {alpha})")
    print(f"Conclusion: There is significant evidence that the mean GPA is different from {mu_0}")
else:
    print(f"\nFAIL TO REJECT H0 (p-value {p_value:.4f} >= {alpha})")
    print(f"Conclusion: There is not enough evidence that the mean GPA is different from {mu_0}")

# Confidence Interval
ci_lower = mean_all - t_critical * se_all
ci_upper = mean_all + t_critical * se_all

print(f"\n--- {int((1-alpha)*100)}% Confidence Interval for Mean GPA ---")
print(f"CI: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"Interpretation: We are {int((1-alpha)*100)}% confident that the true population mean GPA")
print(f"                lies between {ci_lower:.4f} and {ci_upper:.4f}")

# 3. Test hypothesis and construct CI for proportion of a population
print("\n" + "="*90)
print("REQUIREMENT 3: HYPOTHESIS TEST & CI FOR POPULATION PROPORTION")
print("="*90)

# Proportion of students with GPA >= 3.0
success_count = (df['gpa'] >= 3.0).sum()
n_total = len(df['gpa'].dropna())
p_hat = success_count / n_total

print(f"\nProportion Analysis: Students with GPA ≥ 3.0")
print(f"Sample Size: {n_total}")
print(f"Number with GPA ≥ 3.0: {success_count}")
print(f"Sample Proportion (p̂): {p_hat:.4f}")

# Hypothesis Test: H0: p = 0.5 vs H1: p ≠ 0.5
p_0 = 0.5
se_p = np.sqrt(p_0 * (1 - p_0) / n_total)

z_stat = (p_hat - p_0) / se_p
p_value_prop = 2 * (1 - norm.cdf(abs(z_stat)))
z_critical = norm.ppf(1 - alpha/2)

print(f"\n--- Hypothesis Test ---")
print(f"H0: p = {p_0} (50% of students have GPA ≥ 3.0)")
print(f"H1: p ≠ {p_0} (Proportion is different from 50%)")
print(f"Significance Level (α): {alpha}")
print(f"\nTest Statistic (z): {z_stat:.4f}")
print(f"P-value: {p_value_prop:.4f}")
print(f"Critical Value (±): {z_critical:.4f}")

if p_value_prop < alpha:
    print(f"\nREJECT H0 (p-value {p_value_prop:.4f} < {alpha})")
    print(f"Conclusion: The proportion of students with GPA ≥ 3.0 is significantly different from 50%")
else:
    print(f"\nFAIL TO REJECT H0 (p-value {p_value_prop:.4f} >= {alpha})")
    print(f"Conclusion: Not enough evidence that proportion differs from 50%")

# Confidence Interval for proportion
se_p_hat = np.sqrt(p_hat * (1 - p_hat) / n_total)
ci_lower_p = p_hat - z_critical * se_p_hat
ci_upper_p = p_hat + z_critical * se_p_hat

print(f"\n--- {int((1-alpha)*100)}% Confidence Interval for Proportion ---")
print(f"CI: ({ci_lower_p:.4f}, {ci_upper_p:.4f})")
print(f"Interpretation: We are {int((1-alpha)*100)}% confident that the true proportion of students")
print(f"                with GPA ≥ 3.0 lies between {ci_lower_p:.4f} and {ci_upper_p:.4f}")

# 4. Test hypothesis and construct CI for difference in means (2 populations)
print("\n" + "="*90)
print("REQUIREMENT 4: HYPOTHESIS TEST & CI FOR DIFFERENCE IN MEANS")
print("="*90)
print("Comparing: Working Students vs Non-Working Students")

n1 = len(working)
n2 = len(not_working)
mean1 = working.mean()
mean2 = not_working.mean()
std1 = working.std(ddof=1)
std2 = not_working.std(ddof=1)

print(f"\n--- Group 1: Working Students ---")
print(f"Sample Size (n1): {n1}")
print(f"Mean GPA (x̄1): {mean1:.4f}")
print(f"Std Dev (s1): {std1:.4f}")

print(f"\n--- Group 2: Non-Working Students ---")
print(f"Sample Size (n2): {n2}")
print(f"Mean GPA (x̄2): {mean2:.4f}")
print(f"Std Dev (s2): {std2:.4f}")

# Pooled variance (assuming equal variances)
pooled_var = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
pooled_std = np.sqrt(pooled_var)
se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)

print(f"\n--- Pooled Statistics ---")
print(f"Pooled Variance: {pooled_var:.4f}")
print(f"Pooled Std Dev: {pooled_std:.4f}")
print(f"Standard Error of Difference: {se_diff:.4f}")

# Hypothesis Test: H0: μ1 = μ2 vs H1: μ1 ≠ μ2
diff_means = mean1 - mean2
t_stat_diff = diff_means / se_diff
df_diff = n1 + n2 - 2
p_value_diff = 2 * (1 - t.cdf(abs(t_stat_diff), df_diff))
t_critical_diff = t.ppf(1 - alpha/2, df_diff)

print(f"\n--- Hypothesis Test ---")
print(f"H0: μ1 - μ2 = 0 (No difference in mean GPA)")
print(f"H1: μ1 - μ2 ≠ 0 (There is a difference in mean GPA)")
print(f"Significance Level (α): {alpha}")
print(f"\nDifference in Means (x̄1 - x̄2): {diff_means:.4f}")
print(f"Test Statistic (t): {t_stat_diff:.4f}")
print(f"Degrees of Freedom: {df_diff}")
print(f"P-value: {p_value_diff:.4f}")
print(f"Critical Value (±): {t_critical_diff:.4f}")

if p_value_diff < alpha:
    print(f"\nREJECT H0 (p-value {p_value_diff:.4f} < {alpha})")
    print(f"Conclusion: There IS a significant difference in mean GPA between working and non-working students")
else:
    print(f"\nFAIL TO REJECT H0 (p-value {p_value_diff:.4f} >= {alpha})")
    print(f"Conclusion: There is NO significant difference in mean GPA between the two groups")

# Confidence Interval for difference
ci_lower_diff = diff_means - t_critical_diff * se_diff
ci_upper_diff = diff_means + t_critical_diff * se_diff

print(f"\n--- {int((1-alpha)*100)}% Confidence Interval for Difference in Means ---")
print(f"CI: ({ci_lower_diff:.4f}, {ci_upper_diff:.4f})")
print(f"Interpretation: We are {int((1-alpha)*100)}% confident that the true difference in mean GPA")
print(f"                (Working - Non-Working) lies between {ci_lower_diff:.4f} and {ci_upper_diff:.4f}")

# 5. Test hypothesis and construct CI for difference in proportions
print("\n" + "="*90)
print("REQUIREMENT 5: HYPOTHESIS TEST & CI FOR DIFFERENCE IN PROPORTIONS")
print("="*90)
print("Comparing: Proportion with GPA ≥ 3.0 (Working vs Non-Working)")

# Working students
working_high_gpa = (df[df['has_work'] == True]['gpa'] >= 3.0).sum()
n_working = len(df[df['has_work'] == True]['gpa'].dropna())
p1_hat = working_high_gpa / n_working

# Non-working students
not_working_high_gpa = (df[df['has_work'] == False]['gpa'] >= 3.0).sum()
n_not_working = len(df[df['has_work'] == False]['gpa'].dropna())
p2_hat = not_working_high_gpa / n_not_working

print(f"\n--- Group 1: Working Students ---")
print(f"Sample Size: {n_working}")
print(f"Number with GPA ≥ 3.0: {working_high_gpa}")
print(f"Proportion (p̂1): {p1_hat:.4f}")

print(f"\n--- Group 2: Non-Working Students ---")
print(f"Sample Size: {n_not_working}")
print(f"Number with GPA ≥ 3.0: {not_working_high_gpa}")
print(f"Proportion (p̂2): {p2_hat:.4f}")

# Pooled proportion
p_pooled = (working_high_gpa + not_working_high_gpa) / (n_working + n_not_working)
se_diff_prop = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_working + 1/n_not_working))

print(f"\n--- Pooled Statistics ---")
print(f"Pooled Proportion: {p_pooled:.4f}")
print(f"Standard Error: {se_diff_prop:.4f}")

# Hypothesis Test: H0: p1 = p2 vs H1: p1 ≠ p2
diff_prop = p1_hat - p2_hat
z_stat_prop = diff_prop / se_diff_prop
p_value_diff_prop = 2 * (1 - norm.cdf(abs(z_stat_prop)))

print(f"\n--- Hypothesis Test ---")
print(f"H0: p1 - p2 = 0 (No difference in proportions)")
print(f"H1: p1 - p2 ≠ 0 (There is a difference in proportions)")
print(f"Significance Level (α): {alpha}")
print(f"\nDifference in Proportions (p̂1 - p̂2): {diff_prop:.4f}")
print(f"Test Statistic (z): {z_stat_prop:.4f}")
print(f"P-value: {p_value_diff_prop:.4f}")
print(f"Critical Value (±): {z_critical:.4f}")

if p_value_diff_prop < alpha:
    print(f"\nREJECT H0 (p-value {p_value_diff_prop:.4f} < {alpha})")
    print(f"Conclusion: There IS a significant difference in the proportion of students with GPA ≥ 3.0")
else:
    print(f"\nFAIL TO REJECT H0 (p-value {p_value_diff_prop:.4f} >= {alpha})")
    print(f"Conclusion: There is NO significant difference in proportions between the two groups")

# Confidence Interval for difference in proportions
se_diff_prop_ci = np.sqrt(p1_hat * (1 - p1_hat) / n_working + p2_hat * (1 - p2_hat) / n_not_working)
ci_lower_diff_prop = diff_prop - z_critical * se_diff_prop_ci
ci_upper_diff_prop = diff_prop + z_critical * se_diff_prop_ci

print(f"\n--- {int((1-alpha)*100)}% Confidence Interval for Difference in Proportions ---")
print(f"CI: ({ci_lower_diff_prop:.4f}, {ci_upper_diff_prop:.4f})")
print(f"Interpretation: We are {int((1-alpha)*100)}% confident that the true difference in proportions")
print(f"                (Working - Non-Working) lies between {ci_lower_diff_prop:.4f} and {ci_upper_diff_prop:.4f}")


# Generate visualization for the hypothesis tests
print("\nGenerating visualization plots...")

fig = plt.figure(figsize=(14, 10))

# 1. Test for Population Mean (t-distribution)
ax1 = plt.subplot(2, 2, 1)
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df_test)
ax1.plot(x, y, 'b-', linewidth=2, label='t-distribution')
ax1.axvline(t_stat, color='red', linestyle='--', linewidth=2, label=f't-stat = {t_stat:.3f}')
ax1.axvline(-t_critical, color='green', linestyle='--', linewidth=1.5, label=f'Critical = ±{t_critical:.3f}')
ax1.axvline(t_critical, color='green', linestyle='--', linewidth=1.5)
ax1.fill_between(x[x < -t_critical], 0, t.pdf(x[x < -t_critical], df_test), alpha=0.3, color='red')
ax1.fill_between(x[x > t_critical], 0, t.pdf(x[x > t_critical], df_test), alpha=0.3, color='red')
ax1.set_title('Test for Population Mean\n(Req 2)', fontweight='bold')
ax1.set_xlabel('t-value')
ax1.set_ylabel('Probability Density')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Test for Proportion (z-distribution)
ax2 = plt.subplot(2, 2, 2)
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
ax2.plot(x, y, 'b-', linewidth=2, label='Normal distribution')
ax2.axvline(z_stat, color='red', linestyle='--', linewidth=2, label=f'z-stat = {z_stat:.3f}')
ax2.axvline(-z_critical, color='green', linestyle='--', linewidth=1.5, label=f'Critical = ±{z_critical:.3f}')
ax2.axvline(z_critical, color='green', linestyle='--', linewidth=1.5)
ax2.fill_between(x[x < -z_critical], 0, norm.pdf(x[x < -z_critical]), alpha=0.3, color='red')
ax2.fill_between(x[x > z_critical], 0, norm.pdf(x[x > z_critical]), alpha=0.3, color='red')
ax2.set_title('Test for Proportion\n(Req 3)', fontweight='bold')
ax2.set_xlabel('z-value')
ax2.set_ylabel('Probability Density')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Test for Difference in Means
ax3 = plt.subplot(2, 2, 3)
x = np.linspace(-4, 4, 1000)
y = t.pdf(x, df_diff)
ax3.plot(x, y, 'b-', linewidth=2, label='t-distribution')
ax3.axvline(t_stat_diff, color='red', linestyle='--', linewidth=2, label=f't-stat = {t_stat_diff:.3f}')
ax3.axvline(-t_critical_diff, color='green', linestyle='--', linewidth=1.5, label=f'Critical = ±{t_critical_diff:.3f}')
ax3.axvline(t_critical_diff, color='green', linestyle='--', linewidth=1.5)
ax3.fill_between(x[x < -t_critical_diff], 0, t.pdf(x[x < -t_critical_diff], df_diff), alpha=0.3, color='red')
ax3.fill_between(x[x > t_critical_diff], 0, t.pdf(x[x > t_critical_diff], df_diff), alpha=0.3, color='red')
ax3.set_title('Test for Difference in Means\n(Req 4)', fontweight='bold')
ax3.set_xlabel('t-value')
ax3.set_ylabel('Probability Density')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Test for Difference in Proportions
ax4 = plt.subplot(2, 2, 4)
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
ax4.plot(x, y, 'b-', linewidth=2, label='Normal distribution')
ax4.axvline(z_stat_prop, color='red', linestyle='--', linewidth=2, label=f'z-stat = {z_stat_prop:.3f}')
ax4.axvline(-z_critical, color='green', linestyle='--', linewidth=1.5, label=f'Critical = ±{z_critical:.3f}')
ax4.axvline(z_critical, color='green', linestyle='--', linewidth=1.5)
ax4.fill_between(x[x < -z_critical], 0, norm.pdf(x[x < -z_critical]), alpha=0.3, color='red')
ax4.fill_between(x[x > z_critical], 0, norm.pdf(x[x > z_critical]), alpha=0.3, color='red')
ax4.set_title('Test for Difference in Proportions\n(Req 5)', fontweight='bold')
ax4.set_xlabel('z-value')
ax4.set_ylabel('Probability Density')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout(pad=3.0)
plt.suptitle("Hypothesis Test Visualizations", fontsize=16, fontweight='bold', y=1.02)
plt.subplots_adjust(top=0.9)
plt.savefig('images/hypothesis_tests_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'hypothesis_tests_visualization.png'")

plt.show()
# SUMMARY TABLE
print("\n" + "="*90)
print("SUMMARY OF ALL HYPOTHESIS TESTS")
print("="*90)

summary_data = {
    'Test': [
        'Population Mean',
        'Population Proportion',
        'Difference in Means',
        'Difference in Proportions'
    ],
    'Null Hypothesis': [
        f'μ = {mu_0}',
        f'p = {p_0}',
        'μ1 - μ2 = 0',
        'p1 - p2 = 0'
    ],
    'Test Statistic': [
        f't = {t_stat:.4f}',
        f'z = {z_stat:.4f}',
        f't = {t_stat_diff:.4f}',
        f'z = {z_stat_prop:.4f}'
    ],
    'P-value': [
        f'{p_value:.4f}',
        f'{p_value_prop:.4f}',
        f'{p_value_diff:.4f}',
        f'{p_value_diff_prop:.4f}'
    ],
    'Decision (α=0.05)': [
        'Reject H0' if p_value < alpha else 'Fail to Reject H0',
        'Reject H0' if p_value_prop < alpha else 'Fail to Reject H0',
        'Reject H0' if p_value_diff < alpha else 'Fail to Reject H0',
        'Reject H0' if p_value_diff_prop < alpha else 'Fail to Reject H0'
    ],
    'Confidence Interval': [
        f'({ci_lower:.4f}, {ci_upper:.4f})',
        f'({ci_lower_p:.4f}, {ci_upper_p:.4f})',
        f'({ci_lower_diff:.4f}, {ci_upper_diff:.4f})',
        f'({ci_lower_diff_prop:.4f}, {ci_upper_diff_prop:.4f})'
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

print("HYPOTHESIS TESTS COMPLETE!")