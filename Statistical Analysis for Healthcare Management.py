#!/usr/bin/env python
# coding: utf-8

# Graded Assignment on Statistical Analysis for Healthcare Management

# Description
# Business Problem Statement:
# HealthCare Plus is a multi-specialty hospital that provides medical consultations, treatments, and diagnostic services. The hospital management wants to use statistical analysis to optimize operations, improve patient care, and make data-driven decisions.
# To achieve this, HealthCare Plus has collected data on patient admission times, recovery durations, patient satisfaction scores, effectiveness of different treatments, hospital expenses, and staff efficiency. The goal is to analyze this data and provide insights that can help improve hospital operations, enhance patient satisfaction, and reduce unnecessary expenses.
# You have been assigned the task of conducting a statistical analysis of the collected data to support hospital management in making informed decisions.

# In[170]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import skewnorm


# ****************************************************************************************************
# ## Section A
# ****************************************************************************************************

# #### 1. HealthCare Plus recorded the daily number of patient admissions for the past 10 days:
# 
#     [32, 28, 35, 30, 29, 27, 31, 34, 33, 30]
# 
#     •	Compute the mean, median, and mode of patient admissions.
#     •	Which measure best represents patient admissions?
#     •	If the hospital increases its admission capacity by 10%, how will this affect the measures of central tendency?

# ## Answer:

# In[81]:


# Given data
admissions = np.array([32, 28, 35, 30, 29, 27, 31, 34, 33, 30])

# Compute measures of central tendency
mean_admissions = np.mean(admissions)
median_admissions = np.median(admissions)
mode_admissions = stats.mode(admissions, keepdims=True).mode[0]

print(f"Mean: {mean_admissions}")
print(f"Median: {median_admissions}")
print(f"Mode: {mode_admissions}")

# Measure best represents patient admissions
print(f"\nBest measure:")
print(f"The mean best represents patient admissions, as it summarizes the overall trend effectively without significant skewness. However, if future data contains extreme values, the median might be a better choice.")

# Increase admissions by 10%
increased_admissions = admissions * 1.10

# Compute new measures
new_mean = np.mean(increased_admissions)
new_median = np.median(increased_admissions)
new_mode = stats.mode(increased_admissions, keepdims=True).mode[0]

print(f"\nAfter 10% Increase:")
print(f"New Mean: {new_mean}")
print(f"New Median: {new_median}")
print(f"New Mode: {new_mode}")

# Impact of a 10% Increase in Admission Capacity on Measures of Central Tendency
print(f"\nImpact of Increase in Admission Capacity on measures:")
print(f"All three measures increase proportionally by 10%.")
print(f"The overall distribution shape remains the same, but the values shift higher.")
print(f"The relative variation remains unchanged because the proportionality is applied uniformly.")

print(f"\nGraphical presentation:")

# Patient Admissions Visualization
plt.figure(figsize=(8,5))
plt.plot(admissions, marker='o', linestyle='-', label='Original Admissions')
plt.plot(increased_admissions, marker='s', linestyle='--', label='10% Increased Admissions')
plt.axhline(mean_admissions, color='blue', linestyle='dotted', label='Original Mean')
plt.axhline(new_mean, color='red', linestyle='dotted', label='New Mean')
plt.xlabel('Days')
plt.ylabel('Number of Admissions')
plt.title('Patient Admissions Trend')
plt.legend()
plt.show()

# Central Tendency Comparison Visualization
plt.figure(figsize=(8,5))
bars = ['Mean', 'Median', 'Mode']
original_values = [mean_admissions, median_admissions, mode_admissions]
new_values = [new_mean, new_median, new_mode]
bar_width = 0.3
x = np.arange(len(bars))
plt.bar(x - bar_width/2, original_values, width=bar_width, label='Original', color='blue')
plt.bar(x + bar_width/2, new_values, width=bar_width, label='After 10% Increase', color='red')
plt.xticks(x, bars)
plt.ylabel('Values')
plt.title('Comparison of Measures of Central Tendency')
plt.legend()
plt.show()


# ________________________________________

# ### 2. The recovery duration (in days) of 10 patients who underwent the same surgery is recorded as follows:
# 
#     [5, 7, 6, 8, 9, 5, 6, 7, 8, 6]
# 
#     •	Calculate the range, variance, and standard deviation.
#     •	What does the standard deviation indicate about variability in recovery times?
#     •	If two new patients take 4 and 10 days to recover, how will this impact the standard deviation?

# ## Answer:

# In[75]:


# Given data
recovery_days = np.array([5, 7, 6, 8, 9, 5, 6, 7, 8, 6])

# Compute range, variance, and standard deviation for recovery days
range_recovery = np.ptp(recovery_days)
variance_recovery = np.var(recovery_days, ddof=0)
std_dev_recovery = np.std(recovery_days, ddof=0)

print(f"\nRecovery Duration Analysis:")
print(f"Range: {range_recovery}")
print(f"Variance: {variance_recovery}")
print(f"Standard Deviation: {std_dev_recovery}")

# Adding two new patients (4 and 10 days recovery)
updated_recovery_days = np.append(recovery_days, [4, 10])
updated_variance = np.var(updated_recovery_days, ddof=0)
updated_std_dev = np.std(updated_recovery_days, ddof=0)

print(f"\nAfter Adding Two New Patients:")
print(f"Updated Variance: {updated_variance}")
print(f"Updated Standard Deviation: {updated_std_dev}")

# Interpretation of Standard Deviation in Recovery Times
print(f"\nVariability in Recovery Times:")
print(f"Before Adding: The standard deviation reflects how recovery times vary among the 10 original patients.")
print(f"After Adding: The inclusion of 4 and 10 days increases the spread, increasing the standard deviation. This means there is greater variability in recovery times, indicating that recovery outcomes may not be as predictable.")

print(f"\nGraphical presentation:")

# Recovery Duration Visualization
plt.figure(figsize=(8,5))
plt.hist(recovery_days, bins=5, alpha=0.7, label='Original Recovery Days', color='blue', edgecolor='black')
plt.hist(updated_recovery_days, bins=6, alpha=0.5, label='Updated Recovery Days', color='red', edgecolor='black')
plt.axvline(np.mean(recovery_days), color='blue', linestyle='dashed', label='Original Mean')
plt.axvline(np.mean(updated_recovery_days), color='red', linestyle='dashed', label='Updated Mean')
plt.xlabel('Recovery Days')
plt.ylabel('Frequency')
plt.title('Distribution of Recovery Durations')
plt.legend()
plt.show()


# ________________________________________

# ### 3. Patient satisfaction scores (on a scale of 1 to 10) collected from a hospital survey are:
#     [8, 9, 7, 8, 10, 7, 9, 6, 10, 8, 7, 9]
#     
#     •	Compute skewness and kurtosis.
#     •	Interpret the results—does the data suggest a normal distribution?
#     •	If the hospital implements a new customer service initiative, and satisfaction scores shift higher, what type of skewness change would you expect?

# ## Answer:

# In[122]:


# Given data
satisfaction_scores = np.array([8, 9, 7, 8, 10, 7, 9, 6, 10, 8, 7, 9])

# Compute skewness and kurtosis for satisfaction scores
skewness = stats.skew(satisfaction_scores)
kurtosis = stats.kurtosis(satisfaction_scores)

print(f"\nSatisfaction Scores Analysis:")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")

# Interpretation of skewness and kurtosis
if abs(skewness) < 0.5:
    skewness_interpretation = "The data is approximately symmetric."
elif skewness > 0.5:
    skewness_interpretation = "The data is positively skewed (right-skewed), meaning more values are concentrated on the lower end."
else:
    skewness_interpretation = "The data is negatively skewed (left-skewed), meaning more values are concentrated on the higher end."

if kurtosis < 0:
    kurtosis_interpretation = "The data has lighter tails than a normal distribution (platykurtic)."
elif kurtosis > 0:
    kurtosis_interpretation = "The data has heavier tails than a normal distribution (leptokurtic), meaning more extreme values."
else:
    kurtosis_interpretation = "The data has a normal kurtosis (mesokurtic)."

print(f"\nInterpretation of Skewness: {skewness_interpretation}")
print(f"Interpretation of Kurtosis: {kurtosis_interpretation}")

# Visualization of satisfaction scores
plt.figure(figsize=(8,5))
plt.hist(satisfaction_scores, bins=5, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(satisfaction_scores), color='red', linestyle='dashed', label='Mean')
plt.xlabel('Satisfaction Score')
plt.ylabel('Frequency')
plt.title('Distribution of Patient Satisfaction Scores')
plt.legend()
plt.show()

# Interpretation of skewness change if scores increase
print("\nIf satisfaction scores increase due to a customer service initiative, skewness would likely decrease or shift toward negative values, indicating a more left-skewed (higher scores concentrated on the right) distribution.")


# ________________________________________

# ### 4. HealthCare Plus wants to analyze the relationship between nurse staffing levels and patient recovery time. Data from 6 hospital departments is provided:
# 
# | Number of Nurses | Average Recovery Time (days) |
# |------------------|------------------------------|
# | 10	           | 8                            |
# | 12	           | 7                            |
# | 15	           | 6                            |
# | 18	           | 5                            |
# | 20	           | 4                            |
# | 22	           | 3                            |
#     
#     •	Compute the correlation coefficient between nurse staffing and patient recovery time.
#     •	If the hospital increases the number of nurses by 5 per department, how will this affect the recovery time based on the trend?

# ## Answer:

# In[156]:


# Nurse staffing vs. recovery time data
nurses = np.array([10, 12, 15, 18, 20, 22])
recovery_times = np.array([8, 7, 6, 5, 4, 3])

# Compute correlation coefficient
correlation_coefficient = np.corrcoef(nurses, recovery_times)[0, 1]
print(f"\nCorrelation between nurse staffing levels and patient recovery time: {correlation_coefficient:.4f}")

# Predict impact of increasing nurses by 5 per department
nurses_increased = nurses + 5
slope, intercept = np.polyfit(nurses, recovery_times, 1)  # Linear regression
predicted_recovery_times = slope * nurses_increased + intercept

print(f"\nPredicted Recovery Times after increasing nurses by 5:")
for n, r in zip(nurses_increased, predicted_recovery_times):
    print(f"{n} nurses -> {r:.2f} days")

# Visualization of nurse staffing vs. recovery times
plt.figure(figsize=(8,5))
plt.scatter(nurses, recovery_times, color='blue', label='Actual Data')
plt.plot(nurses_increased, predicted_recovery_times, color='red', linestyle='dashed', label='Predicted Trend')
plt.xlabel("Number of Nurses")
plt.ylabel("Average Recovery Time (days)")
plt.title("Impact of Nurse Staffing on Recovery Time")
plt.legend()
plt.show()


# ________________________________________
# ## Section B
# ________________________________________

# ### 5. The hospital claims that the average patient wait time in the emergency department is 30 minutes. A sample of 10 patient wait times (in minutes) is recorded:
#     [32, 29, 31, 34, 33, 27, 30, 28, 35, 26]
# 
#     •	Test whether the hospital’s claim is valid at a 5% significance level.
#     •	State the null and alternative hypotheses.
#     •	If the wait time significantly exceeds 30 minutes, what changes should the hospital implement to reduce waiting time?

# ## Answer:

# Null Hypothesis 
# 
# $$ H_0: \text{[The average patient wait time in the emergency department is 30 minutes]} $$
# 
# or 
# 
# $$ \mu = 30 $$
# 

# Alternate Hypothesis
# 
# $$ H_1: \text{[The average patient wait time in the emergency department is 30 minutes]} $$
# 
# or 
# 
# $$ \mu \neq 30 $$

# What all information do we have 
# 
# ```n = 10```
# 
# $$ \bar{x} =  30.5$$
# 
# $$ \mu = 30$$
# 
# $$ s = 3.03 $$ 
# 

# Would this formula apply ?
# 
# $$ t = \frac{\bar{x} - \mu}{\frac{s}{\sqrt{n}}}$$ 
# 
# $$ t = \frac{30.5 - 30}{\frac{3.03}{\sqrt{10}}}$$ 
# 
# $$ t = \frac{0.5}{0.95817}$$ 
# 
# $$ t = 0.52$$ 

# In[128]:


wait_times = np.array([32, 29, 31, 34, 33, 27, 30, 28, 35, 26])

# Hypothesis testing for wait times
hypothesized_mean = 30
alpha = 0.05

t_stat, p_value = stats.ttest_1samp(wait_times, hypothesized_mean)
print(f"\nHypothesis Testing for Wait Times:")
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")

# Visualization of hypothesis test
plt.figure(figsize=(8,5))
sns.histplot(wait_times, kde=True, bins=5, color='blue', edgecolor='black')
plt.axvline(hypothesized_mean, color='red', linestyle='dashed', linewidth=2, label='Hypothesized Mean')
plt.axvline(np.mean(wait_times), color='green', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.xlabel("Wait Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Patient Wait Times")
plt.legend()
plt.show()

# Interpretation of hypothesis test
if p_value < alpha:
    print("The null hypothesis is rejected. The average wait time significantly differs from 30 minutes.")
    print("Recommendation: The hospital should implement queue management, improve staff efficiency, and optimize scheduling to reduce wait times.")
else:
    print("The null hypothesis is not rejected. There is no significant difference from the claimed 30-minute average wait time.")


# In[29]:


# Function to plot skew-normal distribution
def plot_skew_normal_distribution(data, sample_mean, sample_std, mu_0):
    x = np.linspace(min(data)-5, max(data)+5, 1000)
    y = stats.skewnorm.pdf(x, a=2, loc=sample_mean, scale=sample_std)
    
    plt.figure(figsize=(8,5))
    plt.hist(data, bins=5, density=True, alpha=0.6, color='g', label='Sample Data')
    plt.plot(x, y, 'r', label='Skew-Normal Fit')
    plt.axvline(sample_mean, color='b', linestyle='dashed', linewidth=2, label='Sample Mean')
    plt.axvline(mu_0, color='k', linestyle='dashed', linewidth=2, label='Claimed Mean')
    plt.legend()
    plt.xlabel("Wait Time (minutes)")
    plt.ylabel("Density")
    plt.title("Skew-Normal Distribution of Wait Times")
    plt.show()

# Function to plot t-score distribution
def plot_t_score_distribution(t_value, df):
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    plt.figure(figsize=(8,5))
    plt.plot(x, y, 'b-', label='t-Distribution')
    plt.axvline(t_value, color='r', linestyle='dashed', linewidth=2, label=f't-Value: {t_value:.2f}')
    plt.axvline(-stats.t.ppf(1 - alpha/2, df), color='k', linestyle='dashed', linewidth=2, label='Critical t-Value')
    plt.axvline(stats.t.ppf(1 - alpha/2, df), color='k', linestyle='dashed', linewidth=2)
    plt.legend()
    plt.xlabel("t-Score")
    plt.ylabel("Density")
    plt.title("t-Distribution with t-Value")
    plt.show()

# Function to plot skew-normal distribution using z-scores
def plot_skew_normal_distribution_z_score(mean, std_dev, skewness, kurtosis, tick_locations, tick_labels, title=None):
    alpha = skewness  # Skewness parameter for skew-normal distribution
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = skewnorm.pdf(x, alpha, loc=mean, scale=std_dev)
    
    plt.figure(figsize=(8,5))
    plt.plot(x, y, label=f'Skew-Normal Distribution\nMean = {mean}, Std Dev = {std_dev}, Skew = {skewness}, Kurtosis = {kurtosis}')
    plt.xlabel('Z-Score')
    plt.ylabel('Probability Density')
    plt.grid(True)
    
    if title:
        plt.title(title)
    else:
        plt.title('Skew-Normal Distribution Curve')
    
    plt.xticks(tick_locations, tick_labels)
    plt.show()

# Given data
wait_times = np.array([32, 29, 31, 34, 33, 27, 30, 28, 35, 26])
mu_0 = 30  # Claimed population mean
alpha = 0.05

# Compute sample statistics
n = len(wait_times)
sample_mean = np.mean(wait_times)
sample_std = np.std(wait_times, ddof=1)  # Using Bessel's correction

# Compute t-value
t_value = (sample_mean - mu_0) / (sample_std / np.sqrt(n))

df = n - 1  # Degrees of freedom
critical_t = stats.t.ppf(1 - alpha/2, df)  # Two-tailed test
p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))  # Two-tailed p-value

# Decision
reject_H0 = abs(t_value) > critical_t

# Print results
print(f"Sample Mean: {sample_mean:.2f}")
print(f"Sample Standard Deviation: {sample_std:.2f}")
print(f"t-Value: {t_value:.2f}")
print(f"Critical t-Value: {critical_t:.2f}")
print(f"p-Value: {p_value:.4f}")
print("Decision: Reject H0" if reject_H0 else "Decision: Fail to Reject H0")

# Define tick locations and labels for z-score plot
tick_locations = [-2, -1, 0, 1, 2]
tick_labels = ['-2σ', '-1σ', 'Mean', '+1σ', '+2σ']

# Call functions for visualization
plot_skew_normal_distribution(wait_times, sample_mean, sample_std, mu_0)
plot_t_score_distribution(t_value, df)
plot_skew_normal_distribution_z_score(0, 1, 1, 3, tick_locations, tick_labels, title='Z-Score Skew-Normal Distribution')


# ________________________________________

# ### 6. A survey was conducted on hospital cleanliness and patient satisfaction. The following data was collected:
#     
# | Cleanliness Rating | Satisfied Patients | Unsatisfied Patients |
# |--------------------|--------------------|----------------------|
# | High	             | 90	              | 10                   |
# | Medium	         | 60		          | 40                   |
# | Low	             | 30		          | 70                   |
# 
#     •	Perform an analysis to check whether hospital cleanliness and patient satisfaction are dependent.
#     •	If cleanliness ratings improve, how do you expect the distribution of satisfied and unsatisfied patients to change

# ## Answer:

# Null Hypothesis (H₀): Cleanliness rating and patient satisfaction are independent (no relationship).

# Alternative Hypothesis (H₁): Cleanliness rating and patient satisfaction are dependent (cleanliness affects satisfaction).

#  | Cleanliness Rating | Satisfied Patients | Unsatisfied Patients | Total  |
# |--------------------|--------------------|----------------------|--------|
# | High              | 90                 | 10                   | 100    |
# | Medium            | 60                 | 40                   | 100    |
# | Low               | 30                 | 70                   | 100    |
# | **Total**         | 180                | 120                  | 300    |
# 

# Expected Frequencies using the formula:
# 
# $$
# E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}
# $$
# 
# $$
# E = \frac{(90 + 10) \times (90 + 60 + 30)}{90 + 60 + 30 + 10 + 40 + 70}
# $$$$
# E = \frac{100 \times 180}{300} = 60
# $$
# 
# 
# 
#  
# 

# ### Chi-Square Statistic Formula
# 
# $$
# \chi^2 = \sum \frac{(O - E)^2}{E}
# $$
# 
# Where:
# - \( O \) = Observed frequency  
# - \( E \) = Expected frequency  
# 
# ### Steps to Calculate:
# 
# 1. Compute the expected frequencies using:  
# 
#    $$
#    E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}
#    $$
# 
# 2. Apply the Chi-Square formula for each category:
# 
#    $$
#    \chi^2 = \sum \frac{(O - E)^2}{E}
#    $$
# 
# 3. Sum all values to get the final **Chi-Scleanliness and satisfaction are dependent.
# 

# In[64]:


# Hospital Cleanliness vs. Patient Satisfaction Analysis
cleanliness_data = np.array([[90, 10],  # High
                              [60, 40],  # Medium
                              [30, 70]]) # Low

chi2, p, dof, expected = chi2_contingency(cleanliness_data)

print(f"\nChi-Square Test for Independence:")
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

if p < alpha:
    print("The null hypothesis is rejected. Cleanliness and patient satisfaction are dependent.")
    print("Recommendation: Improving cleanliness is likely to increase patient satisfaction.")
else:
    print("The null hypothesis is not rejected. There is no significant relationship between cleanliness and patient satisfaction.")


# ________________________________________

# ### 7. The hospital tested three different treatment methods (A, B, and C) for managing post-surgery pain. The recovery durations (in days) under each treatment are:
#     
#     •	Treatment A: [5, 6, 7, 5, 6]
#     •	Treatment B: [8, 9, 7, 8, 10]
#     •	Treatment C: [4, 5, 6, 5, 4]
# 
#     •	Conduct an analysis to check if there is a significant difference in recovery times among the treatment methods.
#     •	State the null and alternative hypotheses.
#     •	If the hospital introduces a new treatment (D), what data should be collected before concluding its effectiveness?

# ## Answer:

# In[123]:


import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# Recovery durations for each treatment
treatment_A = np.array([5, 6, 7, 5, 6])
treatment_B = np.array([8, 9, 7, 8, 10])
treatment_C = np.array([4, 5, 6, 5, 4])
treatment_D = np.array([6, 7, 6, 8, 7])  # New Treatment D

# Perform One-Way ANOVA
f_statistic, p_value = stats.f_oneway(treatment_A, treatment_B, treatment_C, treatment_D)

# Print results
print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in recovery times among treatments.")
    
    # Combine data for Tukey HSD test
    all_data = np.concatenate([treatment_A, treatment_B, treatment_C, treatment_D])
    group_labels = (["A"] * len(treatment_A) + ["B"] * len(treatment_B) + 
                    ["C"] * len(treatment_C) + ["D"] * len(treatment_D))

    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
    print("\nPost-hoc Tukey's HSD Test Results:")
    print(tukey_results)
    
else:
    print("Fail to reject the null hypothesis: No significant difference in recovery times among treatments.")

# Calculate means and standard deviations for each treatment
means = [np.mean(treatment_A), np.mean(treatment_B), np.mean(treatment_C), np.mean(treatment_D)]
stds = [np.std(treatment_A, ddof=1), np.std(treatment_B, ddof=1), 
        np.std(treatment_C, ddof=1), np.std(treatment_D, ddof=1)]

# Visualization - Boxplot with Mean, Std Dev, and Significance
plt.figure(figsize=(8, 5))
ax = sns.boxplot(data=[treatment_A, treatment_B, treatment_C, treatment_D], palette="Set2")
plt.xticks(ticks=[0, 1, 2, 3], labels=['Treatment A', 'Treatment B', 'Treatment C', 'Treatment D'])
plt.xlabel("Treatment Methods")
plt.ylabel("Recovery Time (days)")
plt.title("Comparison of Recovery Durations for Different Treatments")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add mean & standard deviation annotations
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.scatter(i, mean, color='black', marker='o', label="Mean" if i == 0 else "")
    plt.errorbar(i, mean, yerr=std, fmt='none', color='black', capsize=5)

# Add significance markers based on Tukey HSD results
sig_pairs = []
sig_labels = []
x_positions = []
y_offset = max(means) + 1  # Offset above the highest mean

for comb in combinations(["A", "B", "C", "D"], 2):  
    group1, group2 = comb
    result = tukey_results.summary()
    p_val = tukey_results.pvalues[list(combinations(["A", "B", "C", "D"], 2)).index(comb)]
    
    if p_val < alpha:  # Significant difference
        sig_pairs.append((["A", "B", "C", "D"].index(group1), ["A", "B", "C", "D"].index(group2)))
        sig_labels.append(f"p={p_val:.3f}")
        x_positions.append((sig_pairs[-1][0] + sig_pairs[-1][1]) / 2)

# Draw significance markers
for (x1, x2), label, xpos in zip(sig_pairs, sig_labels, x_positions):
    plt.plot([x1, x2], [y_offset, y_offset], color='black', linewidth=1)
    plt.text(xpos, y_offset + 0.2, label, ha='center', color='red')

# Show legend only once
plt.legend()
plt.show()


# ________________________________________

# ### 8. The hospital administration time (in minutes) for 12 patients is recorded as:
#     
#     [12, 15, 14, 16, 18, 13, 14, 17, 15, 19, 16, 14]
# 
#     •	Analyze whether the administration times follow a normal distribution.
#     •	Explain why this analysis is important in healthcare data.
#     •	If emergency cases increase, how would you expect the distribution of administration times to change?

# ## Answer:

# In[162]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Given data
admin_times = np.array([12, 15, 14, 16, 18, 13, 14, 17, 15, 19, 16, 14])

# Histogram with KDE
plt.figure(figsize=(8, 5))
sns.histplot(admin_times, bins=6, kde=True, color='skyblue', edgecolor='black')
plt.axvline(np.mean(admin_times), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.xlabel("Administration Time (minutes)")
plt.ylabel("Frequency")
plt.title("Distribution of Administration Times")
plt.legend()
plt.show()

# Q-Q Plot
plt.figure(figsize=(6, 6))
stats.probplot(admin_times, dist="norm", plot=plt)
plt.title("Q-Q Plot for Administration Times")
plt.show()

# Shapiro-Wilk test for normality
shapiro_test = stats.shapiro(admin_times)
print(f"Shapiro-Wilk Test Statistic: {shapiro_test.statistic:.4f}, P-Value: {shapiro_test.pvalue:.4f}")

# Interpretation
alpha = 0.05
if shapiro_test.pvalue < alpha:
    print("Reject the null hypothesis: The data does not follow a normal distribution.")
else:
    print("Fail to reject the null hypothesis: The data appears to follow a normal distribution.")


# ________________________________________

# ### 9. The hospital is studying the distribution of patient arrival times in the emergency department. Historical data suggests that emergency cases arrive at an average rate of 5 per hour.
# 
#     •	Model this scenario using an appropriate probability distribution.
#     •	What is the probability that exactly 3 emergency cases will arrive in the next hour?
#     •	If a major accident occurs in the city, how would this affect the probability distribution of emergency arrivals?

# ## Answer:

# In[165]:


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Given Poisson parameter (average arrivals per hour)
lambda_val = 5  # Mean arrival rate per hour

# Compute probability of exactly 3 arrivals
k = 3
poisson_prob = stats.poisson.pmf(k, lambda_val)
print(f"Probability of exactly 3 emergency cases arriving in the next hour: {poisson_prob:.4f}")

# Visualizing the Poisson Distribution
x = np.arange(0, 15)  # Possible number of emergency arrivals
poisson_pmf = stats.poisson.pmf(x, lambda_val)

plt.figure(figsize=(8, 5))
plt.bar(x, poisson_pmf, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(lambda_val, color='red', linestyle='dashed', linewidth=2, label='Mean (λ=5)')
plt.xlabel("Number of Emergency Arrivals")
plt.ylabel("Probability")
plt.title("Poisson Distribution of Emergency Arrivals per Hour")
plt.legend()
plt.show()


# ________________________________________

# ### 10. The number of surgeries performed per day in the hospital follows a specific distribution pattern. Historical data shows the following frequencies:
# 
# | Surgeries Performed | Frequency |
# |---------------------|-----------|
# | 0	                  |  5        |
# | 1		              | 12        |
# | 2		              | 18        |
# | 3		              | 22        |
# | 4		              | 15        |
# | 5		              |  8        |
# 
#     •	Identify and justify the type of probability distribution that best fits this data.
#     •	Calculate the expected number of surgeries performed per day.
#     •	If a new surgical team is hired, how will this affect the probability distribution of daily surgeries?

# ## Answer:

# In[168]:


import numpy as np
import matplotlib.pyplot as plt

# Given data
surgeries = np.array([0, 1, 2, 3, 4, 5])
frequencies = np.array([5, 12, 18, 22, 15, 8])
total_observations = np.sum(frequencies)

# Compute the expected value (mean)
expected_value = np.sum(surgeries * frequencies) / total_observations
print(f"Expected number of surgeries per day: {expected_value:.2f}")

# Visualizing the distribution
plt.figure(figsize=(8, 5))
plt.bar(surgeries, frequencies / total_observations, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel("Number of Surgeries Performed")
plt.ylabel("Probability")
plt.title("Probability Distribution of Daily Surgeries")
plt.axvline(expected_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean = {expected_value:.2f}')
plt.legend()
plt.show()


# ----------
