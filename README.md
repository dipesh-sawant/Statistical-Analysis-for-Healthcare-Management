# Healthcare Data Analysis Using Python, Pandas & NumPy

## Overview
HealthCare Plus is a multi-specialty hospital that aims to optimize operations, enhance patient care, and make data-driven decisions. This project performs **statistical analysis** on hospital data using Python, Pandas, and NumPy. The analysis includes central tendency measures, variance calculations, hypothesis testing, probability distributions, and correlation analysis.

## Dataset
The project analyzes multiple datasets related to:
- **Patient admissions**
- **Recovery durations**
- **Satisfaction scores**
- **Nurse staffing & recovery time**
- **Emergency wait times**
- **Hospital cleanliness & patient satisfaction**
- **Treatment effectiveness**
- **Administration times**
- **Emergency case arrivals**
- **Surgical operations per day**

## Analysis Performed
### **Section A: Descriptive Statistics & Correlation Analysis**
1. **Patient Admissions**
   - Computed **mean, median, and mode** of daily admissions.
   - Analyzed the impact of a **10% capacity increase** on central tendency.

2. **Recovery Durations**
   - Calculated **range, variance, and standard deviation**.
   - Examined the impact of additional recovery cases on variability.

3. **Patient Satisfaction Scores**
   - Measured **skewness and kurtosis**.
   - Determined whether the distribution is normal.

4. **Nurse Staffing vs. Recovery Time**
   - Calculated the **correlation coefficient** to analyze relationships.
   - Predicted how increasing nurse numbers affects recovery time.

### **Section B: Hypothesis Testing & Probability Distributions**
5. **Emergency Department Wait Time**
   - Conducted a **hypothesis test** to validate the hospital’s 30-minute wait time claim.

6. **Hospital Cleanliness & Satisfaction**
   - Performed a **chi-square test** to check dependency.

7. **Treatment Effectiveness**
   - Used **ANOVA** to determine significant differences in recovery times among treatments A, B, and C.

8. **Administration Time Analysis**
   - Tested for **normal distribution** using visualization and the **Shapiro-Wilk test**.

9. **Emergency Patient Arrivals**
   - Modeled arrivals using a **Poisson distribution**.
   - Calculated the probability of exactly 3 arrivals in the next hour.

10. **Daily Surgery Counts**
   - Identified the **best probability distribution**.
   - Computed the **expected number of surgeries per day**.

## Tools & Libraries Used
- **Python**
- **Pandas** (Data manipulation)
- **NumPy** (Numerical computations)
- **Matplotlib & Seaborn** (Data visualization)
- **SciPy** (Statistical analysis)
