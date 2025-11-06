import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
df = pd.read_csv('data/data.csv')

print("="*60)
print("STUDENT PART-TIME WORK DATA ANALYSIS")
print("="*60)
# Display the sample size
print(f"\nTotal Students: {len(df)}")
# Display percentages of Working/Non-working students
print(f"Working Part-Time: {df['has_work'].sum()} ({df['has_work'].sum()/len(df)*100:.1f}%)")
print(f"Not Working: {(~df['has_work']).sum()} ({(~df['has_work']).sum()/len(df)*100:.1f}%)")

# Create separate Data Frames for working and non-working students
working = df[df['has_work'] == True]
not_working = df[df['has_work'] == False]

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS - GPA")
print("="*60)

print("\n--- ALL STUDENTS ---")
# Display count,mean,std,quartiles,min,max of the sample
print(df['gpa'].describe())

print("\n--- WORKING STUDENTS ---")
# Display count,mean,std,quartiles,min,max of the working students
print(working['gpa'].describe())

print("\n--- NON-WORKING STUDENTS ---")
# Display count,mean,std,quartiles,min,max of the non-working students
print(not_working['gpa'].describe())

print("\n--- WORKING/NON-WORKING COMPARISON ---")
# Mean comparison
print(f"Mean GPA (Working): {working['gpa'].mean():.3f}")
print(f"Mean GPA (Not Working): {not_working['gpa'].mean():.3f}")
print(f"Difference: {not_working['gpa'].mean() - working['gpa'].mean():.3f}")
# Median comparison
print(f"\nMedian GPA (Working): {working['gpa'].median():.3f}")
print(f"Median GPA (Not Working): {not_working['gpa'].median():.3f}")
# Standard deviation comparison
print(f"\nStd Dev (Working): {working['gpa'].std():.3f}")
print(f"Std Dev (Not Working): {not_working['gpa'].std():.3f}")

# Create figure with subplots, each subplot is 3x2
fig = plt.figure(figsize=(14, 14))

# Pie Chart: Gender Distribution
ax1 = plt.subplot(3, 2, 1)
gender_counts = df['gender'].value_counts()
colors = ['#3b82f6', '#ef4444', '#10b981']
ax1.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax1.set_title('Gender Distribution', fontsize=14, fontweight='bold')

# Pie Chart: Work Status Distribution
ax2 = plt.subplot(3, 2, 2)
work_counts = df['has_work'].value_counts()
labels = ['Working', 'Not Working']
colors2 = ['#10b981', '#ef4444']
ax2.pie(work_counts.values, labels=labels, autopct='%1.1f%%',
        colors=colors2, startangle=90)
ax2.set_title('Work Status Distribution', fontsize=14, fontweight='bold')

# Histogram: GPA Distribution
ax3 = plt.subplot(3, 2, 3)
ax3.hist(df['gpa'].dropna(), bins=15, color='#3b82f6', alpha=0.7, edgecolor='black')
ax3.axvline(df['gpa'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["gpa"].mean():.2f}')
ax3.axvline(df['gpa'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["gpa"].median():.2f}')
ax3.set_xlabel('GPA')
ax3.set_ylabel('Frequency')
ax3.set_title('GPA Distribution (All Students)', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Box Plot: GPA Comparison
ax4 = plt.subplot(3, 2, 4)
data_to_plot = [working['gpa'].dropna(), not_working['gpa'].dropna()]
bp = ax4.boxplot(data_to_plot, tick_labels=['Working', 'Not Working'], patch_artist=True)
colors_box = ['#10b981', '#ef4444']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('GPA')
ax4.set_title('GPA Comparison: Working vs Not Working', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Scatter Plot: Study Time vs GPA
ax5 = plt.subplot(3, 2, 5)
working_data = df[df['has_work'] == True]
not_working_data = df[df['has_work'] == False]
ax5.scatter(working_data['study_time_hours'], working_data['gpa'],
           alpha=0.6, s=100, c='#ef4444', label='Working', edgecolors='black')
ax5.scatter(not_working_data['study_time_hours'], not_working_data['gpa'],
           alpha=0.6, s=100, c='#10b981', label='Not Working', edgecolors='black')
ax5.set_xlabel('Study Time (hours/week)')
ax5.set_ylabel('GPA')
ax5.set_title('Study Time vs GPA', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# Overlapping Histograms: GPA Distribution by Work Status
ax6 = plt.subplot(3, 2, 6)
ax6.hist(working['gpa'].dropna(), bins=12, alpha=0.6, label='Working', color='#ef4444', edgecolor='black')
ax6.hist(not_working['gpa'].dropna(), bins=12, alpha=0.6, label='Not Working', color='#10b981', edgecolor='black')
ax6.set_xlabel('GPA')
ax6.set_ylabel('Frequency')
ax6.set_title('GPA Distribution by Work Status', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/student_work_analysis.png', dpi=300, bbox_inches='tight')

# Additional detailed statistics
print("\n" + "="*60)
print("ADDITIONAL STATISTICS")
print("="*60)

# Year distribution
print("\n--- Year of Study Distribution ---")
# Get the counts for each year of study
print(df['year'].value_counts().sort_index())

# Work hours statistics for working students
print("\n--- Work Hours (Working Students Only) ---")
# Keep only the rows with work_time > 0
work_hours = working[working['work_time_hours'] > 0]['work_time_hours']
# Display count,mean,std,quartiles,min,max of the work hours
print(work_hours.describe())

# Study hours by work status
print("\n--- Study Hours Comparison ---")
# Mean comparison
print(f"Mean Study Hours (Working): {working['study_time_hours'].mean():.2f}")
print(f"Mean Study Hours (Not Working): {not_working['study_time_hours'].mean():.2f}")

# Failed course analysis
print("\n--- Failed Course Analysis ---")
failed_by_work = df.groupby('has_work')['failed_course'].value_counts()
print(failed_by_work)

# DETAILED ANALYSIS
# Create a new figure, each subplot is 2x2
fig2 = plt.figure(figsize=(14, 10))

# Violin Plot: GPA Distribution
ax1 = plt.subplot(2, 2, 1)  # top-left
ax1.violinplot([working['gpa'].dropna(), not_working['gpa'].dropna()],
               showmeans=True, showmedians=True)
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['Working', 'Not Working'])
ax1.set_ylabel('GPA')
ax1.set_title('GPA Distribution (Violin Plot)', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Scatter Plot: Work Time vs GPA
ax2 = plt.subplot(2, 2, 2)  # top-right
working_with_hours = working[working['work_time_hours'] > 0]
ax2.scatter(working_with_hours['work_time_hours'], working_with_hours['gpa'],
            alpha=0.6, s=100, c='#3b82f6', edgecolors='black')

# Add regression line
z = np.polyfit(working_with_hours['work_time_hours'].dropna(),
               working_with_hours['gpa'].dropna(), 1)
p = np.poly1d(z)
ax2.plot(working_with_hours['work_time_hours'].sort_values(),
         p(working_with_hours['work_time_hours'].sort_values()),
         "r--", alpha=0.8, linewidth=2)

ax2.set_xlabel('Work Time (hours/week)')
ax2.set_ylabel('GPA')
ax2.set_title('Work Time vs GPA (Working Students)', fontweight='bold')
ax2.grid(alpha=0.3)

# Bar Chart: GPA by Gender and Work Status
ax3 = plt.subplot(2, 2, 3)  # bottom-left
gender_work_gpa = df.groupby(['gender', 'has_work'])['gpa'].mean().unstack()
gender_work_gpa.plot(kind='bar', ax=ax3, color=['#ef4444', '#10b981'])
ax3.set_xlabel('Gender')
ax3.set_ylabel('Average GPA')
ax3.set_title('Average GPA by Gender and Work Status', fontweight='bold')
ax3.legend(['Not Working', 'Working'])
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Box Plot: Study Time Comparison
ax4 = plt.subplot(2, 2, 4)  # bottom-right
ax4.boxplot([working['study_time_hours'].dropna(),
             not_working['study_time_hours'].dropna()],
            tick_labels=['Working', 'Not Working'], patch_artist=True)
ax4.set_ylabel('Study Time (hours/week)')
ax4.set_title('Study Time Comparison', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/student_work_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("\nGraphs saved as")
print("'student_work_analysis.png'\n'student_work_detailed_analysis.png'")
plt.show()

print("ANALYSIS COMPLETE!")
