import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
X = pd.read_csv("../features.csv", index_col=["video_id", "frame"])

print(f"Dataset shape: {X.shape}")
print(f"Total cells: {X.shape[0] * X.shape[1]}")

# Calculate missing data statistics
print("\n" + "="*70)
print("MISSING DATA OVERVIEW")
print("="*70)

# Overall missing data
total_missing = X.isna().sum().sum()
total_cells = X.shape[0] * X.shape[1]
missing_percentage = (total_missing / total_cells) * 100

print(f"\nTotal missing values: {total_missing:,}")
print(f"Overall missing percentage: {missing_percentage:.2f}%")

# Missing data per column
print("\n" + "="*70)
print("MISSING DATA BY COLUMN")
print("="*70)

na_percentage = X.isna().mean() * 100
missing_summary = pd.DataFrame({
    'Column': X.columns,
    'Missing_Count': X.isna().sum(),
    'Missing_Percentage': na_percentage
}).sort_values(by='Missing_Percentage', ascending=False)

print(f"\nColumns with missing data: {(missing_summary['Missing_Percentage'] > 0).sum()}")
print(f"Columns with >10% missing: {(missing_summary['Missing_Percentage'] > 10).sum()}")
print(f"Columns with >50% missing: {(missing_summary['Missing_Percentage'] > 50).sum()}")

print("\nTop 20 columns with most missing data:")
print(missing_summary.head(20).to_string(index=False))

# Missing data per row
print("\n" + "="*70)
print("MISSING DATA BY ROW")
print("="*70)

row_missing = X.isna().sum(axis=1)
row_missing_pct = (row_missing / X.shape[1]) * 100

print(f"\nRows with any missing data: {(row_missing > 0).sum()}")
print(f"Rows with >10% missing: {(row_missing_pct > 10).sum()}")
print(f"Rows with >50% missing: {(row_missing_pct > 50).sum()}")
print(f"\nMissing data per row statistics:")
print(f"  Mean: {row_missing_pct.mean():.2f}%")
print(f"  Median: {row_missing_pct.median():.2f}%")
print(f"  Max: {row_missing_pct.max():.2f}%")

# Save detailed report
print("\nSaving detailed missing data report to 'missing_data_report.csv'...")
missing_summary.to_csv("missing_data_report.csv", index=False)

# Visualizations
print("\nGenerating visualizations...")

# Figure 1: Distribution of missing percentages
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of missing percentages per column
axes[0, 0].hist(na_percentage, bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Missing Percentage (%)')
axes[0, 0].set_ylabel('Number of Columns')
axes[0, 0].set_title('Distribution of Missing Data per Column')
axes[0, 0].grid(True, alpha=0.3)

# Bar plot of top 20 columns with most missing data
top_20 = missing_summary.head(20)
axes[0, 1].barh(range(len(top_20)), top_20['Missing_Percentage'])
axes[0, 1].set_yticks(range(len(top_20)))
axes[0, 1].set_yticklabels(top_20['Column'], fontsize=8)
axes[0, 1].set_xlabel('Missing Percentage (%)')
axes[0, 1].set_title('Top 20 Columns with Most Missing Data')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Histogram of missing data per row
axes[1, 0].hist(row_missing_pct, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('Missing Percentage (%)')
axes[1, 0].set_ylabel('Number of Rows')
axes[1, 0].set_title('Distribution of Missing Data per Row')
axes[1, 0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_na_pct = np.sort(na_percentage)
cumulative = np.arange(1, len(sorted_na_pct) + 1) / len(sorted_na_pct) * 100
axes[1, 1].plot(sorted_na_pct, cumulative)
axes[1, 1].set_xlabel('Missing Percentage (%)')
axes[1, 1].set_ylabel('Cumulative % of Columns')
axes[1, 1].set_title('Cumulative Distribution of Missing Data')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=90, color='r', linestyle='--', label='90% of columns')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('missing_data_visualization.png', dpi=300, bbox_inches='tight')
print("Saved visualization to 'missing_data_visualization.png'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
