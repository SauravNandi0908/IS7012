import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

import matplotlib.pyplot as plt

# Create a sample sports dataset
np.random.seed(42)
n_rows = 500

data = {
    'Player_ID': range(1, n_rows + 1),
    'Name': [f'Player_{i}' for i in range(1, n_rows + 1)],
    'Sport': np.random.choice(['Basketball', 'Football', 'Baseball', 'Soccer'], n_rows),
    'Age': np.random.randint(18, 40, n_rows),
    'Height_cm': np.random.normal(180, 10, n_rows),
    'Weight_kg': np.random.normal(80, 15, n_rows),
    'Games_Played': np.random.randint(5, 82, n_rows),
    'Points_Per_Game': np.random.uniform(5, 30, n_rows),
    'Assists': np.random.randint(0, 10, n_rows),
    'Years_Experience': np.random.randint(0, 20, n_rows),
    'Salary_USD': np.random.uniform(50000, 5000000, n_rows)
}

df = pd.DataFrame(data)
df.to_csv('sports_dataset.csv', index=False)
print("Dataset saved as 'sports_dataset.csv'\n")

df = pd.read_csv('sports_dataset.csv')

print("=" * 50)
print("BASIC INFORMATION")
print("=" * 50)
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few records:\n{df.head()}")
print(f"\nSummary Statistics:\n{df.describe()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")

print("\n" + "=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

print("\n" + "=" * 50)
print("GROUPBY ANALYSIS BY SPORT")
print("=" * 50)
sport_analysis = df.groupby('Sport').agg({
    'Points_Per_Game': ['mean', 'std', 'max'],
    'Salary_USD': ['mean', 'median'],
    'Age': 'mean',
    'Years_Experience': 'mean',
    'Player_ID': 'count'
}).round(2)
print(sport_analysis)

print("\n" + "=" * 50)
print("SALARY STATISTICS BY SPORT")
print("=" * 50)
print(df.groupby('Sport')['Salary_USD'].describe().round(0))

print("\n" + "=" * 50)
print("AGE GROUPS ANALYSIS")
print("=" * 50)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 30, 35, 40], labels=['18-25', '26-30', '31-35', '36+'])
print(df.groupby('Age_Group').agg({'Salary_USD': 'mean', 'Points_Per_Game': 'mean', 'Player_ID': 'count'}).round(2))

print("\n" + "=" * 50)
print("TOP PERFORMERS")
print("=" * 50)
top_scorers = df.nlargest(5, 'Points_Per_Game')[['Name', 'Sport', 'Points_Per_Game', 'Salary_USD']]
print("Top 5 Scorers:\n", top_scorers)

top_earners = df.nlargest(5, 'Salary_USD')[['Name', 'Sport', 'Salary_USD', 'Points_Per_Game']]
print("\nTop 5 Earners:\n", top_earners)

print("\n" + "=" * 50)
print("STATISTICAL TESTS")
print("=" * 50)
f_stat, p_value = stats.f_oneway(*[group['Salary_USD'].values for name, group in df.groupby('Sport')])
print(f"ANOVA Test (Salary by Sport): F-stat={f_stat:.2f}, p-value={p_value:.4f}")

print("\n" + "=" * 50)
print("VISUALIZATIONS")
print("=" * 50)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Salary by Sport boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Sport', y='Salary_USD')
plt.title('Salary Distribution by Sport')
plt.show()

# Scatter: Experience vs Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Years_Experience', y='Salary_USD', hue='Sport', alpha=0.6)
plt.title('Years Experience vs Salary')
plt.show()

# Points vs Salary colored by Sport
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Points_Per_Game', y='Salary_USD', hue='Sport', size='Age', alpha=0.6)
plt.title('Points Per Game vs Salary (sized by Age)')
plt.show()

# Histograms with KDE
df[['Age', 'Points_Per_Game', 'Years_Experience']].hist(figsize=(12, 4), bins=30)
plt.tight_layout()
plt.show()

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Salary_USD', hue='Sport', kde=True)
plt.title('Salary Distribution by Sport')
plt.show()