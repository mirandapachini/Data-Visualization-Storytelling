import os
import numpy as np
import pandas as pd

#os.listdir()

df = pd.read_excel('Retention Rates U of L.xlsx')
print(df.shape)
print(df.describe(include='all'))
print('Available Columns:', df.columns.tolist())

# Counts & percentages of missing values
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)

missing_summary = (
    pd.DataFrame({"n_missing": missing_counts, "pct_missing": missing_pct})
    .sort_values("pct_missing", ascending=False)
)

print(df[['Category', 'entry_major']].value_counts())# summary stats
print(df[['Category', 'entry_major']].describe())


# Top 15 columns with most missing
missing_summary

# Summary stats for retention at the unit level
print("Unit Retention (1to2):")
print(df['ret_1to2_unit'].describe())

print("\nUnit Retention (1to3):")
print(df['ret_1to3_unit'].describe())

# Summary stats for retention at UofL
print("\nUofL Retention (1to2):")
print(df['ret_1to2_uofl'].describe())

print("\nUofL Retention (1to3):")
print(df['ret_1to3 uofl'].describe())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


# 1. Select features and target
target = 'ret_1to2_unit'

# Drop non-predictive or ID-like columns
features = df.drop(columns=[target])

# Convert categorical variables to dummies
features = pd.get_dummies(features, drop_first=True)

# Align target
y = df[target]
X = features

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build the Random Forest model
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rf.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# 5. Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:")
print(importances.head(10))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


# 1. Select features and target
target = 'ret_1to2_uofl'

# Drop non-predictive or ID-like columns
features = df.drop(columns=[target])

# Convert categorical variables to dummies
features = pd.get_dummies(features, drop_first=True)

# Align target
y = df[target]
X = features

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Build the Random Forest model
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

rf.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# 5. Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:")
print(importances.head(10))

import matplotlib.pyplot as plt

# --- Unit Retention ---
plt.figure(figsize=(6,4))
plt.bar(['1→2 Unit', '1→3 Unit'],
        [df['ret_1to2_unit'].mean(), df['ret_1to3_unit'].mean()],
        color=['#4C72B0', '#55A868'])

plt.title('Average Unit Retention')
plt.ylabel('Retention Rate')
plt.ylim(0,1)
plt.show()

# --- UofL Retention ---
plt.figure(figsize=(6,4))
plt.bar(['1→2 UofL', '1→3 UofL'],
        [df['ret_1to2_uofl'].mean(), df['ret_1to3 uofl'].mean()],
        color=['#C44E52', '#8172B2'])

plt.title('Average UofL Retention')
plt.ylabel('Retention Rate')
plt.ylim(0,1)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

labels = ['1→2', '1→3']
unit_means = [df['ret_1to2_unit'].mean(), df['ret_1to3_unit'].mean()]
uofl_means = [df['ret_1to2_uofl'].mean(), df['ret_1to3 uofl'].mean()]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, unit_means, width, label='Unit Retention', color='#4C72B0')
plt.bar(x + width/2, uofl_means, width, label='UofL Retention', color='#C44E52')

plt.xticks(x, labels)
plt.ylabel('Retention Rate')
plt.title('Unit vs. UofL Retention')
plt.ylim(0,1)
plt.legend()
plt.show()

import matplotlib.pyplot as plt

labels = ['1→2 Unit', '1→3 Unit', '1→2 UofL', '1→3 UofL']
values = [
    df['ret_1to2_unit'].mean(),
    df['ret_1to3_unit'].mean(),
    df['ret_1to2_uofl'].mean(),
    df['ret_1to3 uofl'].mean()
]

plt.figure(figsize=(8,5))
plt.bar(labels, values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])

plt.title('Average Retention Rates')
plt.ylabel('Retention Rate')
plt.ylim(0,1)
plt.xticks(rotation=20)
plt.show()

df.groupby('entry_major')['ret_1to2_unit'].mean().sort_values()
df.groupby('entry_major')['cohort_count'].mean()
df.groupby('entry_major')['ret_1to2_unit'].mean().sort_values()
df.groupby('entry_major')['ret_1to3_unit'].mean().sort_values()
