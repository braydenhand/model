import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your data
df = pd.read_csv('23-24.csv')
df2 = pd.read_csv('22-23.csv')
df3 = pd.read_csv('21-22.csv')
df4 = pd.read_csv('19-20.csv')



# Concatenate the dataframes
df = pd.concat([df, df2,df3,df4], ignore_index=True)
df = df.drop("MATCHUP",axis=1)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Fill missing values with 0 (or you can choose an appropriate value)
df.fillna(0, inplace=True)

wins_df = df[df['W/L'] == 1]
losses_df = df[df['W/L'] == 0]

print("WINS SHEET:")
# Inspect data
print(wins_df.info())
print(wins_df.describe())

print("LOSSES SHEET:")
# Inspect data
print(losses_df.info())
print(losses_df.describe())




# Compute rolling mean of the previous 3 rows
rolling_mean = df.iloc[:, 1:].rolling(window=3).mean()

# Shift the rolling mean down by 1 row
rolling_mean_shifted = rolling_mean.shift(1)

# Replace the original values with the shifted rolling mean
df.iloc[:, 1:] = rolling_mean_shifted
print(df)
# Assuming df is your DataFrame containing the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Assuming the first column is not a feature

# Create a new DataFrame with the scaled values
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[1:])

# Add back the first column if needed
df_scaled.insert(0, 'W/L', df['W/L'])

df = df_scaled

# Inspect data
print(df.info())
print(df.describe())

# Handle missing values (example: fill missing with median)
df.fillna(df.median(), inplace=True)

# Feature engineering (example: add new features if necessary)

# Split the data into features and target
X = df.drop('W/L', axis=1)  # Assuming 'W/L' is the target column
y = df['W/L']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Example: using feature importance from a Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Select the top features
top_features = [X.columns[i] for i in indices[:10]]  # Top 10 features

# Reduce the dataset to top features
X_train = X_train[:, indices[:10]]
X_test = X_test[:, indices[:10]]


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Choose a model and set up hyperparameter tuning
model = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [100, 200, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import joblib

# Save the model
joblib.dump(best_model, 'best_model.pkl')

# Load the model
loaded_model = joblib.load('best_model.pkl')

# Predict with the loaded model
y_pred_loaded = loaded_model.predict(X_test)
