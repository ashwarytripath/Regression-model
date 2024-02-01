import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv('india_economy.csv')

# Assuming 'GDP Growth Rate' as the target variable
target_variable = 'GDP Growth Rate'

# Drop rows with missing values for the target variable
df = df.dropna(subset=[target_variable])

# Extract features and target variable
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'linear_regression_model.joblib')
