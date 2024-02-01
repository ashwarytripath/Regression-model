import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv('indianEco.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Define features (X) and target variable (y)
X = df.drop(['Year', 'Country Name', 'GDP Growth Rate'], axis=1)
y = df['GDP Growth Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'linear_regression_model.joblib')
