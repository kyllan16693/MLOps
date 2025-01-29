import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
regdata = pd.read_csv('../data/sampregdata.csv')
regdata = regdata.drop(columns=['Unnamed: 0'])  # Remove index column

# Prepare features
X = regdata[['x2', 'x4']]
y = regdata['y']

# Make a linear regression model
model = LinearRegression()
model.fit(X, y)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(regdata['x2'], y, color='blue', label='Actual y', alpha=0.5)
plt.scatter(regdata['x2'], model.predict(X), color='red', label='Predicted y', alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('x2')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig('model_predictions.png')  # Save the plot as a PNG file
plt.show() 