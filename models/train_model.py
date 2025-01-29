import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
regdata = pd.read_csv('../data/sampregdata.csv')

# Remove the first column, since it's just an index
regdata = regdata.drop(columns=['Unnamed: 0'])

# Only use x4
best_features = ['x4']

# Train the model using the best feature
model = LinearRegression()
model.fit(regdata[best_features], regdata['y'])


print(f'Model trained using feature: {best_features}')
print(f'Model coefficients: {model.coef_}, Intercept: {model.intercept_}')
