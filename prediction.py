import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Prepare the data (scikit-learn expects X to be a 2D array)
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11])

# 2. Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)

# 3. Print the results
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# 4. Make a prediction for a new x value (e.g., x = 6)
new_x = np.array([[6]])
print(f"Prediction for x=6: {model.predict(new_x)[0]}")
