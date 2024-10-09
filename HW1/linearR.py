import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Generate random data
def generate_data(n=100, a=2, b=50, c=10, variance=10):
    np.random.seed(0)
    x = np.random.uniform(-10, 10, n)
    noise = np.random.normal(0, variance, n)
    y = a * x + b + c * noise
    return x, y

# Step 2: Plot data and regression line
def plot_regression(x, y, a, b, c, variance):
    # Fit linear regression
    model = LinearRegression()
    x_reshape = x.reshape(-1, 1)
    model.fit(x_reshape, y)
    
    # Predict y using the regression model
    y_pred = model.predict(x_reshape)
    
    # Plot the original points
    plt.scatter(x, y, color='blue', label='Data Points')
    
    # Plot the regression line
    plt.plot(x, y_pred, color='red', label='Regression Line')
    
    # Show parameters in title
    plt.title(f"Linear Regression: a={a}, b={b}, c={c}, variance={variance}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the plot
x, y = generate_data()
plot_regression(x, y, 2, 50, 10, 10)
