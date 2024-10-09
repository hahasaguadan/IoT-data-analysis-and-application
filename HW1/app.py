from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64

# Use the 'Agg' backend for matplotlib to avoid GUI issues in Flask
plt.switch_backend('Agg')

app = Flask(__name__)

# Step 1: Generate random data with adjustable parameters
def generate_data(n=100, a=2, b=50, c=10, variance=10):
    np.random.seed(0)
    x = np.random.uniform(-10, 10, n)
    noise = np.random.normal(0, variance, n)
    y = a * x + b + c * noise
    return x, y

# Step 2: Plot regression line and data
def plot_to_img(x, y, a, b, c, variance):
    model = LinearRegression()
    x_reshape = x.reshape(-1, 1)
    model.fit(x_reshape, y)
    y_pred = model.predict(x_reshape)
    
    # Create the plot
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, y_pred, color='red', label='Regression Line')
    plt.title(f"Linear Regression: a={a}, b={b}, c={c}, variance={variance}, n={len(x)}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    # Default parameters or user inputs
    a = request.form.get("a", 2, type=float)
    b = request.form.get("b", 50, type=float)
    c = request.form.get("c", 10, type=float)
    variance = request.form.get("variance", 10, type=float)
    n = request.form.get("n", 100, type=int)  # Number of points (default = 100)
    
    # Generate data and plot
    x, y = generate_data(n=n, a=a, b=b, c=c, variance=variance)
    img = plot_to_img(x, y, a, b, c, variance)
    
    # Render the template with the plot and parameters
    return render_template("index.html", img=img, a=a, b=b, c=c, variance=variance, n=n)

if __name__ == "__main__":
    app.run(debug=True)



