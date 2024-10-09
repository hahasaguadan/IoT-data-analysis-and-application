# IoT-data-analysis-and-application

# Flask-based Linear Regression Plot Project(HW1-1)

## Overview

This project solves a linear regression problem using Python, following the CRISP-DM process. The project includes a web application where users can input parameters via sliders and dynamically view the corresponding linear regression plot. The application is built using Flask and the `matplotlib` library for plotting.

## Features

- Users can adjust parameters such as:
  - **Slope (a):** Range from -10 to 10
  - **Intercept (b):** Range from 0 to 100
  - **Noise Factor (c):** Range from 0 to 100
  - **Variance:** Range from 1 to 50
  - **Number of points (n):** Range from 10 to 1000
- The regression plot:
  - Displays data points in blue.
  - Displays the regression line in red.
  - Updates dynamically based on user input via sliders.
  
The project uses a non-interactive `matplotlib` backend (`Agg`) to handle plots, which are then rendered on the webpage as base64-encoded PNG images.

## Project Structure

- `app.py` - The main Flask application. This file handles the user inputs, generates the random data points, fits a linear regression model, and returns a dynamically updated plot.
- `index.html` - The front-end HTML page containing sliders for parameter input and a section to display the plot.

## How to Run the Project

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2. **Navigate to the project directory**:
    ```bash
    cd <project-directory>
    ```

3. **Install the required dependencies**:
    ```bash
    pip install Flask numpy matplotlib scikit-learn
    ```

4. **Run the Flask application**:
    ```bash
    python app.py
    ```

5. **Open the application in a web browser**:
    Open your browser and go to `http://127.0.0.1:5000/` to view the app and interact with the sliders.

## CRISP-DM Steps

1. **Business Understanding**:
   - The task is to visualize a linear regression problem with dynamic parameter control.
   
2. **Data Understanding**:
   - Generate `n` random points in the range of `x` from -10 to 10.
   - The `y` values are calculated as `y = a*x + b + c*N(0, variance)`, where `N(0, variance)` represents random noise from a normal distribution.

3. **Data Preparation**:
   - `n` points are generated based on user inputs.
   
4. **Modeling**:
   - Linear regression is applied to fit a model to the generated points.

5. **Evaluation**:
   - The model's regression line is displayed on the plot.
   
6. **Deployment**:
   - The solution is deployed using a Flask web application.

## Error Fixes & Enhancements

- **RuntimeError: main thread is not in main loop**:
  - This issue was resolved by switching `matplotlib` to the `'Agg'` backend, which is non-interactive and suitable for web applications.
  
- **Allowing user to modify the number of points**:
  - A slider was added to let users control the number of random points (`n`) in the range from 10 to 1000.

## Code Prompt for Version 1

To generate this project in the future, use the following prompt:
    ```bash
Generate a Flask-based Python project that includes two files: app.py and index.html. The project should allow the user to modify parameters for a linear regression plot via sliders, with the plot updating dynamically in real-time. The parameters are:

Slope (a): from -10 to 10
Intercept (b): from 0 to 100
Noise factor (c): from 0 to 100
Variance: from 1 to 50
Number of points (n): from 10 to 1000
The regression plot should show the data points in blue and the regression line in red. Use the matplotlib library to generate the plot, and ensure that it is displayed on the same page using Flask. The page should allow the user to adjust the parameters via sliders and see the plot update immediately.

Here is the structure of the project:

app.py - A Flask application that generates a linear regression plot based on user inputs and sends the plot as a base64 image to the frontend.
index.html - An HTML page that contains sliders for the parameters and displays the plot.
Make sure the app uses a non-interactive matplotlib backend (Agg) to handle plots, and the plot is sent to the frontend as a base64-encoded PNG image.
    ```
