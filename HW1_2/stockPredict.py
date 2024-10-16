# CRISP-DM Step 1: Business Understanding
# Objective: Predict stock prices using multiple linear regression to help in making informed trading decisions.

# CRISP-DM Step 2: Data Understanding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression, SelectKBest, RFE, f_regression
import joblib
from datetime import timedelta

# Load the dataset
# Add thousands separator handling to avoid conversion errors
data = pd.read_csv('2330-training.csv', thousands=',')

# Drop any rows with missing values
data.dropna(inplace=True)

# Sort the dataset by date to ensure correct time series order
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

# Flask setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    columns = ['x1', 'x2', 'x3', 'x4', 'x5']
    selected_features = columns
    rmse, r2, plot_url, feature_plot_url = None, None, None, None
    all_results = []
    best_combination = None
    best_mse = float('inf')

    if request.method == 'POST':
        # Get selected features from the form
        selected_features = request.form.getlist('features')
        
        # Define the features and target variable
        X = data[selected_features]
        y = data['y']
        dates = pd.to_datetime(data['Date'])  # Convert dates for plotting

        # Ensure data is converted to numeric for modeling
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        # Drop any rows with NaN values after conversion
        valid_indices = X.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        dates = dates.loc[valid_indices]

        # Generate all possible feature combinations
        for r in range(1, len(selected_features) + 1):
            for combination in combinations(selected_features, r):
                X_subset = X[list(combination)]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_subset, y, dates, test_size=0.2, random_state=42)

                # Sort test data by date to maintain time series order
                dates_test, X_test, y_test = zip(*sorted(zip(dates_test, X_test.values, y_test)))
                dates_test = pd.Series(dates_test)
                X_test = pd.DataFrame(X_test, columns=X_subset.columns)
                y_test = pd.Series(y_test)

                # Create and train the multiple linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)

                # Calculate evaluation metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Store results
                all_results.append((combination, rmse, r2))

                # Update best combination
                if rmse < best_mse:
                    best_mse = rmse
                    best_combination = combination

        # Plot the stock price chart for the best combination
        X_best = X[list(best_combination)]
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_best, y, dates, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Sort test data by date to maintain time series order
        dates_test, y_test, y_pred = zip(*sorted(zip(dates_test, y_test, y_pred)))
        dates_test = pd.Series(dates_test)
        y_test = pd.Series(y_test)
        y_pred = pd.Series(y_pred)

        # Extend the prediction to the end of October
        last_date = dates_test.iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), end='2024-10-12')
        future_X = X_best.iloc[-1:].values  # Use the last known values as a base for future predictions

        future_y_pred = []
        for future_date in future_dates:
            future_pred = model.predict(future_X)
            future_y_pred.append(future_pred[0])
            future_X = np.roll(future_X, -1)
            future_X[0, -1] = future_pred[0]

        # Append future predictions to y_pred and dates_test
        y_pred = pd.concat([y_pred, pd.Series(future_y_pred)], ignore_index=True)
        dates_test = pd.concat([dates_test, pd.Series(future_dates)], ignore_index=True)

        # Plot the stock price chart
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test[:len(y_test)], y_test.values, label='Actual Prices', color='blue')
        plt.plot(dates_test, y_pred, label='Predicted Prices', color='red')
        plt.axvline(x=last_date, color='black', linestyle='--', label='Last Date of Dataset')  # Add vertical line for last date of dataset
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Actual vs Predicted Stock Prices (Best Combination)')
        plt.legend()
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(30))  # Set x-axis to display monthly intervals
        plt.xticks(rotation=45)
        
        # Save plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

        # Feature Selection RMSE Plot
        feature_counts = list(range(1, len(columns) + 1))
        rmse_results = {
            'Mutual Information': [],
            'Recursive Feature Elimination': [],
            'SelectKBest': []
        }

        # Mutual Information
        for k in feature_counts:
            mi_selector = SelectKBest(mutual_info_regression, k=k)
            X_new = mi_selector.fit_transform(X, y)
            if X_new.shape[1] > 0:
                X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_results['Mutual Information'].append(rmse)

        # Recursive Feature Elimination
        for k in feature_counts:
            rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=k)
            X_new = rfe_selector.fit_transform(X, y)
            if X_new.shape[1] > 0:
                X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_results['Recursive Feature Elimination'].append(rmse)

        # SelectKBest with f_regression
        for k in feature_counts:
            skb_selector = SelectKBest(f_regression, k=k)
            X_new = skb_selector.fit_transform(X, y)
            if X_new.shape[1] > 0:
                X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_results['SelectKBest'].append(rmse)

        # Plot the RMSE results
        plt.figure(figsize=(10, 6))
        plt.plot(feature_counts, rmse_results['Mutual Information'], marker='o', linestyle='-', label='Mutual Information')
        plt.plot(feature_counts, rmse_results['Recursive Feature Elimination'], marker='^', linestyle='-', label='Recursive Feature Elimination')
        plt.plot(feature_counts, rmse_results['SelectKBest'], marker='s', linestyle='-', label='SelectKBest')
        plt.xlabel('Number of Features')
        plt.ylabel('RMSE')
        plt.title('RMSE for Different Feature Selection Algorithms')
        plt.legend()
        plt.grid(True)
        
        # Save feature selection plot to a string buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        feature_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')

    return render_template('index.html', columns=columns, selected_features=selected_features, rmse=best_mse, r2=r2, plot_url=plot_url, all_results=all_results, best_combination=best_combination, feature_plot_url=feature_plot_url)

if __name__ == '__main__':
    app.run(debug=True)

