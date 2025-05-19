import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import sklearn
from pathlib import Path
import pickle as pkl
import sys
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import csv
from sklearn.metrics import mean_squared_error, r2_score
import calendar

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
import xgboost
print("xgboost version:", xgboost.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)

# Limit threads for MKL, OpenBLAS, and BLIS
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['BLIS_NUM_THREADS'] = '1'

# load data 
def load_data(site_data_dir, file_name, y_col):
    """
    Load site data from a CSV file, plot the original and NA-removed data.

    Parameters:
    - site_data_dir: Path object or string. Directory containing the CSV file.
    - file_name: string. Name of the CSV file to load (e.g., 'data_for_XGB_US-XYZ.csv').
    - y_col: string. The name of the column to plot.

    Returns:
    - site_data: DataFrame containing the original site data.
    - site_data_no_na: DataFrame containing the site data after dropping rows with missing values in y_col.
    """
    # Load the data
    site_data = pd.read_csv(Path(site_data_dir) / file_name)
    site_data['Date'] = pd.to_datetime(site_data['Date'])
    

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.plot(site_data['Date'], site_data[y_col].values, marker='o', linestyle='-', color='b')
    plt.title(f'Plot of {y_col} against Date (original data)')
    plt.xlabel('Date')
    plt.ylabel(y_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

    # Drop rows with missing values in target variable: y_col
    site_data_no_na = site_data.dropna(subset=[y_col])
    return site_data, site_data_no_na

# find hyperparameters
def find_hyperparameters(site_data_no_na, predictors, y_col, model_dir):
    """
    Find the best hyperparameters for XGBoost using GridSearchCV, and save the model and best parameters.

    Parameters:
    - site_data_no_na: DataFrame. Cleaned data (no NA values in y_col).
    - subset_columns: list of strings. Column names to be used as predictors (X).
    - y_col: string. Name of the response variable (y).
    - model_dir: Path object or string. Directory to save the model and best parameters.

    Returns:
    - reg.model: trained XGBRegressor model with the best parameters.
    """
    
    # Define hyperparameter search space
    parameters = {
        "objective": ["reg:squarederror"],
        "learning_rate": [0.00001, 0.001, 0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "min_child_weight": [3, 5, 7],
        "subsample": [0.6, 0.8],
        "reg_lambda": [0, 0.1, 1, 10],
        "reg_alpha": [0, 0.1, 1, 10],
        "n_estimators": [50, 100, 250]
    }
    
    # Prepare training data
    X = site_data_no_na[predictors]
    y = site_data_no_na[y_col]

    # Create base model
    model = XGBRegressor()

    # Set up GridSearchCV
    xgb_grid = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=10,
        verbose=1,
        scoring='neg_mean_squared_error',
        n_jobs = 10)

    # Perform grid search
    xgb_grid.fit(X, y)

    # Print cross-validation results
    print("Cross-validation scores:")
    cv_results = xgb_grid.cv_results_
    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(f"Mean Score: {mean_score}, Parameters: {params}")

    # Extract best parameters
    best_params = xgb_grid.best_params_
    print("\nBest Parameters Found:")
    print(best_params)

    # Create a new model using the best parameters
    model = XGBRegressor(
        random_state=42, booster='gbtree', tree_method='hist',
        **best_params
    )
    print("\nBest Model:")
    print(model)

    # Save model object as .pkl
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)  # Make sure model_dir exists
    model_path = model_dir / "FC_XGB_model.pkl"
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)
    print(f"\nModel saved to {model_path}.")

    # Save best hyperparameters to CSV
    csv_file = model_dir / "FC_XGB_model.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])  # Write header
        for param, value in best_params.items():
            writer.writerow([param, value])
    print(f'Best parameters saved to {csv_file}.')
    return xgb_grid.best_params_
  
# get accurate prediction
def get_accurate_prediction(site_data, site_data_no_na, predictors, y_col, site_data_dir, reg):
    """
    Train an XGBoost model, predict on all data, and save outputs.
    
    """
    # Fit model on flux data without NA
    X = site_data_no_na[predictors]
    y = site_data_no_na[y_col]
    reg.fit(X, y)
    # Predict on all data
    X_all = site_data[predictors]
    y_pred = reg.predict(X_all)
    # Save predictions into original dataframe
    site_data['XGB_FC_fall'] = y_pred
    site_data['XGB_FC_f'] = np.where(site_data[y_col].notnull(), site_data[y_col], site_data['XGB_FC_fall'])
    # Save full dataframe with predictions
    prediction_file = site_data_dir / "FC_XGB_prediction.csv"
    site_data.to_csv(prediction_file, index=False)
    print(f"Predictions saved to: {prediction_file}")
    return site_data  # return the dataframe with predictions

# model performance
def compute_performance_metrics(truth, prediction):
    """
    Compute RMSE, R², and MAPE between true and predicted values.
    """
    # Ensure inputs are numpy arrays
    truth = np.array(truth)
    prediction = np.array(prediction)

    # RMSE
    rmse = np.sqrt(mean_squared_error(truth, prediction))

    # R²
    r2 = r2_score(truth, prediction)

    # MAPE (handle division by zero carefully)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((truth - prediction) / truth)
        mape = mape[~np.isinf(mape)]  # Remove infinities (from division by zero)
        mape = np.nanmean(mape) * 100  # Convert to percentage

    metrics = {
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    print(metrics)
    return metrics

def create_train_test_folds(site_data, site_data_dir, y_col):
    """
    Shuffle the site data, assign fold numbers, and save 10 train/test datasets.
    """
    # Create FC_data_train_test directory
    FC_dir = Path(site_data_dir) / "FC_data_train_test"
    FC_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle the data
    shuffle_data = site_data.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffle_data['Time'] = np.arange(1, len(shuffle_data) + 1)
    
    # Assign fold numbers
    n_rows = len(shuffle_data)
    fold_size = int(np.ceil(n_rows / 10))
    shuffle_data['fold_number'] = np.nan
    
    for i in range(1, 11):
        start_idx = (i - 1) * fold_size
        end_idx = min(i * fold_size, n_rows)
        shuffle_data.loc[start_idx:end_idx-1, 'fold_number'] = i

    # Sanity check
    if shuffle_data['fold_number'].isna().sum() > 0:
        print("Warning: Some rows have NA fold numbers!")

    # Sort back by Time
    shuffle_data = shuffle_data.sort_values('Time').reset_index(drop=True)
    
    # Save train/test for each fold
    for fold_number in range(1, 11):
        train = shuffle_data.copy()
        train[y_col] = np.where(train['fold_number'] != fold_number, train[y_col], np.nan)

        test = shuffle_data.copy()
        test[y_col] = np.where(test['fold_number'] == fold_number, test[y_col], np.nan)

        # Save files
        train.to_csv(FC_dir / f"train{fold_number}.csv", index=False)
        test.to_csv(FC_dir / f"test{fold_number}.csv", index=False)

    print(f"Train/test files saved in {FC_dir}")
    
def check_model_performance(data_train_test_dir, predictors, y_col, reg):
    """
    Train an XGBoost model with 10-fold evaluation, plot learning curves, and compute model performance metrics on test set.
    """

    # Initialize lists to store performance metrics for each fold
    rmse_list = []
    r2_list = []
    mape_list = []

    for i in range(1, 11):
        print(f"Processing Fold {i}...")

        # Load train and test sets
        train = pd.read_csv(data_train_test_dir / f"train{i}.csv", index_col=0).dropna(subset=[y_col])
        test = pd.read_csv(data_train_test_dir / f"test{i}.csv", index_col=0).dropna(subset=[y_col])

        X_train = train[predictors]
        y_train = train[y_col]
        X_test = test[predictors]
        y_test = test[y_col]

        # Fit model with early stopping
        reg.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        # Plot learning curve
        results = reg.evals_result()
        # plt.figure(figsize=(8, 5))
        plt.plot(results['validation_0']['rmse'], label='Train RMSE')
        plt.plot(results['validation_1']['rmse'], label='Valid RMSE')
        plt.xlabel('Number of iterations')
        plt.ylabel('Root Mean Squared Error (RMSE)')
        plt.legend()
        plt.title(f'Training and Validation RMSE (Fold {i})')
        # plt.grid(True)
        plt.show()
        # plt.savefig(learning_curve_dir / f'FC_learning_curve{i}.png')
        plt.close()
        # print(f"Saved learning curve for Fold {i}")

        # Evaluate performance on test set
        y_test_pred = reg.predict(X_test)
        scores = compute_performance_metrics(y_test, y_test_pred)

        # Append metrics for this fold to the lists
        rmse_list.append(scores['RMSE'])
        r2_list.append(scores['R2'])
        mape_list.append(scores['MAPE'])

    # Compute the mean and standard deviation of metrics
    mean_rmse = np.mean(rmse_list)
    sd_rmse = np.std(rmse_list)
    mean_r2 = np.mean(r2_list)
    sd_r2 = np.std(r2_list)
    mean_mape = np.mean(mape_list)
    sd_mape = np.std(mape_list)

    # Print results
    print("--------------------------------------------------------")
    print(f"\nMean RMSE: {mean_rmse:.3f} ± {sd_rmse:.3f}")
    print(f"Mean R²: {mean_r2:.3f} ± {sd_r2:.3f}")
    print(f"Mean MAPE: {mean_mape:.3f} ± {sd_mape:.3f}")
    print("--------------------------------------------------------")
    
# feature importance
def feature_importance(site_data_no_na, predictors, y_col, site_data_dir, reg):
    """
    Compute feature importances and plot the results.
    """
    # Prepare the data
    X = site_data_no_na[predictors]
    y = site_data_no_na[y_col]
    
    # Fit the regression model
    reg.fit(X, y)

    # Create a DataFrame to store the feature importances
    feature_importance_df = pd.DataFrame()
    feature_importance_df['Feature_Importances'] = reg.feature_importances_
    feature_importance_df['predictors'] = predictors

    # Sort by feature importances
    feature_importance_df = feature_importance_df.sort_values(by='Feature_Importances', ascending=False)

    # Save feature importances to CSV
    feature_importance_df.to_csv(site_data_dir / "FC_feature_importances.csv", index=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance_df['predictors'], feature_importance_df['Feature_Importances'])
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # # Save the plot as PNG file
    # plt.savefig("FC_feature_importances.png")
    plt.show()

    return feature_importance_df


def cal_annual_sum(data, var_name, start_year, end_year, site_data_dir):
    """
    Calculate the annual sum of a variable (e.g., CO2) from half-hourly data.
    """

    # Initialize an empty dataframe to store aggregated data
    agg_data = pd.DataFrame()

    # Define molar mass of CO2 in g/mol
    molar_mass = 12.011

    # Define a function to get the number of days in a year
    def get_days_in_year(year):
        return 366 if calendar.isleap(year) else 365

    # Iterate over years from start_year to end_year
    for i in range(start_year, end_year + 1):
        # Subset data for the current year
        data_sub = data[data['Year'] == i]
        
        # Extract data variable
        half_hour = data_sub[var_name]
        
        # Calculate mean CO2 value
        agg_co2 = half_hour.mean(skipna=True)
        
        # Calculate aggregated value
        agg = agg_co2 * 1e-6 * molar_mass * 3600 * 24 * get_days_in_year(i)
        
        # Concatenate to agg_data dataframe
        agg_data = pd.concat(
            [agg_data, pd.DataFrame({'Year': [i], 'annual_sum': [agg]})],
            ignore_index=True
        )

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(agg_data['Year'], agg_data['annual_sum'], marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Annual Sum')
    plt.title('Annual Sum')
    plt.grid(True)
    plt.show()

    return agg_data

def cal_monthly_sum(df, var_name, start_year, end_year, site_data_dir):
    """
    Aggregate half-hourly data to monthly sums and plot the results.
    """
    import pandas as pd
    import calendar
    import matplotlib.pyplot as plt

    print("Check if you have 'Month' and 'Year' columns in input data!")

    # Initialize an empty DataFrame to store monthly sums
    monthly_df = pd.DataFrame(columns=['Year', 'Month', 'monthly_sum'])

    # Define molar mass of CO2 in g/mol
    molar_mass = 12.011

    # Define a function to get the number of days in a month
    def get_days_in_month(year, month):
        return calendar.monthrange(year, month)[1]

    # Filter data for the specified range of years
    df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

    # Iterate over each year and month
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            df_sub = df[(df['Year'] == year) & (df['Month'] == month)]

            # Extract data variable
            half_hour = df_sub[var_name]

            # Calculate mean CO2 value
            agg_co2 = half_hour.mean(skipna=True)

            # Calculate aggregated value
            days_in_month = get_days_in_month(year, month)
            agg = agg_co2 * 1e-6 * molar_mass * 3600 * 24 * days_in_month

            # Concatenate result
            monthly_df = pd.concat(
                [monthly_df, pd.DataFrame({'Year': [year], 'Month': [month], 'monthly_sum': [agg]})],
                ignore_index=True
            )

    # Plot the results
    plt.figure(figsize=(10, 6))
    for year in monthly_df['Year'].unique():
        year_data = monthly_df[monthly_df['Year'] == year]
        plt.plot(year_data['Month'], year_data['monthly_sum'], label=f'Year {year}')

    plt.xlabel('Month')
    plt.ylabel(var_name)
    plt.title(f'{var_name} by Month')
    plt.legend()
    plt.grid(True)
    plt.show()

    return monthly_df
