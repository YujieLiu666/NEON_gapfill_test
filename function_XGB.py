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
import joblib

print("---------------------------------------")
print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)
import xgboost
print("xgboost version:", xgboost.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("---------------------------------------")

# If you use the package on HPC:
# Limit threads for MKL, OpenBLAS, and BLIS
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['BLIS_NUM_THREADS'] = '1'

#%% load input data
def load_data(site_data_dir, file_name, y_col, plot=True):
    """
    Load site data from a CSV file, optionally plot the original data.

    Parameters:
    - site_data_dir : Path or str
        Directory containing the CSV file.
    - file_name : str
        Name of the CSV file to load (e.g., 'data_for_XGB_US-XYZ.csv').
    - y_col : str
        The name of the column to plot.
    - plot : bool, default True
        If True, generate a plot of the original target column.

    Returns:
    - site_data : DataFrame
        Original site data.
    - site_data_no_na : DataFrame
        Site data after dropping rows with missing values in y_col.
    """
    site_data = pd.read_csv(Path(site_data_dir) / file_name)
    if 'Date' in site_data.columns:
        site_data['Date'] = pd.to_datetime(site_data['Date'])

    # Drop rows with missing target values
    site_data_no_na = site_data.dropna(subset=[y_col])

    # Optional plotting
    if plot:
        plt.figure(figsize=(12, 6))
        if 'Date' in site_data.columns:
            plt.plot(site_data['Date'], site_data[y_col], marker='o', linestyle='None', color='blue', alpha=0.6)
            plt.xlabel('Date')
        else:
            plt.plot(site_data[y_col], marker='o', linestyle='None', color='blue', alpha=0.6)
        plt.ylabel(y_col)
        plt.title(f'{y_col} original data')
        plt.grid(True)
        plt.show()

    return site_data, site_data_no_na

#%% hypterparamter tuning
def find_hyperparameters(site_data_no_na, predictors, y_col, model_dir, n_jobs=10): 
    """
    Find the best hyperparameters for XGBoost using GridSearchCV, 
    and save the model and best parameters.

    Parameters:
    - site_data_no_na: DataFrame. Cleaned data (no NA values in y_col).
    - predictors: list of strings. Column names to be used as predictors (X).
    - y_col: string. Name of the response variable (y).
    - model_dir: Path object or string. Directory to save the model and best parameters.
    - n_jobs: int. Number of parallel jobs for GridSearchCV.
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

    # Base model
    model = XGBRegressor()

    # Set up GridSearchCV
    xgb_grid = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=10,
        verbose=1,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs
    )

    # Perform grid search
    xgb_grid.fit(X, y)

    # Print CV results
    print("Cross-validation scores:")
    cv_results = xgb_grid.cv_results_
    for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
        print(f"Mean Score: {mean_score:.4f}, Parameters: {params}")

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
    model_path = model_dir / f"XGB_model_{y_col}.pkl"
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)
    print(f"\nModel saved to {model_path}.")

#%% do gapfilling
def get_accurate_prediction(site_data, site_data_no_na, predictors, y_col, site_data_dir, reg, plot=False):
    """
    Train an XGBoost model, predict on all data, and save outputs.

    Parameters
    ----------
    site_data : pandas.DataFrame
        Full dataset (may contain missing values in y_col).
    site_data_no_na : pandas.DataFrame
        Cleaned dataset without missing values in y_col (used for training).
    predictors : list of str
        Column names used as predictors (X).
    y_col : str
        Column name of the target variable (y).
    site_data_dir : pathlib.Path
        Directory where output CSV will be saved.
    reg : object
        Regression model with fit/predict methods (e.g., XGBRegressor).
    plot : bool, optional (default=False)
        If True, generate plots of observed vs. predicted values.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with two new columns:
        - 'XGB_FC_fall': model predictions for all rows
        - 'XGB_FC_f': observed values when available, otherwise filled with predictions
    """

    # --- Step 1: Fit model on non-missing data ---
    np.random.seed(42) 
    X = site_data_no_na[predictors]
    y = site_data_no_na[y_col]
    reg.fit(X, y)

    # --- Step 2: Predict on all rows ---
    X_all = site_data[predictors]
    y_pred = reg.predict(X_all)

    # --- Step 3: Save predictions into dataframe ---
    site_data['XGB_FC_fall'] = y_pred
    site_data['XGB_FC_f'] = np.where(
        site_data[y_col].notnull(),
        site_data[y_col],
        site_data['XGB_FC_fall']
    )

    # --- Step 4: Save to CSV ---
    # prediction_file = site_data_dir / "XGB_prediction.csv"
    # site_data.to_csv(prediction_file, index=False)
    # print(f"Predictions saved to: {prediction_file}")

    # --- Step 5 (optional): Plots ---
    if plot:
        # Plot 1: Time series
        plt.figure(figsize=(14, 6))
        plt.scatter(site_data['Date'], site_data[y_col], 
                    label="Observed", s=10, alpha=0.3, color="blue", edgecolors="none")
        plt.scatter(site_data['Date'], site_data['XGB_FC_fall'], 
                    label="Predicted", s=10, alpha=0.3, color="orange", edgecolors="none")
        plt.xlabel("Date")
        plt.ylabel(y_col)
        plt.title(f"Observed vs Predicted over Time for {y_col}")
        plt.legend()
        plt.show()

        # Plot 2: Scatter with 1:1 line
        plt.figure(figsize=(14, 6))
        plt.scatter(site_data['Date'], site_data['XGB_FC_f'], 
                    label=" ", s=10, alpha=0.3, color="green", edgecolors="none")
        plt.xlabel("Date")
        plt.title("Measured + gap-filled time series")
        plt.legend()
        plt.show()

    return site_data

#%% check model performance
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

def check_model_performance(site_data, predictors, y_col, reg, n_folds=10):
    """
    Perform model evaluation with k-fold cross-validation:
    - Shuffle data into k folds (by default n_folds = 10)
    - Train XGBoost (or other estimator) on each fold
    - Plot learning curves for each fold
    - Compute mean and standard deviation of RMSE, R2, and MAPE across folds

    Parameters
    ----------
    site_data : pandas.DataFrame
        Input dataset containing predictors and target variable.
    predictors : list of str
        List of column names in site_data to be used as predictors (features).
    y_col : str
        Column name of the target variable in site_data.
    reg : object
        Regression model object with `fit`, `predict`, and `evals_result` methods 
        (e.g., XGBRegressor).
    n_folds : int, optional (default=10)
        Number of folds for cross-validation.
    """

    # --- Step 1: Shuffle & assign folds ---
    print("Step 1: Shuffle & assign folds ...")
    shuffle_data = site_data.sample(frac=1).reset_index(drop=True)
    shuffle_data['Time'] = np.arange(1, len(shuffle_data) + 1)

    n_rows = len(shuffle_data)
    fold_size = int(np.ceil(n_rows / n_folds))
    shuffle_data['fold_number'] = np.nan

    for i in range(1, n_folds + 1):
        start_idx = (i - 1) * fold_size
        end_idx = min(i * fold_size, n_rows)
        shuffle_data.loc[start_idx:end_idx-1, 'fold_number'] = i

    shuffle_data = shuffle_data.sort_values('Time').reset_index(drop=True)

    # --- Step 2: Train & evaluate for each fold ---
    print("Step 2: Train & evaluate for each fold ...")
    rmse_list, r2_list, mape_list = [], [], []

    for fold_number in range(1, n_folds + 1):
        print(f" Processing Fold {fold_number}...")

        # Split into train/test by fold
        train = shuffle_data.copy()
        train[y_col] = np.where(train['fold_number'] != fold_number, train[y_col], np.nan)
        train = train.dropna(subset=[y_col])

        test = shuffle_data.copy()
        test[y_col] = np.where(test['fold_number'] == fold_number, test[y_col], np.nan)
        test = test.dropna(subset=[y_col])

        X_train, y_train = train[predictors], train[y_col]
        X_test, y_test = test[predictors], test[y_col]

        # Fit model to each fold
        print(f"  Train model for Fold {fold_number}...")
        reg.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        # Learning curve for each fold
        print(f"  Plotting learning curve for Fold {fold_number}...")
        results = reg.evals_result()
        plt.plot(results['validation_0']['rmse'], label='Train RMSE')
        plt.plot(results['validation_1']['rmse'], label='Valid RMSE')
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title(f'Fold {fold_number} Learning Curve')
        plt.show()
        plt.close()

        # Predictions & metrics
        y_pred = reg.predict(X_test)
        scores = compute_performance_metrics(y_test, y_pred)
        rmse_list.append(scores['RMSE'])
        r2_list.append(scores['R2'])
        mape_list.append(scores['MAPE'])

    # --- Step 3: Print summary ---
    print("--------------------------------------------------------")
    print(f"\nMean RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}")
    print(f"Mean R²: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
    print(f"Mean MAPE: {np.mean(mape_list):.3f} ± {np.std(mape_list):.3f}")
    print("--------------------------------------------------------")

#%% check feature importance
def check_feature_importance(site_data_no_na, predictors, y_col, reg):
    """
    Compute and visualize feature importances.

    Parameters
    ----------
    site_data_no_na : pandas.DataFrame
        Input dataset with no missing values.
    predictors : list of str
        List of column names in `site_data_no_na` to be used as predictors (features).
    y_col : str
        Column name of the target variable in `site_data_no_na`.
    reg : object
        Regression model with `fit`, `predict`, and `feature_importances_` attributes
        (e.g., XGBRegressor).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing features and their importance values, sorted in descending order.
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
    feature_importance_df = feature_importance_df.sort_values(
        by='Feature_Importances', ascending=False
    )

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.bar(feature_importance_df['predictors'], feature_importance_df['Feature_Importances'])
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return feature_importance_df
#%% compute annual sums
def cal_FC_annual_sum(data, var_name, start_year, end_year, plot=True):
    """
    Calculate the annual sum of a variable (e.g., FCO2) from half-hourly data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing half-hourly data with a 'Year' column.
    var_name : str
        Column name of the variable to sum.
    start_year : int
        First year to aggregate.
    end_year : int
        Last year to aggregate.
    plot : bool, optional (default=True)
        Whether to plot the annual sums.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Year' and 'annual_sum'.
    """

    # Initialize an empty dataframe to store aggregated data
    agg_data = pd.DataFrame()

    # Define molar mass of CO2 in g/mol
    molar_mass = 12.011

    # Function to get the number of days in a year
    def get_days_in_year(year):
        return 366 if calendar.isleap(year) else 365

    # Iterate over years
    for i in range(start_year, end_year + 1):
        data_sub = data[data['Year'] == i]
        half_hour = data_sub[var_name]

        # Calculate aggregated annual sum
        agg_co2 = half_hour.mean(skipna=True)
        agg = agg_co2 * 1e-6 * molar_mass * 3600 * 24 * get_days_in_year(i)

        agg_data = pd.concat(
            [agg_data, pd.DataFrame({'Year': [i], 'annual_sum': [agg]})],
            ignore_index=True
        )

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(
            agg_data['Year'],
            agg_data['annual_sum'],
            marker='o',
            linestyle='-',
            color='b'
        )
        for x, y in zip(agg_data['Year'], agg_data['annual_sum']):
            plt.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=9)
        plt.xlabel('Year')
        plt.ylabel(r'FCO$_2$ (gC m$^{-2}$ y$^{-1}$)')
        plt.title(f'{var_name} by Year')
        plt.grid(True)
        plt.show()
        
    return agg_data

#%% compute monthly sums
def cal_FC_monthly_sum(df, var_name, start_year, end_year, plot=True):
    """
    Aggregate half-hourly data to monthly sums and optionally plot the results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing half-hourly data with 'Year' and 'Month' columns.
    var_name : str
        Column name of the variable to aggregate (e.g., FCO2).
    start_year : int
        First year to aggregate.
    end_year : int
        Last year to aggregate.
    plot : bool, optional (default=True)
        Whether to generate a plot of the monthly sums.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'Year', 'Month', and 'monthly_sum'.
    """


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

    # Plot the results if requested
    if plot:
        plt.figure(figsize=(10, 6))
        for year in monthly_df['Year'].unique():
            year_data = monthly_df[monthly_df['Year'] == year]
            plt.plot(year_data['Month'], year_data['monthly_sum'], label=f'Year {year}')

        plt.xlabel('Month')
        plt.ylabel(r'FCO$_2$ (gC m$^{-2}$ month$^{-1}$)')
        plt.title(f'{var_name} by Month')
        plt.legend()
        plt.grid(True)
        plt.show()

    return monthly_df

#%% uncertainty analysis
def get_synthetic_data(original_data, n_bootstrap=50, drop_fraction=0.25,
                       sample_size=90200, gapfill_cols=None, random_state=None):
    """
    Generate synthetic datasets using bootstrap sampling and random row dropping.

    Parameters
    ----------
    original_data : pandas.DataFrame
        The original dataset to generate synthetic samples from.
    n_bootstrap : int, optional (default=50)
        Number of synthetic datasets to generate.
    drop_fraction : float, optional (default=0.25)
        Fraction of rows to randomly drop before bootstrap sampling.
    sample_size : int, optional (default=90200)
        Number of rows to sample in each bootstrap dataset.
    gapfill_cols : list of str or str, optional
        Columns to check for missing values. Rows with all NaNs in these columns will be removed.
    random_state : int, optional
        Random seed for reproducibility during bootstrap sampling.

    Returns
    -------
    boot_datasets : list of pandas.DataFrame
        List of synthetic datasets generated from the original data.
    """

    # Set NumPy random seed for reproducibility of uniform random numbers
    np.random.seed(42)

    # List to store all synthetic bootstrap datasets
    boot_datasets = []

    # Loop to generate each synthetic dataset
    for i in range(n_bootstrap):
        # Make a copy of the original data to avoid modifying it
        df = original_data.copy()

        # Add a random column for row-dropping
        df['rand'] = np.random.uniform(size=len(df))

        # Drop rows based on drop_fraction (keep rows where rand >= drop_fraction)
        df = df[df['rand'] >= drop_fraction]

        # Bootstrap sampling: sample 'sample_size' rows with replacement
        df = df.sample(n=sample_size, replace=True, random_state=random_state)

        # If gapfill_cols is specified, ensure it is a list
        if gapfill_cols is not None:
            if isinstance(gapfill_cols, str):
                gapfill_cols = [gapfill_cols]
            # Remove rows where all specified gapfill_cols are NaN
            mask = df[gapfill_cols].isna().all(axis=1)
            df = df.loc[~mask]

        # Remove the temporary 'rand' column
        df = df.drop(columns=['rand'], errors='ignore')

        # Append the processed synthetic dataset to the list
        boot_datasets.append(df)

    # Return all generated synthetic datasets
    return boot_datasets


def train_on_synthetic_predict(site_data, boot_datasets, predictors, y_col):
    """
    Train XGBoost models on bootstrap datasets and predict on original site_data.

    Parameters:
    - site_data: DataFrame. Original data to make predictions on.
    - boot_datasets: list of DataFrames. Each bootstrap dataset used for training.
    - predictors: list of strings. Column names used as features (X).
    - y_col: string. Column name for the response variable (y).

    Returns:
    - predictions: DataFrame. Columns: 'Bootstrap_1', 'Bootstrap_2', ..., with predictions on site_data.
    """
    np.random.seed(42)
    predictions = pd.DataFrame(index=site_data.index)

    for i, boot_df in enumerate(boot_datasets, 1):
        # Prepare training data
        X_train = boot_df[predictors]
        y_train = boot_df[y_col]

        # Train model
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)

        # Predict on original site_data
        X_pred = site_data[predictors]
        predictions[f'Bootstrap_{i}'] = model.predict(X_pred)

    return predictions
#%% end
