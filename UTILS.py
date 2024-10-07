import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.impute import SimpleImputer

# for map rendering
import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from IPython.display import Image
import time
import os

# for feature engineering
from mlforecast import MLForecast
from window_ops.rolling import rolling_mean, rolling_std
from mlforecast.lag_transforms import ExponentiallyWeightedMean, ExpandingMean, ExpandingStd

class InitialEDA:
    """Class for performing exploratory data analysis (EDA) on a DataFrame."""
    
    @staticmethod
    def drop_notinformative(df, threshold=0.95):
        """Drop columns with high percentage of one unique value."""
        return df.loc[:, df.nunique(normalize=True) < threshold]
    
    @staticmethod
    def get_categorical(df):
        """Identifies categorical columns in a DataFrame."""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    @staticmethod
    def get_numeric(df):
        """Identifies numeric columns in a DataFrame."""
        return df.select_dtypes(include=['number']).columns.tolist()

    @staticmethod
    def get_possible_bool(df):
        """Identifies columns that could be boolean (two unique values)."""
        return df.columns[df.nunique() == 2].tolist()

    @staticmethod
    def plot_histograms(df):
        """Plots histograms for numeric variables in the DataFrame."""
        numeric_cols = InitialEDA.get_numeric(df)
        df[numeric_cols].hist(bins=30, figsize=(15, 10))
        plt.suptitle('Histograms of Numeric Variables')
        plt.show()
    
    @staticmethod
    def plot_barplots_normalized(df, exclude: list[str] = []):
        """Plots horizontal bar plots of normalized value counts for categorical variables in the DataFrame."""
        
        categorical_cols = InitialEDA.get_categorical(df)
        categorical_cols = [col for col in categorical_cols if col not in exclude]

        # Set number of rows and columns for subplots
        n_cols = 2
        n_rows = math.ceil(len(categorical_cols) / n_cols)
        
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, len(categorical_cols) * 3))
        axs = axs.flatten()  # Flatten in case we have multiple rows and columns

        for i, col in enumerate(categorical_cols):
            sns.barplot(x=df[col].value_counts(normalize=True), 
                        y=df[col].value_counts(normalize=True).index, 
                        ax=axs[i])
            axs[i].set_title(f'Normalized Value Counts of {col}')
            axs[i].set_xlabel('Normalized Value Counts')
            axs[i].set_ylabel('Categories')

        # Hide any extra axes
        for i in range(len(categorical_cols), len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def render_folium_map(map_object, html_path, png_path, width=1000, height=800, wait_time=5):
        """
        Render a Folium map as a static image and save both HTML and PNG versions.
        
        Parameters:
        - map_object: A Folium map object
        - html_path: Path to save the HTML version of the map
        - png_path: Path to save the PNG version of the map
        - width: Width of the browser window (default 1000)
        - height: Height of the browser window (default 800)
        - wait_time: Time to wait for the map to render in seconds (default 5)
        
        Returns:
        - Displays the PNG image in the notebook
        """
        # Save the map to the specified HTML file
        map_object.save(html_path)
        
        # Set up the web driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        
        # Open the HTML file
        driver.get("file://" + os.path.abspath(html_path))
        
        # Wait for the map to render
        time.sleep(wait_time)
        
        # Set the size of the browser window
        driver.set_window_size(width, height)
        
        # Capture the screenshot
        driver.save_screenshot(png_path)
        
        # Close the browser
        driver.quit()
        
        # Display the image in the notebook
        display(Image(png_path))
        
        print(f"Check out interactive HTML map at: {html_path}")
        print(f"PNG image saved to: {png_path}")

class PreprocessingStarter(InitialEDA):
    """Class for data preprocessing tasks, inheriting from InitialEDA."""
    
    @staticmethod
    def drop_columns_nan(dataframe, threshold: int) -> pd.DataFrame:
        """Drops columns from a DataFrame that contain a certain amount of missing values (NaNs)."""
        return dataframe.dropna(axis=1, thresh=len(dataframe) - threshold)
    
    @staticmethod
    def impute_nan_custom(df, method_numerical='mean', value_categorical='Non-existent'):
        """Imputes missing values in a DataFrame with user-defined options."""
        numerical_cols = PreprocessingStarter.get_numeric(df)
        categorical_cols = PreprocessingStarter.get_categorical(df)
        
        # Impute numerical columns
        num_imputer = SimpleImputer(strategy=method_numerical)
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        
        # Impute categorical columns
        df[categorical_cols] = df[categorical_cols].fillna(value=value_categorical)
        
        return df

    @staticmethod
    def replace_less_frequent(df: pd.DataFrame, list_col: list[str], threshold: float = 0.02, new_value='other') -> pd.DataFrame:
        """Replaces less frequent values in specified columns of a DataFrame with a new value."""
        for col in list_col:
            freq = df[col].value_counts(normalize=True)
            to_replace = freq[freq < threshold].index
            df[col].replace(to_replace, new_value, inplace=True)
        return df
    

class FeatureEngineering():
    
    @staticmethod
    def recreate_features(df):
        df = df.copy()
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['namedayweek'] = df['date'].dt.day_name()  # Adding name of the day of the week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['shift'] = df['unique_id'].str.split('-').str[0]
        df['district'] = df['unique_id'].str.split('-').str[1]
        
        # cast categorical columns
        cat_cols = ['namedayweek', 'shift', 'district']

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        feature_cols = [
            'day', 
            'dayofweek', 
            'namedayweek', 
            'month', 
            'quarter', 
            'shift', 
            'district']
        
        return df, cat_cols, feature_cols

    @staticmethod
    def add_aditional_features(df):
        df = df.copy()

        holidays_2023 = [
        '2023-01-01', '2023-01-06', '2023-04-07', '2023-04-10', '2023-05-01', '2023-06-24',
        '2023-08-15', '2023-09-11', '2023-09-25', '2023-10-12', '2023-11-01', '2023-12-06',
        '2023-12-08', '2023-12-25', '2023-12-26'
        ]

        # Convert holiday list to datetime format
        holidays_2023 = pd.to_datetime(holidays_2023)

        # Create 'holiday' column with 1 for holidays and 0 for non-holidays
        df['holiday'] = df['date'].apply(lambda x: 1 if pd.to_datetime(x) in holidays_2023 else 0)
        df['weekend'] = np.where(df['dayofweek'].isin([5, 6]), 1, 0)

        # Transform 'month' into its sine and cosine representations
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Transform 'day' of the month into its sine and cosine representations
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        new_features = [
                        'holiday', 
                        'weekend', 
                        'month_sin', 'month_cos',
                        'day_sin', 'day_cos'
                        ]
        
        return df, new_features
    
    @staticmethod
    def ML_forecast_target_lags(df, unique_id, date_col, target_col):
        """
        Function to create lag features and apply specified transforms for a time series dataframe.
        
        Parameters:
        df (pd.DataFrame): Input dataframe.
        unique_id (str): Name of the unique identifier column.
        date_col (str): Name of the date column.
        target_col (str): Name of the target column.

        Returns:
        pd.DataFrame: Transformed dataframe with lag features and specified transforms.
        """

        # Create a copy of the dataframe and rename columns for compatibility
        df_copy = df.copy().rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Sort the dataframe by unique_id and date columns
        df_sorted = df_copy.sort_values(by=[unique_id, 'ds'])

        # Define the lag transforms for lags 1, 2, 7, 30
        lag_transforms = {
            1: [
                (rolling_mean, 3),
                (rolling_std, 3),
                (rolling_mean, 7),
                (rolling_std, 7),
                (rolling_mean, 30),
                (rolling_std, 30),
                ExponentiallyWeightedMean(alpha=0.5),
                ExponentiallyWeightedMean(alpha=0.8),
            ],
            2: [
                (rolling_mean, 3),
                (rolling_std, 3),
                (rolling_mean, 7),
                (rolling_std, 7),
                (rolling_mean, 30),
                (rolling_std, 30),
            ],
            7: [
                (rolling_mean, 3),
            ],
            30: [
                (rolling_mean, 3),
            ]
        }

        # Initialize the MLForecast object with the frequency set to daily (D) and lags for 1, 2, 7, 30 days
        ml_forecast = MLForecast(
            models=[],  # No models needed for just feature creation
            freq='D',   # Daily frequency
            lags=[1, 2, 7, 30],  # Specify the lags
            lag_transforms=lag_transforms
        )

        # Preprocess the data and create lag features with transforms
        df_with_lags = ml_forecast.preprocess(df_sorted, id_col=unique_id, 
                                              time_col='ds', target_col='y', 
                                              dropna=False, static_features=[])

        return df_with_lags.rename(columns={'ds': date_col, 'y': target_col})
    
    @staticmethod
    def build_group_features(df, cols_to_group, target_column, lags, aggfunc):
        if not isinstance(lags, list):
            raise ValueError("Lags should be a list of integers.")

        new_name = "_".join(cols_to_group + [target_column, aggfunc])

        grouped_df = (
            df.set_index("date")
            .groupby(cols_to_group, observed=False)
            .resample("D")[target_column]
            .agg(aggfunc)
            .reset_index()
            .rename(columns={target_column: new_name})
        )

        new_group_cols = []
        for n in lags:
            col_name = f'{new_name}_lag{n}'
            grouped_df[col_name] = grouped_df.groupby(cols_to_group, observed=False)[new_name].shift(n)
            new_group_cols.append(col_name)

        print(f"Dropping columns that might cause target leakage {new_name}")
        grouped_df.drop(columns=new_name, inplace=True)

        merge_cols = ['date'] + cols_to_group
        df = pd.merge(df, grouped_df, on=merge_cols, how='left')

        return df, new_group_cols