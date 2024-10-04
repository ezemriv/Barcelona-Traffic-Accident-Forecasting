import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.impute import SimpleImputer

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

    def build_ts_features(df, cols_to_group, target_column, lags, aggfunc):

        assert "ds" in df.columns.tolist(), "Date must be in df columns"

        # Error control for lags
        if not isinstance(lags, list):
            raise ValueError("Lags should be a list of integers.")
        
        # Create a new name for columns based on the grouping list
        new_name = "_".join(cols_to_group + [target_column] + [aggfunc.__name__])

        # Set the index and perform grouping, resampling, and aggregation in one step
        grouped_df = (
            df.set_index("ds")
            .groupby(cols_to_group)
            .resample("ME")[target_column]
            .agg(aggfunc)
            .reset_index()
            .rename(
                columns = {
                    target_column : new_name
                }
            )
        )

        # Generate lag and diff features for specified lags
        for n in lags:
                grouped_df[f'{new_name}_lag{n}'] = grouped_df.groupby(cols_to_group)[new_name].shift(n)
        
        print(f"Dropping columns that might cause target leakage {new_name}")
        grouped_df.drop(new_name, inplace = True, axis = 1)

        # Merge the generated features back into the original DataFrame
        merge_cols = ['ds'] + cols_to_group
        df = pd.merge(df, grouped_df, on=merge_cols, how='left')

        return df