import numpy as np
import pandas as pd

from DA_1_07 import create_categorical_feature
from DA_1_07 import load_iris_dataset


def calculate_numeric_column_statistics(df: pd.DataFrame, column_name: str, categoric_col_name: str) -> pd.DataFrame | None:
    '''
    Function to calculate statistics by categories for given column.

    Arguments
    ---------
    df: pandas.Dataframe
        The data to which statistic will be calculated
    column_name: str
        The name of column to which statistics will be calculated
    categoric_col_name: str
        The name of column which contains the categorical data that statistics will be grouped by

    Returns
    -------
    pandas.Dataframe | None
        Dataframe with calculated statistics or None if there is some exception
    '''
    try:
        grouped = df.groupby(categoric_col_name, observed=False)
        agged = grouped[column_name].agg(["mean", "std", "count"])
        agged = agged.add_prefix(column_name + "_", axis="columns")
        return agged.reset_index()
    except Exception as e:
        print(f"Exception raised during statistics calculating: {e}")
        return None

def merge_data_and_stats(df: pd.DataFrame, stats: pd.DataFrame, column_name: str) -> pd.DataFrame | None:
    '''
    Function to merge data and given statistics by common column.

    Arguments
    ---------
    df: pandas.Dataframe
        The data to which statistic will be merged
    stats: pandas.Dataframe
        The statistics which will be merged to data
    column_name: str
        The name of common column by which data and stats will be merged
    
    Returns
    -------
    pandas.Dataframe | None
        Merged dataframe or None if there is some exception
    '''
    try:
        return df.merge(stats, on=column_name, how="left")
    except Exception as e:
        print(f"Exception raised during data and statistics merging: {e}")
        return None

def save_and_visualize(df: pd.DataFrame, save_name: str | None = None) -> None:
    '''
    Function to visualize given Dataframe and, optionally, to save it.

    Arguments
    ---------
    data: pandas.Dataframe
        The data to print it
    save_name: str | None, default = None
        Name of file to which save Dataframe. If save_name == None or "", there will be no saving
    '''
    try:
        print(df.to_string())
        if save_name is not None and save_name != "":
            df.to_csv(save_name)
    except Exception as e:
        print(f"Exception raised during visualizing and saving: {e}")

def main():
    '''Main function.'''
    try:
        data, feature_names, target = load_iris_dataset()
        df = create_categorical_feature(data, feature_names, target)

        stats = calculate_numeric_column_statistics(df, "sepal length (cm)", "species")
        if stats is None:
            return

        res = merge_data_and_stats(df, stats, "species")
        if res is None:
            return
        
        save_and_visualize(res, save_name="iris_with_statistics.csv")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
