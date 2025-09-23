import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


URL = "https://lib.stat.cmu.edu/datasets/boston"
# Dataset is downloaded without columns, so there is columns for this dataset from the internet.
COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS",
    "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B",
    "LSTAT", "MEDV",
]

def get_dataset(data_url: str, columns: list | None = None) -> pd.DataFrame | None:
    '''
    Function to download Boston dataset using direct link.
    
    Arguments
    ---------
    data_url: str
        The link to download the dataset from
    columns: list | None, default = None
        A list of column names to replace the original dataset columns with
        or None if you don't want to replace columns

    Returns
    -------
    pd.Dataframe | None
        Downloaded Boston dataset or None if there is some exception
    '''
    try:
        # Function sklearn.datasets.load_boston is deprecated since scikit-learn 1.2
        # That is alternative from scikit-learn.org
        # 22 is number from scikit-learn.org, first lines of dataset is incorrectly encoded or something like this
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        res = pd.concat([pd.DataFrame(data), pd.DataFrame(target)], axis=1)
        if columns is not None:
            res.columns = columns
        return res
    except Exception as e:
        print(f"Exception raised during dataset downloading and processing: {e}")
        return None

def calculate_correlation(dataset: pd.DataFrame) -> pd.DataFrame | None:
    '''
    Function to find correlation matrix for numeric columns of given dataset.

    Arguments
    ---------
    dataset: pandas.Dataframe
        The dataset to calculate the correlation for
    
    Returns
    -------
    pd.Dataframe | None
        Calculated correlation matrix or None if there is some exception
    '''
    try:
        dataset = dataset.select_dtypes(include="number")
        if dataset.size != 0: # Check that dataset is not empty after removing non-numeric columns.
            return dataset.corr()
        else:
            print("Warning! Dataset doesn't contain numeric columns!")
            return None
    except Exception as e:
        print(f"Exception raised during correlation calculation: {e}")
        return None

def draw_heatmap(data: pd.DataFrame, save_name: str | None = None) -> None:
    '''
    Function to draw heatmap for given data and, optionally, to save graph.

    Arguments
    ---------
    data: pandas.Dataframe
        The data to draw heatmap for
    save_name: str | None, default = None
        Name of file to which save graph. If save_name == None or "", there will be no saving
    '''
    try:
        sns.heatmap(data)
        if save_name is not None and save_name != "":
            plt.savefig(save_name)
        plt.show()
    except Exception as e:
        print(f"Exception raised during heatmap drawing: {e}")

def main():
    '''Main function.'''
    dataset = get_dataset(URL, COLUMNS)
    if dataset is not None:
        cor_mat = calculate_correlation(dataset)
        draw_heatmap(cor_mat, "corr_mat.png")


if __name__ == "__main__":
    main()
