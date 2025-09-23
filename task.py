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

def get_dataset(data_url: str, columns: list | None) -> pd.DataFrame | None:
    '''
    Function to download Boston dataset using direct link.
    
    Arguments
    ---------
    data_url: str
        The link to download the dataset from
    columns: list | None
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

def main():
    '''Main function.'''
    dataset = get_dataset(URL, COLUMNS)
    if dataset is not None:
        dataset = dataset.select_dtypes(include="number")
        cor_mat = dataset.corr()
        sns.heatmap(cor_mat)
        plt.savefig("correlation_matrix.png")
        plt.show()


if __name__ == "__main__":
    main()
