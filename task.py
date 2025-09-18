import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_dataset() -> pd.DataFrame:
    '''Function to download boston dataset using direct link.'''
    # Function sklearn.datasets.load_boston is deprecated since scikit-learn 1.2
    # That is alternative from scikit-learn.org
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # 22 is number from scikit-learn.org, first lines of dataset is incorrectly encoded or something like this
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return pd.concat([pd.DataFrame(data), pd.DataFrame(target)], axis=1)

def main():
    '''Main function.'''
    dataset = get_dataset()
    cor_mat = dataset.corr()
    sns.heatmap(cor_mat)
    plt.show()


if __name__ == "__main__":
    main()
