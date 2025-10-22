import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from DA_2_41 import moving_average
from DA_2_45 import generate_time_series


SEASONALITY_PERIOD = 33
NUM_PERIODS = 99
VALUE_NAME = "Rubles"

def get_dataframe(start_date: str, num_periods: int, seasonality_period: int, value_name: str, shift: float):
    '''
    Function to get dataframe.

    Arguments
    ---------
    start_date: str
        The date on which the dataframe data will start
    num_periods: int
        The number of periods in data
    seasonality_period: int
        The period of one season
    value_name:
        The name of generated value
    shift:
        The offset by which the data will be shifted
    
    Returns
    -------
    pandas.Dataframe
        Generated dataframe
    '''
    df = generate_time_series(
        start_date=start_date,
        num_periods=num_periods,
        seasonality_period=seasonality_period,
    )
    if df is None:
        return None
    df.rename(columns={df.columns[0]: value_name}, inplace=True)
    return df + shift

def decompose(df: pd.DataFrame, seasonality_period: int, value_name: str):
    '''
    Function to decompose given time series

    Arguments
    ---------
    df: pd.Dataframe
        The time series
    seasonality_period: int
        The period of one season
    value_name:
        The name of generated value
    
    Returns
    -------
    dict
        Time series decomposition
    '''
    try:
        if df is None or df.empty:
            print("DataFrame is empty")
            return None
        if len(df) < seasonality_period:
            print(f"Seasonality period ({seasonality_period}) is greater then array length ({len(df)})")
            return None
        if seasonality_period <= 0:
            print("Seasonality period must be positive")
            return None

        decomposition = seasonal_decompose(
            df[value_name],
            model='additive',
            period=seasonality_period,
        )        
        return {
            "observed": decomposition.observed, 
            "trend": decomposition.trend, 
            "seasonal": decomposition.seasonal, 
            "resid": decomposition.resid,
        }

    except KeyboardInterrupt:
        print("Interrupting...")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Exception raised during decomposition: {e}")
        return None

def plot_from_dict(arrays: dict, value_name: str, save_name: str | None = None):
    '''
    Function to visualize dashboar and, optionally, to save it.

    Arguments
    ---------
    arrays: dict
        The dictionary, containing arrays for plotting
    save_name: str | None, default = None
        Name of file to which save Dataframe. If save_name == None or "", there will be no saving
    '''
    try:
        def set_labels(ax, title, x, y):
            ax.set_title(title, fontsize=15)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.tick_params(axis='x', labelrotation=45)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), layout="constrained")
        fig.suptitle("Cost dashboard", fontsize=25)
        axes[0, 0].plot(arrays["original"])
        set_labels(axes[0, 0], "Original", "Date", value_name)

        axes[0, 1].plot(arrays["ma"])
        set_labels(axes[0, 1], "Moving average", "Date", value_name)

        axes[1, 0].plot(arrays["decomposed"][0], c="r", label="trend")
        axes[1, 0].plot(arrays["decomposed"][1], c="b", label="seasonal")
        set_labels(axes[1, 0], "Decomposed trend and seasonal", "Date", value_name)
        axes[1, 0].legend()

        axes[1, 1].hist(arrays["resids"], bins=30)
        set_labels(axes[1, 1], "Residuals histogram", f"Residue ({value_name})", "Count")
        if save_name is not None and save_name != "":
            fig.savefig(save_name)
        plt.show()

    except KeyboardInterrupt:
        print("Interrupting...")
        raise KeyboardInterrupt
    except Exception as e:
        print(f"Exception raised during plotting from dict: {e}")
        return None

def main():
    '''Main function.'''
    df = get_dataframe(
        start_date="2024-01-01",
        num_periods=NUM_PERIODS,
        seasonality_period=SEASONALITY_PERIOD,
        value_name=VALUE_NAME,
        shift=10,
    )
    if df is None:
        return
    decomposed = decompose(df, SEASONALITY_PERIOD, VALUE_NAME)
    if decomposed is None:
        return

    try:
        ma = moving_average(df, col=VALUE_NAME, k=5)
        ma = pd.DataFrame(ma, index=df.index)
    except Exception as e:
        print(f"Exception raised during calculating of moving average: {e}")
        return

    try:
        plot_dict = {
            "original": df,
            "ma": ma,
            "decomposed": (decomposed["trend"], decomposed["seasonal"]),
            "resids": decomposed["resid"],
        }
    except Exception as e:
        print(f"Exception raised during data extraction: {e}")
        return

    plot_from_dict(plot_dict, VALUE_NAME, "cost_dashboard.png")


if __name__ == "__main__":
    main()
