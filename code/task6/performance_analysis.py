from random import randint
from typing import Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from knapsack import KnapSack
# %%
result_df = pd.read_pickle("./result_basic.pkl")
result_df_interesting = pd.read_pickle("./result_interesting_basic.pkl")

def generate_graphs(result_df: pd.DataFrame, namer: str):
    time_basic = result_df.plot(y=["cut_time", "brute_time"], 
                   kind="line", 
                   ylabel="milliseconds",
                   xlabel="iterations");
    # %%
    profit_basic = result_df.plot(y=["cut_profit", "brute_profit"], kind="line", ylabel="profit", xlabel="iterations");
    profit_difference = (result_df["brute_profit"] - result_df["cut_profit"]).plot(kind="line", ylabel="profit", xlabel="iterations")

    def save_pandas_fig(plot_obj, name):
        fig = plot_obj.get_figure()
        fig.savefig(namer + name + ".png")

    save_pandas_fig(time_basic, namer + "time_basic")
    save_pandas_fig(profit_basic, namer + "profit_basic")
    save_pandas_fig(
        profit_difference,
        namer + "profit_difference"
        )

generate_graphs(result_df, "basic")
generate_graphs(result_df_interesting, "interesting")
