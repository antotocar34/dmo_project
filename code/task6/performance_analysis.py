# %%
import random
random.seed(10)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from knapsack import KnapSack
# %%
def generate_random_knapsack(n=1000):

    # weights:
    weights = [random.randint(50,100) for i in range(n)]

    profits = [random.randint(50,100) for i in range(n)]

    capacity = random.randint(np.floor((sum(weights)-1)*0.5),np.floor(sum(weights)*0.8))

    knapsack = KnapSack(n=n, w=weights, p=profits, c=capacity)

    return knapsack
# %%
def seconds_to_milliseconds(s): return round(s * 1000)

def simulate(N=300):
    result_df = pd.DataFrame(columns=["capacity", "weights", 
                                      "profits", "greedy_profit", 
                                      "optimal_profit", "greedy_time", 
                                      "optimal_time"])
    for i in range(N):
        random_ks = generate_random_knapsack()


        greedy_start = timer()
        greedy_sol = random_ks.greedy_solve()
        greedy_end = end = timer()



        greedy_profit = random_ks.profit(greedy_sol)

        optimal_start = timer()
        optimal_sol = random_ks.optimize()
        optimal_end = timer()
        optimal_profit = random_ks.profit(optimal_sol)

        row_to_add = {
            "capacity": random_ks.c,
            "weights": random_ks.w,
            "profits": random_ks.p,
            "greedy_profit": greedy_profit,
            "optimal_profit": optimal_profit,
            "greedy_time": seconds_to_milliseconds(greedy_end - greedy_start),
            "optimal_time": seconds_to_milliseconds(optimal_end - optimal_start)
        }
        result_df = result_df.append(row_to_add, ignore_index=True)
        if i % 50 == 0:
            print(f"{N - i} left to go!")

    print("All done :)")
    return result_df
# %%
result_df = simulate()
# %%
result_df["profit_difference"] = result_df["optimal_profit"] - result_df["greedy_profit"]
result_df["profit_difference"].hist()
# %%
result_df.plot(y=["greedy_time", "optimal_time"], 
               kind="line", 
               ylabel="milliseconds");
# %%
result_df.plot(y=["greedy_profit", "optimal_profit"], kind="line");
# %%
