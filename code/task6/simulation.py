# %%
from typing import List
import random
from time import sleep
import numpy as np
from timeit import default_timer as timer

import pandas as pd

from knapsack import KnapSack
# %%
# %%


def generate_random_knapsack1(n=1000):
    weights = [random.randint(50,100) for _ in range(n)]

    profits = [random.randint(50,100) for _ in range(n)]

    capacity = random.randint(np.floor((sum(weights)-1)*0.5),np.floor(sum(weights)*0.8))

    knapsack1 = KnapSack(n=n, w=weights, p=profits, c=capacity)
    knapsack2 = KnapSack(n=n, w=weights, p=profits, c=capacity)

    return knapsack1, knapsack2

def generate_random_knapsack2(n=1000):
    weights = [random.randint(50,100) for _ in range(n)]

    profits = [random.randint(50,100) for _ in range(n)]

    capacity = random.randint(np.floor((sum(weights)-1)*0.5),np.floor(sum(weights)*0.8))

    knapsack1 = KnapSack(n=n, w=weights, p=profits, c=capacity)
    knapsack2 = KnapSack(n=n, w=weights, p=profits, c=capacity)

    return knapsack1, knapsack2

def seconds_to_milliseconds(s): return round(s * 1000)

def simulate(N=500, n=10, ):
    result_df = pd.DataFrame(columns=["capacity", "weights", 
                                      "profits", "cut_profit", 
                                      "brute_profit", "cut_time", 
                                      "brute_time"])
    for i in range(N):
        print(f"ITERATION NUMBER: {i}")
        random_ks1, random_ks2 = generate_random_knapsack1(n=n)


        cut_start = timer()
        cut_sol = random_ks1.optimize(cut=True)
        cut_end = timer()
        cut_profit = random_ks1.profit(cut_sol)

        brute_start = timer()
        brute_sol = random_ks2.optimize(cut=False)
        brute_end = timer()
        brute_profit = random_ks2.profit(brute_sol)

        row_to_add = {
            "capacity": random_ks1.c,
            "weights": random_ks1.w,
            "profits": random_ks1.p,
            "cut_profit": cut_profit,
            "brute_profit": brute_profit,
            "cut_time": seconds_to_milliseconds(cut_end - cut_start),
            "brute_time": seconds_to_milliseconds(brute_end - brute_start)
        }
        result_df = result_df.append(row_to_add, ignore_index=True)
        if i % 50 == 0:
            print(f"{N - i} left to go!")

    print("All done :)")
    return result_df
# %%
result_df = simulate(N=50, n=100)

result_df.to_pickle("./result.pkl")
