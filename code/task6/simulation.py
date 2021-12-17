# %%
from typing import List
from random import randint
from time import sleep
import numpy as np
from timeit import default_timer as timer

import pandas as pd

from knapsack import KnapSack
# %%
# %%


def generate_random_knapsack(n=1000):
    weights = [randint(50,100) for _ in range(n)]

    profits = [randint(50,100) for _ in range(n)]

    capacity = randint(np.floor((sum(weights)-1)*0.5),np.floor(sum(weights)*0.8))

    knapsack1 = KnapSack(n=n, w=weights, p=profits, c=capacity)
    knapsack2 = KnapSack(n=n, w=weights, p=profits, c=capacity)

    return knapsack1, knapsack2

def interesting_generate_random_knapsack(n=1000):
    """
    Try to generate an instance which does not perform well.
    """

    # weights:

    j = 55
    weights = [randint(80, 90) for _ in range(j)] + [randint(40, 70) for _ in range(j+1, n+1)]

    profits = [randint(600, 800) for _ in range(j)] + [randint(100, 200) for _ in range(j+1, n+1)]

    capacity = randint(100, 200)

    knapsack1 = KnapSack(n=n, w=weights, p=profits, c=capacity)
    knapsack2 = KnapSack(n=n, w=weights, p=profits, c=capacity)

    return knapsack1, knapsack2


def seconds_to_milliseconds(s): return round(s * 1000)

def simulate(generator, N=500, n=10, ):
    result_df = pd.DataFrame(columns=["capacity", "weights", 
                                      "profits", "cut_profit", 
                                      "brute_profit", "cut_time", 
                                      "brute_time"])
    for i in range(N):
        print(f"ITERATION NUMBER: {i}")
        random_ks1, random_ks2 = generator(n=n)


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
result_df = simulate(N=50, n=100, generator=generate_random_knapsack)
result_interesting_df = simulate(N=50, n=100, generator=interesting_generate_random_knapsack)

result_df.to_pickle("./result_basic.pkl")
result_interesting_df.to_pickle("./result_interesting_basic.pkl")
