from gurobipy import Model, GRB
from typing import List
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random


def create_model():
    model = Model()
    model.Params.Cuts = 0
    model.Params.Heuristics = 0
    model.Params.Presolve = 0
    return model

class KnapSack:
    """
    An implementation of a Knapsack problem.

    n: Number of variables
    c: Capacity
    p: n sized list of profits
    w: n sized list of weights
    """
    def __init__(self, n: int, c: int, p: List[int], w: List[int]):
        # Checks that all vectors are of correct length.
        assert [len(p), len(p)] == [n, n]
        self.n = n
        self.c = c
        self.w = w
        self.p = p

    def profit(self, x):
        assert(len(x) == len(self.p)), f"{x}\n{p}"
        return sum(self.p[i]*x[i] for i in range(self.n))

    def optimize_linear_relaxation(self) -> List[int]:
        """
        Solve the relaxation of the knapsack
        """
        model = create_model()
        x = model.addVars(self.n)

        # Set objective constraint
        model.setObjective(x.prod(self.p), sense=GRB.MAXIMIZE)

        # Add capacity constraint
        # TODO Add the small numerical trick
        model.addConstr(x.prod(self.w) <= self.c)

        # Add the 0 <= x_i <= 1 constraint
        model.addConstrs(x[i] <= 1 for i in range(self.n))
        model.addConstrs(-x[i] <= 0 for i in range(self.n))

        # Optimize the model
        model.optimize()
        solution = model.x
        return solution

    def optimize(self) -> List[int]:
        """
        Solves the regular knapsack.
        """
        model = create_model()
        x = model.addVars(self.n, vtype=GRB.BINARY)

        # Set objective constraint
        model.setObjective(x.prod(self.p), sense=GRB.MAXIMIZE)

        # Add capacity constraint
        model.addConstr(x.prod(self.w) <= self.c)

        model.optimize()

        return model.x

    def greedy_solve(self):
        """
        Given a capacity and a list of density ordered vertices,
        return a list of variables until the cumulative sum
        of the next one violates the capacity.
        """
        # Order variable indices by density
        density_vars = sorted(list(range(self.n)), key=(lambda i: self.p[i] / self.w[i]), reverse=True)

        keep = [] # Variables you are keeping
        running_total = self.c
        for i in density_vars:
            if (running_total - self.w[i] == 0):
                keep.append(i)
                return keep
            if (running_total - self.w[i]) < 0:
                continue
            keep.append(i)
            running_total -= self.w[i]
        
        return [1 if x in keep else 0 for x in range(self.n)]

    def solve_cover_problem(self, solution: List[int]) -> List[int]:
        """
        Given an existing solution, return a cover, whose inequality the 
        solution violates.
        """
        # TODO Add the small numerical trick
        coverknapsack = KnapSack(
            n=self.n,
            c=-1*self.c,
            p=[-(1 - x) for x in solution],
            w=[-1*v for v in self.w]
        )

        cover = coverknapsack.greedy_solve()

        return cover