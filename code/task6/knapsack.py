from gurobipy import Model, GRB
from typing import List


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

    def optimize_linear_relaxation(self) -> List[int]:
        """
        Solve the relaxation of the knapsack
        """
        model = Model()
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
        model = Model()
        x = model.addVars(self.n, vtype=GRB.BINARY)

        # Set objective constraint
        model.setObjective(x.prod(self.p), sense=GRB.MAXIMIZE)

        # Add capacity constraint
        model.addConstr(x.prod(self.w) <= self.c)

        model.optimize()

        return model.x

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
        # TODO Is this model always feasible? We should handle the case where it it is not.
        # Yes, this model should always be feasible (you can always set y^*_i to 1).
        # TODO make a method for a greedy algorithm
        cover = coverknapsack.optimize()

        return cover
