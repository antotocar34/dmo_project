from typing import List

from gurobipy import Model, GRB, GurobiError, Env

def create_model():
    model = Model()
    model.Params.OutputFlag = 0
    model.Params.Heuristics = 0
    model.Params.Presolve = 0
    model.Params.Cuts = 0
    model.Params.Lazyconstraints = 1
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
        assert [len(p), len(w)] == [n, n]
        self.n = n
        self.c = c
        self.w = w
        self.p = p
        self.model = create_model()
      
    def profit(self, x: List[int]):
        """
        Takes a solution and returns the value 
        of the objective function.
        """
        try:
            assert(len(x) == len(self.p)), f"Solution length: {len(x)}\n Profit vector length: {len(self.p)}"
        except AssertionError:
            breakpoint()
        return sum(self.p[i]*x[i] for i in range(self.n))


    def optimize(self, cut: bool =True) -> List[int]:
        """
        Solves the regular knapsack by enumeration if cut = False,
        and with the cover inequality branch & cut if cut = True
        """
        self.x = self.model.addVars(self.n, vtype=GRB.BINARY)

        # Set objective constraint
        self.model.setObjective(self.x.prod(self.p), sense=GRB.MAXIMIZE)

        # Add capacity constraint
        self.model.addConstr(self.x.prod(self.w) <= self.c)

        if cut:
          self.model.optimize(lambda model, where: self.mycallback(model, where))
        else:
          self.model.optimize()

        return self.model.x

    def greedy_solve(self) -> List[int]:
        """
        As described in task 1
        """
        # Order variable indices by density.
        density_vars = sorted(list(range(self.n)), key=(lambda i: self.p[i] / self.w[i]), reverse=True)

        keep = [] # Variables you are keeping
        running_total = self.c
        for i in density_vars:
            if (running_total - self.w[i] == 0):
                keep.append(i)
                break 
            if (running_total - self.w[i]) < 0:
                continue
            keep.append(i)
            running_total -= self.w[i]
        
        result = [1 if x in keep else 0 for x in range(self.n)]
        return result

    def solve_cover_problem(self, solution: List[int]) -> List[int]:
        """
        Given an existing solution, return a cover, whose inequality the 
        solution violates. If there is no cover return an empty list.
        """
        assert len(solution) == self.n
        # Create the cover knapsack
        coverknapsack = KnapSack(
            n=self.n,
            c=(sum(self.w) - self.c),
            p=[(1 - x) for x in solution],
            w=self.w
        )

        cover = coverknapsack.greedy_solve()
        
        # Do the variable substitution, as explained in task 5.
        cover = [1 - y for y in cover.copy()]

        assert len(cover) == coverknapsack.n
        
        # If there is no cover return without doing anything.
        if coverknapsack.profit(cover) > 1:
          return []

        # list of indices of variables in the cover.
        cover_indices = [i for i in range(coverknapsack.n) if cover[i] == 1]

        return cover_indices
      
    def mycallback(self, model, where):
      """
      Function which adds cover constraint if it exists.
      """
      if where == GRB.Callback.MIPNODE:        
        try:
          # Get fractional solution at a node.
          x_star = self.model.cbGetNodeRel(model.getVars())
          cover_indices = self.solve_cover_problem(x_star)
          if not cover_indices: # If there was no cover just return.
            return

          # Add cover constraint
          self.model.cbLazy( sum(self.x[i] for i in cover_indices) <= len(cover_indices) - 1 )

        except GurobiError:
            pass
