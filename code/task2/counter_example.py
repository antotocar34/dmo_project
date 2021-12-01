from typing import List

from gurobipy import GRB, Model

n = 3
w = [2, 3, 10]
p = [1, 2, 20]
c = 9


def define_model():
    model = Model()

    x = model.addVars(n, name="x")

    model.setObjective(x.prod(p), sense=GRB.MAXIMIZE)

    model.addConstr(x.prod(w) <= c)
    model.addConstrs(x[i] <= 1 for i in range(n))
    model.addConstrs(-x[i] <= 0 for i in range(n))

    # model.addConstr(-x <= [0 for _ in range(n)] )
    
    return model


model = define_model()

model.optimize()
solution = model.x

def make_cover_inequality(C: List[int]):
    """
    Make function that takes a solution and evaluates to a boolean.
    """
    # Make sure that this subset of variables is a cover
    assert sum([w[i-1] for i in C]) > c

    return lambda x: sum([x[i-1] for i in range(len(C))]) <= len(C) - 1

C = [3] # Define a cover, denotes variable 2 and 3
inequality = make_cover_inequality(C)

print(f"Optimal solution is {solution}")
print(f"Does the solution satisfy the cover inequality corresponding to {C}?\n\t{inequality(solution)}")
