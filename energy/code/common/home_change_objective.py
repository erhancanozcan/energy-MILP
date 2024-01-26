import numpy as np
import pandas as pd
from gurobipy import *


def change_objective_solve(model,obj_coeff):
    
    
    model.setObjective(quicksum(model.getVars()[i]*obj_coeff[i] for i in range(len(obj_coeff))))
    model.update()
    
    model.Params.LogToConsole=0
    model.optimize()
    return model
