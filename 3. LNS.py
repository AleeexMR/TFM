import random
from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.metrics import adjusted_rand_score
from itertools import permutations
    
########################################################################
#                     Destroy and repair functions
########################################################################
def destroy(model, u_ik, z_jk, x_ijk, w_ik, percentage=0.3):
    # number of elements to destroy
    num_to_destroy = int(len(u_ik) * percentage)
    
    all_keys = list(u_ik.keys())
    destroyed_u_ik = set()
    destroyed_z_jk = set()
    destroyed_x_ijk = set()
    destroyed_w_ik = set()
    
    while len(destroyed_u_ik) < num_to_destroy:
        # Randomly select a key to destroy
        i, k = random.choice(all_keys)
        
        # Find the set of keys (x, 0), (x, 1), (x, 2) for the chosen x
        related_keys = [(i, 0), (i, 1), (i, 2)]
        
        #ensure two out of the three related keys are destroyed
        keys_to_destroy = random.sample(related_keys, 2)
        destroyed_u_ik.update(keys_to_destroy)
        
        # Update the all_keys to exclude already destroyed keys to avoid double destruction
        all_keys = [key for key in all_keys if key not in destroyed_u_ik]
    
    # finally destroy the selected keys and their related variables
    for (i, k) in destroyed_u_ik:
        u_ik[i, k].start = 0  
        for j in range(n):
            if (j, k) in z_jk:
                destroyed_z_jk.add((j, k))
                z_jk[j, k].start = 0
            if (i, j, k) in x_ijk:
                destroyed_x_ijk.add((i, j, k))
                x_ijk[i, j, k].start = 0
        destroyed_w_ik.add((i, k))
        w_ik[i, k].start = 0
        
    return list(destroyed_u_ik), list(destroyed_z_jk), list(destroyed_x_ijk), list(destroyed_w_ik)


def repair(model, u_ik, z_jk, x_ijk, w_ik, destroyed_u_ik, destroyed_z_jk, destroyed_x_ijk, destroyed_w_ik):
    # First we fix non-destroyed variables
    for key in u_ik.keys():
        if key not in destroyed_u_ik:
            u_ik[key].lb = u_ik[key].X
            u_ik[key].ub = u_ik[key].X
    
    for key in z_jk.keys():
        if key not in destroyed_z_jk:
            z_jk[key].lb = z_jk[key].X
            z_jk[key].ub = z_jk[key].X
    
    for key in x_ijk.keys():
        if key not in destroyed_x_ijk:
            x_ijk[key].lb = x_ijk[key].X
            x_ijk[key].ub = x_ijk[key].X
    
    for key in w_ik.keys():
        if key not in destroyed_w_ik:
            w_ik[key].lb = w_ik[key].X
            w_ik[key].ub = w_ik[key].X

    #We optimize the model with these variables fixed
    model.update()
    model.optimize()
    
    # Unfix the variables after optimization
    for key in u_ik.keys():
        if key not in destroyed_u_ik:
            u_ik[key].lb = 0
            u_ik[key].ub = 1
    
    for key in z_jk.keys():
        if key not in destroyed_z_jk:
            z_jk[key].lb = 0
            z_jk[key].ub = 1
    
    for key in x_ijk.keys():
        if key not in destroyed_x_ijk:
            x_ijk[key].lb = 0
            x_ijk[key].ub = 1
    
    for key in w_ik.keys():
        if key not in destroyed_w_ik:
            w_ik[key].lb = 0
            w_ik[key].ub = GRB.INFINITY
    
    return model

# objective comparison function (objective function of the model)
def objective_value(model):
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        return model.ObjVal
    else:
        return float('inf')

######################################################################################
#           Function to initialize the model and main LNS function
######################################################################################
# I use the created algorithm to obtain a first ("bad") feasible solution (TimeLimit=reduced)
def initialize_model(n, C, D7, m=2, u=0.7, eps=0.05):
    model = Model("FuzzyClusteringAlex")

    u_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik", lb=0, ub=1)
    z_jk = model.addVars(n, C, vtype=GRB.BINARY, name="z_jk")
    x_ijk = model.addVars(n, n, C, vtype=GRB.BINARY, name="x_ijk")
    w_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="w_ik")
    u_ik_m = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik_m")

    if m == 2:
        model.addConstrs((u_ik_m[i, k] == u_ik[i, k] * u_ik[i, k] for i in range(n) for k in range(C)), "u_ik_m_def")

    model.addConstrs((quicksum(u_ik[i, k] for k in range(C)) == 1 for i in range(n)), "sum_u_ik")

    model.addConstrs((u_ik[j, k] <= (u - eps) * (1 - z_jk[j, k]) + z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_lower")
    model.addConstrs((u_ik[j, k] >= (u + eps) * z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_upper")

    model.addConstrs((quicksum(x_ijk[i, j, k] for j in range(n) if j != i) == 1 for i in range(n) for k in range(C)), "one_closest")
    model.addConstrs((w_ik[i, k] >= D6[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")
    model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
    model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 1 for k in range(C)), "at_least_u")
    model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")

    objective = quicksum(u_ik_m[i, k] * w_ik[i, k] for i in range(n) for k in range(C))
    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('NonConvex', 2)
    model.setParam('Timelimit', 4) #This value is very important, if we put a big number, the algorithm will start with a very good solution
    # so the LNS will not improve it very much, if we put a reasonable small value, the algorithm will improve the objective function a lot
    #The best option is to fix it doing some trials, and it has to be adapted for each scenario, for example, for the most chain-dependent 
    #scenarios, you have to put a higher number because it is difficult to find a good initial solution, but for D6 or D7, this number has 
    #to be small because good solutions are found very quickly. 
    
    return model, u_ik, z_jk, x_ijk, w_ik

# Main LNS loop
def large_neighborhood_search(model, u_ik, z_jk, x_ijk, w_ik, max_iterations=500, destroy_percentage=0.3, T0=1.0, alpha=0.95):
    start_time = time.time()
    model.optimize()
    xb = model.copy()
    best_obj_val = objective_value(model)
    print(f"Initial objective value: {best_obj_val}")
    
    T = T0  # Initialize temperature
    objective_values = [best_obj_val]  # I will store the objective values to make a plot
    
    for iteration in range(max_iterations):
        destroyed_u_ik, destroyed_z_jk, destroyed_x_ijk, destroyed_w_ik = destroy(model, u_ik, z_jk, x_ijk, w_ik, percentage=destroy_percentage)
        xt = repair(model, u_ik, z_jk, x_ijk, w_ik, destroyed_u_ik, destroyed_z_jk, destroyed_x_ijk, destroyed_w_ik)
        
        current_obj_val = objective_value(xt)
        if current_obj_val == float('inf'):
            print(f"Iteration {iteration + 1}, no feasible solution found.")
        else:
            print(f"Iteration {iteration + 1}, objective value: {current_obj_val}")
        
        objective_values.append(current_obj_val)
        
        if current_obj_val < best_obj_val or random.random() < math.exp(-(current_obj_val - best_obj_val) / T):
            model = xt
            best_obj_val = current_obj_val
            if current_obj_val < best_obj_val:
                xb = xt.copy()

        # Update temperature
        T *= alpha

    end_time = time.time()  # End time
    computational_time = end_time - start_time
    return xb, objective_values, computational_time

######################################################################################
#                               Sample usage
########################################################################################
D6 = np.array([[0., 0.19, 0.19, 0.19, 1., 1., 1., 1., 1., 1., 1.,1., 1., 1.,1.],
               [0.19, 0.,0.19,0.19,1., 1., 1., 1., 1. , 1., 1.,1., 1.,1., 1.],
               [0.19, 0.19, 0., 0.19,1.,1.,1.,1., 1., 1., 1.,1., 1., 1., 1.],
               [0.19, 0.19, 0.19, 0.,1., 1., 1., 1., 1., 1.,1.,1., 1., 1., 1.],
               [1., 1., 1., 1.,0., 0.19, 0.19, 0.19, 0.19, 0.19,1.,1.,1.,1.,1.],
               [1., 1., 1., 1.,0.19, 0., 0.19, 0.19, 0.19, 0.19,1.,1.,1.,1.,1],
               [1., 1., 1., 1.,0.19, 0.19, 0., 0.19, 0.19, 0.19,1.,1.,1.,1.,1.],
               [1., 1., 1., 1.,0.19, 0.19, 0.19, 0., 0.19, 0.19,1.,1.,1.,1.,1.],
               [1., 1., 1., 1.,0.19, 0.19, 0.19, 0.19, 0., 0.19,1.,1.,1.,1.,1],
               [1., 1., 1., 1.,0.19, 0.19, 0.19, 0.19, 0.19, 0.,1.,1.,1.,1.,1],
              [1., 1., 1., 1., 1., 1., 1, 1,1,1, 0.,0.75,0.75,0.75,0.75],
              [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.75, 0.,0.75,0.75,0.75],
              [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.75,0.75 ,0.,0.75,0.75],
              [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.75,0.75,0.75, 0.,0.75],
              [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.75,0.75,0.75,0.75,0]])

n = D6.shape[0]
C = 3  # Number of clusters

model, u_ik, z_jk, x_ijk, w_ik = initialize_model(n, C, D6)
best_solution, objective_values, computational_time = large_neighborhood_search(model, u_ik, z_jk, x_ijk, w_ik)

#Final values of the variables
final_u_ik = {key: u_ik[key].X for key in u_ik.keys()}
final_w_ik = {key: w_ik[key].X for key in w_ik.keys()}
final_z_jk = {key: z_jk[key].X for key in z_jk.keys()}
final_x_ijk = {key: x_ijk[key].X for key in x_ijk.keys()}

#To obtain the ARi of the solution:
base_cluster = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

#To obtain the classification from the obtained grades of memberships
pred_labels = []
for i in range(n):
    max_index = 0  
    max_value = -float('inf')  
    for k in range(C): 
        if final_u_ik[(i, k)] > max_value:
            max_value = final_u_ik[(i, k)]
            max_index = k  
    pred_labels.append(max_index)  

ari = adjusted_rand_score(pred_labels, base_cluster) 

print(f"ARI: {ari}")

# Plot the objective values over iterations
plt.plot([i for i in range(len(objective_values))], objective_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Objective Value over 500 Iterations')
plt.grid(True)
plt.show()
