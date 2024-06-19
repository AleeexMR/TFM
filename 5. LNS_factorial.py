import random
from gurobipy import Model, GRB, quicksum
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from itertools import permutations
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal


################################################################################
# factor data generating function
################################################################################
def factoSimul(TL=100, PPvector=[100, 100, 100], scenario="scenario1"):
    
    # total number of series
    PP = sum(PPvector)
    
    # factors
    F1 = np.random.normal(1, 1, (TL, 3))
    F2 = np.random.normal(2, 1, (TL, 3))
    F3 = np.random.normal(3, 1, (TL, 3))
    
    # factor loadings
    L1 = np.random.normal(0, 1, (int(PP/3), 3))
    L2 = np.random.normal(0, np.sqrt(2), (int(PP/3), 3))
    L3 = np.random.normal(0, np.sqrt(3), (int(PP/3), 3))
    
    # time series
    xData1 = F1 @ L1.T
    xData2 = F2 @ L2.T
    xData3 = F3 @ L3.T
    xData = np.hstack((xData1, xData2, xData3))
    
    #now we define the three scenarios creating the three different types of errors
    if scenario == "scenario1":
        # error DFM
        Error = np.random.normal(1, 0.1, (TL, PP))
        zData = xData + Error
        
    elif scenario == 'scenario2':
        covariance = np.zeros((PP, PP))
        for i in range(PP):
            for j in range(i, PP):
                covariance[i, j] = covariance[j, i] = 0.3 ** abs(i - j)
        
        error1 = multivariate_normal.rvs(mean=np.zeros(PP), cov=covariance, size=TL)
        error2 = multivariate_normal.rvs(mean=np.zeros(PP), cov=covariance, size=TL)
        delta = np.ones(PP)
        delta[::2] = 0  # Apply delta to even indices
        Error = 0.9 * error1 + delta * error2
        zData = xData + np.sqrt(0.1) * Error
    
    elif scenario == 'scenario3':
        covariance = np.zeros((PP, PP))
        for i in range(PP):
            for j in range(i, PP):
                covariance[i, j] = covariance[j, i] = 0.3 ** abs(i - j)
        
        error1 = multivariate_normal.rvs(mean=np.zeros(PP), cov=covariance, size=TL)
        Error = np.zeros((TL, PP))
        Error[0, :] = error1[0, :]
        for i in range(1, TL):
            Error[i, :] = 0.2 * Error[i-1, :] + error1[i, :]
        
        zData = xData + np.sqrt(0.1) * Error

    return zData


##################################################################################
#  functions to obtain the GCC measure matrix (1.4 in thesis)
##################################################################################
def GCC_sim(x, y, k=1):
    N = len(x)
    M_xy = np.zeros((N - k, 2 * (k + 1)))

    for i in range(k + 1):
        M_xy[:, i] = x[i:N-k+i]
        M_xy[:, i + k + 1] = y[i:N-k+i]

    M_x = M_xy[:, :k+1]
    M_y = M_xy[:, k+1:2*(k+1)]

    R_xy = np.corrcoef(M_xy, rowvar=False)

    if M_x.shape[1] == 1:
        R_x = np.corrcoef(M_x, rowvar=False)
        R_y = np.corrcoef(M_y, rowvar=False)
        GCC = 1 - np.linalg.det(R_xy)**(1 / ((k + 1))) / (R_x**(1 / (k + 1)) * R_y**(1 / (k + 1)))
    else:
        R_x = R_xy[:k+1, :k+1]
        R_y = R_xy[k+1:2*(k+1), k+1:2*(k+1)]
        GCC = 1 - np.linalg.det(R_xy)**(1 / (k + 1)) / (np.linalg.det(R_x)**(1 / (k + 1)) * np.linalg.det(R_y)**(1 / (k + 1)))

    return GCC

def GCCmatrix(serie_0, k=1):
    # Number of series (columns in the input array)
    nSerie = serie_0.shape[1]
    
    # Initialize the dissimilarity matrix with zeros
    DM = np.zeros((nSerie, nSerie))
    
    # Construction of the dissimilarity matrix
    for ii in range(nSerie - 1):
        for jj in range(ii + 1, nSerie):
            g = GCC_sim(serie_0[:, ii], serie_0[:, jj], k)
            DM[ii, jj] = 1 - g
            DM[jj, ii] = 1 - g
    
    return DM

xx1 = factoSimul(scenario='scenario3')
DM= GCCmatrix(xx1, k = 1)

def slcd(D):
    n = D.shape[0]  # assuming D is a square matrix
    for h in range(1, n):  # outer loop, goes through n-1 iterations
        CD = D.copy()  # copy the distance matrix to a new matrix CD
        for j in range(n - 1):  # iterate over rows
            for k in range(j + 1, n):  # iterate over columns from j+1 to n-1
                # Update CD[j, k] and CD[k, j] with the maximum distance between the clusters j and k
                CD[j, k] = np.max(D[[j, k], :], axis=0).min()
                CD[k, j] = CD[j, k]  # symmetric matrix
        D = CD  # update D with the newly computed distances
    return D  
DM1=slcd(DM)


########################################################################
#                     Destroy and repair functions
########################################################################
def destroy(model, u_ik, z_jk, x_ijk, w_ik, percentage=0.15):
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
        
        # Ensure two out of the three related keys are destroyed
        keys_to_destroy = random.sample(related_keys, 2)
        destroyed_u_ik.update(keys_to_destroy)
        
        # Update the all_keys to exclude already destroyed keys to avoid double destruction
        all_keys = [key for key in all_keys if key not in destroyed_u_ik]
    
    # Destroy the selected keys and their related variables
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


# Define the repair function
def repair(model, u_ik, z_jk, x_ijk, w_ik, destroyed_u_ik, destroyed_z_jk, destroyed_x_ijk, destroyed_w_ik):
    # Fix non-destroyed variables
    for key in u_ik.keys():
        if key not in destroyed_u_ik:
            u_ik[key].lb = u_ik[key].X
            u_ik[key].ub = u_ik[key].X
    
    # for key in z_jk.keys():
    #     if key not in destroyed_z_jk:
    #         if z_jk[key].X is not None:
    #             z_jk[key].lb = z_jk[key].X
    #             z_jk[key].ub = z_jk[key].X
    
    # for key in x_ijk.keys():
    #     if key not in destroyed_x_ijk:
    #         if x_ijk[key].X is not None:
    #             x_ijk[key].lb = x_ijk[key].X
    #             x_ijk[key].ub = x_ijk[key].X
    
    for key in w_ik.keys():
        if key not in destroyed_w_ik:
            w_ik[key].lb = w_ik[key].X
            w_ik[key].ub = w_ik[key].X

    model.update()
    model.optimize()
    
    # Unfix the variables after optimization
    for key in u_ik.keys():
        if key not in destroyed_u_ik:
            u_ik[key].lb = 0
            u_ik[key].ub = 1
    
    # for key in z_jk.keys():
    #     if key not in destroyed_z_jk:
    #         z_jk[key].lb = 0
    #         z_jk[key].ub = 1
    
    # for key in x_ijk.keys():
    #     if key not in destroyed_x_ijk:
    #         x_ijk[key].lb = 0
    #         x_ijk[key].ub = 1
    
    for key in w_ik.keys():
        if key not in destroyed_w_ik:
            w_ik[key].lb = 0
            w_ik[key].ub = GRB.INFINITY


    return model

# Define the objective comparison function
def objective_value(model):
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        return model.ObjVal
    else:
        return float('inf')


######################################################################################
#           Function to initialize the model and main LNS function
######################################################################################
# Initialize the model and variables
def initialize_model(n, C, DM1, m=2, u=0.8, eps=0.05):
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
    model.addConstrs((w_ik[i, k] >= DM1[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")
    model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
    model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 1 for k in range(C)), "at_least_u")
    model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")

    objective = quicksum(u_ik_m[i, k] * w_ik[i, k] for i in range(n) for k in range(C))
    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('NonConvex', 2)
    model.setParam('Timelimit', 150)
    
    return model, u_ik, z_jk, x_ijk, w_ik

# Main LNS loop
def large_neighborhood_search(model, u_ik, z_jk, x_ijk, w_ik, max_iterations=200, destroy_percentage=0.15, T0=1.0, alpha=0.95):
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
######################################################################################
n = DM1.shape[0]
C = 3  # Number of clusters

model, u_ik, z_jk, x_ijk, w_ik = initialize_model(n, C, DM1)
best_solution, objective_values, computational_time = large_neighborhood_search(model, u_ik, z_jk, x_ijk, w_ik)

# Extract the final values of the variables
final_u_ik = {key: u_ik[key].X for key in u_ik.keys()}
final_w_ik = {key: w_ik[key].X for key in w_ik.keys()}
final_z_jk = {key: z_jk[key].X for key in z_jk.keys()}
final_x_ijk = {key: x_ijk[key].X for key in x_ijk.keys()}

#To obtain the ARi of the solution:
base_cluster = [0] * 100 + [2] * 100 + [1] * 100

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
plt.title('Objective Value over 200 Iterations (d=0.15)')
plt.grid(True)
plt.show()

