import numpy as np
from gurobipy import Model, GRB, quicksum
from sklearn.metrics import adjusted_rand_score

# Define the distance matrix, in this case, we have the matrix of the scenario 7
D7 = np.array([[0., 0.75, 0.75, 0.75, 1., 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
              [0.75, 0., 0.75, 0.75, 1., 1., 1.,1.,1.,1.,1.,1.,1.,1.,1.],
              [0.75, 0.75, 0., 0.75, 1., 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
              [0.75, 0.75, 0.75, 0., 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
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

n = D7.shape[0] #number of time series
C = 3 #Number of clusters
u = 0.6 #threshold to consider that one observation belongs to a cluster
m = 2 # square in the objective function
eps = 0.001
M = np.max(D7) #The maximum value of the matrix of dissimilarities

#We introduce the optimization model in the next function in order to obtain 
#the predicted classification
def run_optimization():
    model = Model("FuzzyClusteringAlex")

    # Decision variables
    u_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik", lb=0, ub=1)
    z_jk = model.addVars(n, C, vtype=GRB.BINARY, name="z_jk")
    x_ijk = model.addVars(n, n, C, vtype=GRB.BINARY, name="x_ijk")
    w_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="w_ik")
    
    # Auxiliary variables for u_ik^m
    u_ik_m = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik_m")
    
    if m == 2:
        model.addConstrs((u_ik_m[i, k] == u_ik[i, k] * u_ik[i, k] for i in range(n) for k in range(C)), "u_ik_m_def")
    
    # Constraints
    model.addConstrs((quicksum(u_ik[i, k] for k in range(C)) == 1 for i in range(n)), "sum_u_ik")
    model.addConstrs((u_ik[j, k] <= (u - eps) * (1 - z_jk[j, k]) + z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_lower")
    model.addConstrs((u_ik[j, k] >= (u + eps) * z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_upper")
    model.addConstrs((quicksum(x_ijk[i, j, k] for j in range(n) if j != i) == 1 for i in range(n) for k in range(C)), "one_closest")
    
    # def of w_ik to use Single linkage
    #model.addConstrs((w_ik[i, k] >= D6[i, j] * x_ijk[i, j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_upper")
    #model.addConstrs((w_ik[i, k] <= D6[i, j] + M * (1 - x_ijk[i, j, k])
    #                     for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_lower")

    # def of w_ik to use Average linkage
    # for i in range(n):
    #     for k in range(C):
    #         sum_distances = quicksum(D2[i, j] * z_jk[j, k] for j in range(n) if j != i)
    #         sum_memberships = quicksum(z_jk[j, k] for j in range(n) if j != i)
    #         model.addConstr(w_ik[i, k] * sum_memberships == sum_distances, f"w_ik_average_{i}_{k}")
    
    # def of w_ik to use complete distance 
    model.addConstrs((w_ik[i, k] >= D7[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")

    model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
    model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 1 for k in range(C)), "at_least_u")
    model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")
    
    objective = quicksum(u_ik_m[i, k] * w_ik[i, k] for i in range(n) for k in range(C))
    model.setObjective(objective, GRB.MINIMIZE)
    
    model.setParam('NonConvex', 2) #To consider nonconvex problems
    model.setParam('Timelimit', 100) #Time limit parameter
    model.optimize()
    
    #Using the values of the membership degrees obtained, we compute the classification to each 
    #element to the clusters
    a = {indices: var.X for indices, var in u_ik.items()}
    pred_labels = []
    for i in range(n):
        max_index = 0  
        max_value = -float('inf')  
        for k in range(C): 
            if a[(i, k)] > max_value:
                max_value = a[(i, k)]
                max_index = k  
        pred_labels.append(max_index)  
    
    return pred_labels

#############################################################################
#                        Monte Carlo simulation
############################################################################
T = 100
aris = []

real_cluster = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

for _ in range(T):
    pred_labels = run_optimization()
    ari = adjusted_rand_score(pred_labels, real_cluster)
    aris.append(ari)

average_ari = np.mean(aris)
print(f"Average ARI over {T} runs: {average_ari}")
