from gurobipy import Model, GRB, quicksum
import numpy as np
from fcmeans import FCM
from sklearn.metrics import adjusted_rand_score

#This script contains the final fuzzy clustering algorithm created with gurobi,
# with the restrictions mentioned above to be able to use any of the three linkages.

#D is the distance matrix, n is the number of individuals/time series, C is the number of clusters,
#u is the threshold for grades of membership, m is a given parameter (2), and eps is a small
# positive number

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
               [1., 1., 1., 1., 1., 1., 1, 1,1,1, 0.,0.75,1.,1.,1],
               [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.75, 0.,0.75,1.,1.],
               [1., 1., 1., 1., 1., 1., 1, 1,1,1,1,0.75 ,0.,0.75,1.],
               [1., 1., 1., 1., 1., 1., 1, 1,1,1,1,1,0.75, 0.,0.75],
               [1., 1., 1., 1., 1., 1., 1, 1,1,1,1.,1.,1.,0.75,0]])


D7 = np.array([[0., 0.191, 0.192, 0.193, 1., 1., 1., 1., 1., 1., 1.,1., 1., 1.,1.],
                [0.191, 0.,0.194,0.195,1., 1., 1., 1., 1. , 1., 1.,1., 1.,1., 1.],
                [0.192, 0.194, 0., 0.196,1.,1.,1.,1., 1., 1., 1.,1., 1., 1., 1.],
                [0.193, 0.195, 0.196, 0.,1., 1., 1., 1., 1., 1.,1.,1., 1., 1., 1.],
                [1., 1., 1., 1.,0., 0.19, 0.19, 0.19, 0.19, 0.19,1.,1.,1.,1.,1.],
                [1., 1., 1., 1.,0.19, 0., 0.19, 0.19, 0.19, 0.19,1.,1.,1.,1.,1],
                [1., 1., 1., 1.,0.19, 0.19, 0., 0.19, 0.19, 0.19,1.,1.,1.,1.,1.],
                [1., 1., 1., 1.,0.19, 0.19, 0.19, 0., 0.19, 0.19,1.,1.,1.,1.,1.],
                [1., 1., 1., 1.,0.19, 0.19, 0.19, 0.19, 0., 0.19,1.,1.,1.,1.,1],
                [1., 1., 1., 1.,0.19, 0.19, 0.19, 0.19, 0.19, 0.,1.,1.,1.,1.,1],
                [1., 1., 1., 1., 1., 1., 1, 1,1,1, 0., 0.19, 0.19, 0.19, 0.19],
                [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.19, 0., 0.19, 0.19, 0.19],
                [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.19, 0.19, 0., 0.19, 0.19],
                [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.19, 0.19, 0.19, 0., 0.19],
                [1., 1., 1., 1., 1., 1., 1, 1,1,1,0.19, 0.19, 0.19, 0.19, 0.]])

D4 = np.array([[0., 0.75, 0.75, 0.75, 1., 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
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

###############################################################################
#                           Fuzzy clustering algorithm
###############################################################################

n = D7.shape[0]
C = 3 
u = 0.6
m = 2
eps = 0.001
M = np.max(D7)

model = Model("FuzzyClusteringAlex")

# Decision variables, their type and their bounds
u_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik", lb=0, ub=1)
z_jk = model.addVars(n, C,vtype=GRB.BINARY, name="z_jk")
x_ijk = model.addVars(n, n, C, vtype=GRB.BINARY, name="x_ijk")
w_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="w_ik")

#Auxiliary variables for u_ik^m 
u_ik_m = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik_m")

#These constraints define u_ik_m as (u_ik)^m because in the objective function i 
#cannot add quadratic terms directly.
#This works directly only for m=2 and it is time consuming but is the only way 
#the code works wit m=2. For other values of m, other solver has to be used
if m == 2:
    model.addConstrs((u_ik_m[i, k] == u_ik[i, k] * u_ik[i, k] for i in range(n) for k in range(C)), "u_ik_m_def")

# Constraints
# Sum of u_ik over k equals 1
model.addConstrs((quicksum(u_ik[i, k] for k in range(C)) == 1 for i in range(n)), "sum_u_ik")

# u_jk constraints
model.addConstrs((u_ik[j, k] <= (u - eps) * (1 - z_jk[j, k]) + z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_lower")
model.addConstrs((u_ik[j, k] >= (u + eps) * z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_upper")

# x_ijk and delta_ijk constraints
model.addConstrs((quicksum(x_ijk[i, j, k] for j in range(n) if j != i) == 1 for i in range(n) for k in range(C)), "one_closest")

# w_ik constraints single linkage
#model.addConstrs((w_ik[i, k] >= D7[i, j] * x_ijk[i, j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_upper")
#model.addConstrs((w_ik[i, k] <= D7[i, j] + M * (1 - x_ijk[i, j, k])
#                     for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_lower")

# def of w_ik to use complete distance w_ik 
model.addConstrs((w_ik[i, k] >= D6[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")

# def of w_ik to use average lingake
# for i in range(n):
#           for k in range(C):
#               sum_distances = quicksum(D7[i, j] * z_jk[j, k] for j in range(n) if j != i)
#               sum_memberships = quicksum(z_jk[j, k] for j in range(n) if j != i)
#               model.addConstr(w_ik[i, k] * sum_memberships == sum_distances, f"w_ik_average_{i}_{k}")


model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 1 for k in range(C)), "at_least_u")

# for all individuals to belong to some cluster:
model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")

#Finally, the objective function
objective = quicksum(u_ik_m[i, k]*w_ik[i, k] for i in range(n) for k in range(C))
model.setObjective(objective, GRB.MINIMIZE)


#The following commented lines of code are tests with initial solutions(not necessaries)
# taken from the fuzzy c-means algorithm.

# fcm = FCM(n_clusters=C)
# fcm.fit(D4)  # Assuming D4 is your data matrix. Adjust accordingly if D4 is a distance matrix
# centers = fcm.centers
# u_ik_initial = fcm.u  # Membership degree matrix from FCM
# u_ik_initial = {(i, j): u_ik_initial[i, j] for i in range(u_ik_initial.shape[0]) for j in range(u_ik_initial.shape[1])}
# print(u_ik_initial)

# w_ik_initial = {
#     (0, 0): 0.18999999999999995,
#     (0, 1): 1,
#     (0, 2): 1,
#     (1, 0): 0.18999999999999995,
#     (1, 1): 1,
#     (1, 2): 1,
#     (2, 0): 0.18999999999999995,
#     (2, 1): 1,
#     (2, 2): 1,
#     (3, 0): 0.18999999999999995,
#     (3, 1): 1,
#     (3, 2): 1,
#     (4, 0): 1,
#     (4, 1): 1,
#     (4, 2): 0.18999999999999995,
#     (5, 0): 1,
#     (5, 1): 1,
#     (5, 2): 0.18999999999999995,
#     (6, 0): 1,
#     (6, 1): 1,
#     (6, 2): 0.18999999999999995,
#     (7, 0): 1,
#     (7, 1): 1,
#     (7, 2): 0.18999999999999995,
#     (8, 0): 1,
#     (8, 1): 1,
#     (8, 2): 0.18999999999999995,
#     (9, 0): 1,
#     (9, 1): 1,
#     (9, 2): 1.18999999999999995,
#     (10, 0): 1,
#     (10, 1): 0.18999999999999995,
#     (10, 2): 1,
#     (11, 0): 1,
#     (11, 1): 0.18999999999999995,
#     (11, 2): 1,
#     (12, 0): 1,
#     (12, 1): 0.18999999999999995,
#     (12, 2): 1,
#     (13, 0): 1,
#     (13, 1): 0.18999999999999995,
#     (13, 2): 1,
#     (14, 0): 1,
#     (14, 1): 0.18999999999999995,
#     (14, 2): 1
# }

# for (i, k), value in u_ik_initial.items():
#     u_ik[i, k].start = value
    
# for (i, k), value in w_ik_initial.items():
#      w_ik[i, k].start = value

# Set the NonConvex parameter to allow solving non-convex problems, if i dont put this, an error occurs
model.setParam('NonConvex', 2)
model.setParam('Timelimit', 100)
model.optimize()

#print(model) to see some insights about the model
#for constr in model.getConstrs():
#    print(constr)

#To obtain the values of delta_ijk:
#delta_ijk_sol = model.getAttr('X', delta_ijk)
#delta_ijk_sol_filtered = {key: value for key, value in delta_ijk_sol.items() if key[0] <= 5}
#x_ijk_sol=model.getAttr('X',x_ijk)
#for key in delta_ijk_sol:
    #print(f"{key}: {delta_ijk_sol[key]}")
#for key in x_ijk_sol:
#    print(f"{key}: {x_ijk_sol[key]}")

# Possible results
if model.Status == GRB.INF_OR_UNBD or model.Status == GRB.INFEASIBLE:
    print("Model is infeasible")
else:
    u_ik_sol = model.getAttr('X', u_ik)
    print("Solution:", u_ik_sol)
    
    
#############################################################################
#                     from u_ik to label to obtain the ARI
#############################################################################
#Code to obtain the ARI of the solution:
final_u_ik = {key: u_ik[key].X for key in u_ik.keys()}
pred_labels = []
for i in range(n):
    max_index = 0  
    max_value = -float('inf')  
    for k in range(C): 
        if final_u_ik[(i, k)] > max_value:
            max_value = final_u_ik[(i, k)]
            max_index = k  
    pred_labels.append(max_index)
    
#adjust the following line of code to correctly compute the ARI (looking at the variable pred_labels)
real_clusters = [0,0,0,0,1,1,1,1,1,1,2,2,2,2,2]
print(adjusted_rand_score(pred_labels, real_clusters))