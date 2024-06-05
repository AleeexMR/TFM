import numpy as np

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

from scipy.stats import multivariate_normal

def factoSimul(TL=100, PPvector=[100, 100, 100], scenario="scenario1"):
    R = 3  # Number of groups
    rs = [3, 3, 3]  # Number of factors in each group
    
    # Total number of series
    PP = sum(PPvector)
    
    # Generating the group indices
    P = np.concatenate([np.full(PPvector[i], i+1) for i in range(R)])
    
    # Factors
    F1 = np.random.normal(1, 1, (TL, 3))
    F2 = np.random.normal(2, 1, (TL, 3))
    F3 = np.random.normal(3, 1, (TL, 3))
    
    # Factor loadings
    L1 = np.random.normal(0, 1, (int(PP/3), 3))
    L2 = np.random.normal(0, np.sqrt(2), (int(PP/3), 3))
    L3 = np.random.normal(0, np.sqrt(3), (int(PP/3), 3))
    
    # Time series
    xData1 = F1 @ L1.T
    xData2 = F2 @ L2.T
    xData3 = F3 @ L3.T
    xData = np.hstack((xData1, xData2, xData3))
    
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

xx1 = factoSimul(scenario='scenario1')
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

from gurobipy import Model, GRB, quicksum
from fcmeans import FCM
n = DM1.shape[0]
C = 3
u = 0.65
m = 2
eps = 0.0001
M = np.max(DM1)

model = Model("FuzzyClusteringAlex")

# Decision variables
u_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik", lb=0, ub=1)
z_jk = model.addVars(n, C,vtype=GRB.BINARY, name="z_jk")
x_ijk = model.addVars(n, n, C, vtype=GRB.BINARY, name="x_ijk")
w_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="w_ik")

#Auxiliary variables for u_ik^m 
u_ik_m = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik_m")

#These constraints define u_ik_m as (u_ik)^m because in the objective function i cannot add quadratic terms directly.
#This works directly only for m=2 and it is time consuming but is the only way the code works wit m=2.
#For other values of m, i would have to search another solver
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

# w_ik constraints VG
# model.addConstrs((w_ik[i, k] >= D2[i, j] * x_ijk[i, j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_upper")
# model.addConstrs((w_ik[i, k] <= D2[i, j] + M * (1 - x_ijk[i, j, k])
#                         for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_lower")

# def of w_ik to use complete distance w_ik 
model.addConstrs((w_ik[i, k] >= DM1[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")

# def of w_ik to use average lingake
# for i in range(n):
#           for k in range(C):
#               sum_distances = quicksum(DM1[i, j] * z_jk[j, k] for j in range(n) if j != i)
#               sum_memberships = quicksum(z_jk[j, k] for j in range(n) if j != i)
#               model.addConstr(w_ik[i, k] * sum_memberships == sum_distances, f"w_ik_average_{i}_{k}")


model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 1 for k in range(C)), "at_least_u")

#para que todos los individuos pertenezcan a algun cluster:
model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")

objective = quicksum(u_ik_m[i, k]*w_ik[i, k] for i in range(n) for k in range(C))
model.setObjective(objective, GRB.MINIMIZE)

#Run Fuzzy C-Means to get initial cluster centers and membership matrix
# fcm = FCM(n_clusters=C)
# fcm.fit(DM)  
# centers = fcm.centers
# u_ik_initial = fcm.u  # Membership degree matrix from FCM
# u_ik_initial = {(i, j): u_ik_initial[i, j] for i in range(u_ik_initial.shape[0]) for j in range(u_ik_initial.shape[1])}

# for (i, k), value in u_ik_initial.items():
#     u_ik[i, k].start = value

# Set the NonConvex parameter to allow solving non-convex problems, if i dont put this, an error occurs
model.setParam('NonConvex', 2)
# model.setParam(GRB.Param.MIPGap, 0.016)
model.setParam('Timelimit',10800)
model.optimize()



#To obtain the values of delta_ijk:
#delta_ijk_sol = model.getAttr('X', delta_ijk)
#delta_ijk_sol_filtered = {key: value for key, value in delta_ijk_sol.items() if key[0] <= 5}
#x_ijk_sol=model.getAttr('X',x_ijk)
#for key in delta_ijk_sol:
    #print(f"{key}: {delta_ijk_sol[key]}")
#for key in x_ijk_sol:
#    print(f"{key}: {x_ijk_sol[key]}")

# Possible esults
if model.Status == GRB.INF_OR_UNBD or model.Status == GRB.INFEASIBLE:
    print("Model is infeasible")
else:
    u_ik_sol = model.getAttr('X', u_ik)
    print("Solution:", u_ik_sol)

#con CD
#average u=0.6 s1, todo 2 menos los ultimos

#concd, x=0.8, scenario2 : 0.79146