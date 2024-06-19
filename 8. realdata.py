import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sns


##################################################################################
#                       Load the data
##################################################################################
file_path = "C:/Users/Alex/OneDrive/Documentos/MASTER/TFM/ItalianElectricityPrices.csv"

# Read the data from the first sheet
df = pd.read_csv(file_path, header=None)

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
            g = GCC_sim(serie_0.iloc[:, ii], serie_0.iloc[:, jj], k)  # Use .iloc for integer-based indexing
            DM[ii, jj] = 1 - g
            DM[jj, ii] = 1 - g
    
    return DM

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

DM = GCCmatrix(df, k=1)
DM1=slcd(DM)

plt.figure(figsize=(12, 10))

##################################################################################
#                       Heatmap
##################################################################################

# sns.heatmap(DM, cmap="viridis", vmin=0, vmax=1)
plt.title("Heatmap")
plt.show()


##################################################################################
#                               Dendrogram
##################################################################################
linked = sch.linkage(DM1, method='complete')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(linked,
               orientation='left',
               distance_sort='descending',
               show_leaf_counts=True)
plt.title('Dendrogram using CD measure and Complete Linkage')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()

##################################################################################
#       loop to obtain the 10 silhouette scores
##################################################################################

#if you have memory problems,you can go one by one, modifying the max_cluster 
#variable with the goal of only performing one iteration in the cycle, testing 
#for a given number of clusters

max_clusters = 2
silhouette_scores = []

best_n_clusters = 0
best_silhouette_avg = -1
best_clusters = None

for n_clusters in range(2, max_clusters + 1):
    # Create a new model for each number of clusters
    model = Model("FuzzyClusteringAlex")
    n = DM1.shape[0]
    C = n_clusters
    m = 2
    eps = 0.05
    u = 0.6

    # Decision variables
    u_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik", lb=0, ub=1)
    z_jk = model.addVars(n, C, vtype=GRB.BINARY, name="z_jk")
    x_ijk = model.addVars(n, n, C, vtype=GRB.BINARY, name="x_ijk")
    w_ik = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="w_ik")

    # Auxiliary variables for u_ik^m 
    u_ik_m = model.addVars(n, C, vtype=GRB.CONTINUOUS, name="u_ik_m")

    # Constraints to define u_ik_m as (u_ik)^m for m=2, otherwise Gurobi does not work
    if m == 2:
        model.addConstrs((u_ik_m[i, k] == u_ik[i, k] * u_ik[i, k] for i in range(n) for k in range(C)), "u_ik_m_def")

    # Constraints
    model.addConstrs((quicksum(u_ik[i, k] for k in range(C)) == 1 for i in range(n)), "sum_u_ik")
    model.addConstrs((u_ik[j, k] <= (u - eps) * (1 - z_jk[j, k]) + z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_lower")
    model.addConstrs((u_ik[j, k] >= (u + eps) * z_jk[j, k] for j in range(n) for k in range(C)), "u_jk_upper")
    model.addConstrs((quicksum(x_ijk[i, j, k] for j in range(n) if j != i) == 1 for i in range(n) for k in range(C)), "one_closest")
    model.addConstrs((w_ik[i, k] >= DM1[i, j] * z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "w_ik_complete")
    # for i in range(n):
    #           for k in range(C):
    #               sum_distances = quicksum(DM1[i, j] * z_jk[j, k] for j in range(n) if j != i)
    #               sum_memberships = quicksum(z_jk[j, k] for j in range(n) if j != i)
    #               model.addConstr(w_ik[i, k] * sum_memberships == sum_distances, f"w_ik_average_{i}_{k}")

    model.addConstrs((x_ijk[i, j, k] <= z_jk[j, k] for i in range(n) for j in range(n) for k in range(C) if j != i), "x_z")
    model.addConstrs((quicksum(z_jk[j, k] for j in range(n)) >= 24 for k in range(C)), "at_least_u")
    model.addConstrs((quicksum(z_jk[i, k] for k in range(C)) >= 1 for i in range(n)), "at_least_one_high_membership_cluster")

#try with c=3 and some idea on my mind (don't use)
    # model.addConstr(quicksum(z_jk[j, 0] for j in range(n)) == 48, "cluster_1_size")
    # model.addConstr(quicksum(z_jk[j, 1] for j in range(n)) == 96, "cluster_2_size")
    # model.addConstr(quicksum(z_jk[j, 2] for j in range(n)) == 96, "cluster_3_size")

    # Objective function
    objective = quicksum(u_ik_m[i, k] * w_ik[i, k] for i in range(n) for k in range(C))
    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('NonConvex', 2) # NonConvex parameter
    model.setParam('Timelimit', 3600) 
    model.optimize()

    # To build the clusters
    clusters = np.zeros(n)
    for i in range(n):
        for k in range(C):
            if z_jk[i, k].x > 0.5:
                clusters[i] = k
                break

    # Calculate silhouette score
    score = silhouette_score(DM1, clusters, metric='precomputed')
    silhouette_scores.append(score)
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {score}")

    if score > best_silhouette_avg:
        best_silhouette_avg = score
        best_n_clusters = n_clusters
        best_clusters = clusters

# Plot the silhouette scores if the plot with all the iterations is used
#if the loop is not used (only one cluster is tested) comment the following code:
plt.figure(figsize=(10, 7))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

##################################################################################
#               silhouette plot for the best cluster clussification (K=2)
##################################################################################

sample_silhouette_values = silhouette_samples(DM1, best_clusters, metric='precomputed')

plt.figure(figsize=(10, 7))
y_lower = 10
for i in range(best_n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[best_clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.nipy_spectral(float(i) / best_n_clusters)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title(f'Silhouette Plot for {best_n_clusters} Clusters')
plt.xlabel('Silhouette Coefficient Values')
plt.ylabel('Cluster Label')
plt.axvline(x=best_silhouette_avg, color="red", linestyle="--")
plt.show()


##################################################################################
#              Plot the time series of the cluster 1 and 2 in case k=2
##################################################################################

hours_per_region = 24
regions = 10
hour_to_plot = 9 #because we want the 10

# Calculate the column indices for hour 10 for each region
hour_10_columns = [hour_to_plot + i * hours_per_region for i in range(regions)]

# Separate the columns into two groups
# Group 1: All regions except Sicilia (index 5) and Priolo Gargallo (index 9)
group_1_columns = [col for i, col in enumerate(hour_10_columns) if i not in [4, 8]]

# Group 2: Sicilia (index 5) and Priolo Gargallo (index 9)
group_2_columns = [hour_10_columns[4], hour_10_columns[8]]


legend_labels = ['CNOR', 'CSUD', 'NORD', 'SARD', 'SUD', 'BRNN', 'FOGN', 'ROSN']
#time series for Group 1
plt.figure(figsize=(14, 7))
for column, label in zip(group_1_columns, legend_labels):
    plt.plot(df.index, df[column], label=label, linewidth=0.35) 
plt.title('Time Series for Hour 10 (All regions except Sicilia and Priolo Gargallo)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.ylim(0, 200)  # Set y-axis limits
plt.legend(loc='upper right')
plt.show()

# Plot the time series for Group 2
plt.figure(figsize=(14, 7))
legend_lab2= ['SICI','PRGP']
for column,label  in zip(group_2_columns,legend_lab2):
    plt.plot(df.index, df[column], label=label, linewidth=0.4)  
# plt.plot(df.index, df[107], label=f'Series {column}', linewidth=0.4)
# plt.plot(df.index, df[203], label=f'Series {column}', linewidth=0.4)  
plt.title('Time Series for Hour 10 (Sicilia and Priolo Gargallo)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.ylim(0, 200)  # Set y-axis limits
plt.legend(loc='upper right')
plt.show()

