This github project includes all the scripts that have been used for the realization of my final degree project. In this READ.me there will be a brief description of all of them:

-matrices_scenarios.py: It is a script in which all the matrices of the 7 small scenarios presented in the paper are contained.

-gurobi2_final.py: Contains the non-linear optimisation programs representing the creation of the 3 fuzzy algorithms.There is only one optimisation model, 
but the constraints introduced by the other two are commented, so that to use any of the three models the relevant restrictions must be uncommented or commented.

-factorial_data.py: It contains the necessary functions for creating the factorial data, and for the creation of the cophenetic distance matrices. 
Also the optimisation model of the previous script for use with the created matrices.

-LNS.py and LNS_factorial.py: These are the scripts in which the Large neighbourhood search metaheuristic algorithm is created, and applied to all the scenarios created.

-MC_crossdependent_scen.py: Code in which the montecarlo experiments are carried out to obtain the ARI of the different models in the cross dependency scenarios.

-realdata.py: Application of our algorithm to real data from the Italian electricity market
