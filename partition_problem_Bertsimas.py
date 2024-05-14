from gurobipy import *
import pandas as pd
import numpy as np
import time
'''
Bertsimas's mixed-integer linear programming
'''

def optimization_exec(w,lbd,n,m,k):
    w = np.array(w)

    # Create a new model
    model = Model("model")
    model.setParam('TimeLimit', 60 * 60)
    # Create variables
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    d = model.addVar(name="d",lb=0,vtype=GRB.CONTINUOUS)

    # Set objective
    model.setObjective(d, GRB.MINIMIZE)

    # Add constraints
    for p in range(m - 1):
        for q in range(p + 1, m):
            Mpqs_sum = 0
            Vpqss_sum = 0
            for s in range(r):
                mu_ps = quicksum(w[i, s] * x[i, p] for i in range(n)) / k
                mu_qs = quicksum(w[i, s] * x[i, q] for i in range(n)) / k
                Mpqs = model.addVar(name=f"M_{p}_{q}_{s}")
                model.addConstr(Mpqs >= (mu_ps - mu_qs), name=f"M{p}_{q}_{s}_1")
                model.addConstr(Mpqs >= (mu_qs - mu_ps), name=f"M{p}_{q}_{s}_2")
                Mpqs_sum = Mpqs_sum + Mpqs

                for s_prime in range(r):
                    sigma_delta = quicksum(w[i, s]*w[i, s_prime]* (x[i, p]-x[i, q]) for i in range(n)) / k
                    Vpqss = model.addVar(name=f"V_{p}_{q}_{s}_{s_prime}")
                    model.addConstr(Vpqss >= sigma_delta, name=f"V_{p}_{q}_{s}_{s_prime}_1")
                    model.addConstr(Vpqss >= -1 * sigma_delta, name=f"V_{p}_{q}_{s}_{s_prime}_2")
                    Vpqss_sum = Vpqss_sum + Vpqss

            model.addConstr(d >= Mpqs_sum + lbd * Vpqss_sum, name=f"d_{p}_{q}")

    for p in range(m):
        model.addConstr(quicksum(x[i, p] for i in range(n)) == k)

    for i in range(n):
        model.addConstr(quicksum(x[i, p] for p in range(m)) == 1)

    for p in range(2, m):
        for i in range(p):
            model.addConstr(x[i, p] == 0)

    #Optimization model
    model.optimize()
    partition_res = model.getAttr('x',x)

    part_res = np.zeros((n, m))
    for key, value in partition_res.items():
        row = key[0]
        col = key[1]
        part_res[row, col] = value

    objective_value = d.X
    return part_res, objective_value


def calculate_discrenpency(part_res,w,lbd,n,m):
    w = np.array(w)

    #calculate the first and second moment of each group
    discrenpency_list=[]
    for i in range(m):
        p_result = part_res[:, i].reshape(n, 1)
        p_indices = list(np.where(p_result > 0.001)[0])
        p_element = w[p_indices, :]
        p_mean = np.mean(p_element, axis=0)
        p_cov = np.cov(p_element.T,bias=True)

        for j in range(i+1, m):
            q_result = part_res[:, j].reshape(n,1)
            q_indices = list(np.where(q_result > 0.001)[0])
            q_element = w[q_indices, :]
            q_mean = np.mean(q_element, axis=0)
            q_cov = np.cov(q_element.T, bias=True)
            var_diff = np.abs(q_cov - p_cov)
            mean_diff = np.abs(q_mean - p_mean)
            mean_diff_sum = np.sum(mean_diff)
            var_diff_sum = np.sum(var_diff)
            discrenpency = mean_diff_sum+lbd*var_diff_sum
            discrenpency_list.append(discrenpency)
    max_discrenpency = max(discrenpency_list)
    return max_discrenpency



if __name__=='__main__':
    # the number of group
    m = 3
    # the number of samples in each group
    k = 6
    # the dimension of samples
    r = 3
    # the number of samples
    n = m * k
    # the trade-off parameter
    lbd = 0.1

    objective_values = {'obj':[],'discrenpency':[],'solve_time':[]}

    #input data
    path_w = r'./data/Infant_Spasms_22_2d_normalized.csv'
    w = pd.read_csv(path_w)

    # group n samples into m groups and store the partitioning results
    t1 = time.time()
    partition_res, obj_value = optimization_exec(w,lbd,n,m,k)
    delta_time_min = (time.time() - t1) / 60.0
    partition_res_save_path = r'./result/partition_result_Bertsimas.npy'
    np.save(partition_res_save_path, partition_res)

    # store the partitioning information including objective value, discrepancy, and solving time
    max_discrenpency = calculate_discrenpency(partition_res, w, lbd, n, m)
    objective_values['obj'].append(obj_value)
    objective_values['discrenpency'].append(max_discrenpency)
    objective_values['solve_time'].append(delta_time_min)
    table_obj = pd.DataFrame(objective_values)
    table_obj_save_path = r'./result/partition_information_Bertsimas.csv'
    table_obj.to_csv(table_obj_save_path,index=False)