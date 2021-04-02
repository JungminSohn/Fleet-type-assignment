#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from scipy import stats
from pulp import *


class Fleet:
    def __init__(self, fleet_type=None, casm=None, num_seats=None, rasm=None, 
                 num_airplane=None):
        self.type=fleet_type
        self.casm=casm
        self.num_seats= num_seats
        self.rasm= rasm
        self.num_airplane= num_airplane
        
def L(z):
    return stats.norm.pdf(z,0,1) - z*(1- stats.norm.cdf(z))

def Optimization(fleet):
    data= pd.read_excel("flight_schedule.xlsx")
    #calculating operating cost
    for fleet_type in list(fleet.keys()):
        operating_cost= fleet[fleet_type].casm * data['Distance'] * fleet[fleet_type].num_seats
        data['type{}_OC'.format(fleet_type)] = operating_cost.astype(int)

    #calculate spill cost
    for fleet_type in list(fleet.keys()):
        Z= (fleet[fleet_type].num_seats - data['Demand'])/data['S.D']
        expected_spill_num= L(Z) * data['S.D']
        spill_cost= expected_spill_num * fleet[fleet_type].rasm * data['Distance']
        modified_spill_cost= spill_cost *0.85
        data['type{}_M_spill_cost'.format(fleet_type)]= modified_spill_cost.astype(int)

    #calculate total cost
    for fleet_type in list(fleet.keys()):
        total_cost= data['type{}_OC'.format(fleet_type)] + data['type{}_M_spill_cost'.format(fleet_type)]
        data['type{}_Total_cost'.format(fleet_type)]= total_cost 

    # data.to_csv("fleet_type_assignment_cost_added.csv", encoding='cp949')

    #define 'index sets'
    I = data['Flight_number'].astype('str')
    J = ['A', 'B']

    #airline_node (airport 별)


    airport_list = list(set(data[['Destination', 'Origin']].values.reshape(-1)))

    airport_nodes={}
    for airport in airport_list:
        node_idx = np.where(data[['Destination', 'Origin']] == airport)[0]
        # get nodes list at the airport
        airport_node = data[['Flight_number',
                              'Origin',
                              'Departure_Time',
                              'Destination',
                              'Arrival_Time']].iloc[node_idx, :]
        airport_node= airport_node.reset_index(drop=True)
        airport_node['time'] = np.where(airport_node.Origin == airport, airport_node.Departure_Time, airport_node.Arrival_Time)
        airport_node['S'] = np.where(airport_node.Origin == airport, -1, 1 )
        node_sorting = airport_node.time.argsort()
        airport_node = airport_node[['Flight_number',
                              'Origin',
                              'Destination','time','S']].iloc[node_sorting,:]
        airport_node = airport_node.reset_index(drop=True)
        airport_nodes[airport]=airport_node


    K=[]
    for airport in airport_list:
        for flight in range(len(airport_nodes[airport])):
            K.append(str(airport_nodes[airport].loc[flight].Flight_number)+"-"+str(airport_nodes[airport].loc[flight].Origin))        
            K.append(str(airport_nodes[airport].loc[flight].Flight_number)+"-"+str(airport_nodes[airport].loc[flight].Destination)) 

    #define cost
    cost = {}
    for i in I:
        for j in J:
            cost[i,j] = int(data.loc[data.Flight_number == int(i)]['type{}_Total_cost'.format(j)])

    prob= LpProblem("Fleet_type", LpMinimize)

    X = LpVariable.dicts(name="X", indexs=(I, J), lowBound=0, upBound=1, cat='Integer')
    Y = LpVariable.dicts(name="Y", indexs=(K, J), lowBound=0, cat='Integer')

    #목적함수
    obj_func= lpSum(cost[i,j]* X[i][j] for i in I for j in J)
    prob+= obj_func

    #첫번째 제약식
    for i in I:
        prob+= (X[i]['A'] + X[i]['B']==1)

    #두번째 제약식
    for airport in airport_list:
        for node in range(len(airport_nodes[airport])):
            for fleet_type in J:
                previous_y= Y['{}-{}'.format(airport_nodes['{}'.format(airport)].iloc[node-1, 0], airport)][fleet_type]
                current_y = Y['{}-{}'.format(airport_nodes['{}'.format(airport)].iloc[node, 0], airport)][fleet_type]
                current_x = X['{}'.format(airport_nodes[airport].iloc[node, 0])][fleet_type]
                s = airport_nodes[airport].iloc[node,-1]
                prob+=( previous_y +  s*current_x  == current_y )


    #세번째 제약식
    last_node_list=[]
    for fleet_type in J:
        num_plane=0 
        last_nodes=[]
        for airport in airport_list:
            num_plane += Y['{}-{}'.format(airport_nodes[airport].iloc[-1,0], airport)][fleet_type]
            last_nodes.append(Y['{}-{}'.format(airport_nodes[airport].iloc[-1,0], airport)][fleet_type])
        last_node_list.append(last_nodes)
        prob+= (num_plane<= fleet[fleet_type].num_airplane )
    
    prob.solve()

    #print("Status:", LpStatus[prob.status])

    result_dict = {str(v): int(v.value()) for v in prob.variables()}

    result_list=[]
    for fleet_type in J:
        result_each_type=[]
        for flight in data.Flight_number:
            for result_key in result_dict.keys():
                if result_key == "X_{}_{}".format(flight, fleet_type):
                    result_each_type.append(result_dict["X_{}_{}".format(flight, fleet_type)])
        result_list.append(result_each_type)

    data['typeA_assigned'] = result_list[0]
    data['typeB_assigned'] = result_list[1]

    data= data.sort_values("Flight_number",ascending=True)
    data['Cost($)']=(data.typeA_assigned * data.typeA_Total_cost) + (data.typeB_assigned * data.typeB_Total_cost)
    #print(data['Cost($)'].sum())
    # data= data.append({"Flight_number":"Total Cost", "Cost($)": data['Cost($)'].sum()},ignore_index=True)
    # data.to_csv("Case1_result.csv", encoding='cp949', index= False)
    
    return LpStatus[prob.status], data['Cost($)'].sum(), last_node_list, result_dict


# In[7]:


if __name__ == "__main__":
    num_a_list=[]
    num_b_list=[]
    used_a_list=[]
    used_b_list=[]
    status_list=[]
    cost_list=[]
    for num_a in range(0,50):
        for num_b in range(0,50):
            fleet_input= {"A": Fleet(fleet_type='A', casm=0.14, num_seats= 162, rasm= 0.17, num_airplane=num_a),
                "B": Fleet(fleet_type='B', casm=0.15, num_seats= 200, rasm= 0.17, num_airplane=num_b)}

            status, cost, last_nodes, result= Optimization(fleet_input)
            
            used_fleet=[]
            for fleet_type in range(2):
                count=0
                for final_node in last_nodes[fleet_type]:
                    count+=result[str(final_node)]
                used_fleet.append(count)
            
            num_a_list.append(num_a)
            num_b_list.append(num_b)
            used_a_list.append(used_fleet[0])
            used_b_list.append(used_fleet[1])
            status_list.append(status)
            cost_list.append(cost)
    grid_search={"number_of_a": num_a_list, "number_of_b": num_b_list,"number_of_used_a": used_a_list, "number_of_used_b": used_b_list, "status": status_list, "Total cost": cost_list}
    grid_result= pd.DataFrame(grid_search)
    print(grid_result)
    grid_result.to_csv("grid_result_5.csv", encoding='cp949')


# In[ ]:




