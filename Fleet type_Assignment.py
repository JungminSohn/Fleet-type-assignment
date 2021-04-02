#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
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

fleet= {"A": Fleet(fleet_type='A', casm=0.14, num_seats= 162, rasm= 0.17, num_airplane=9),
        "B": Fleet(fleet_type='B', casm=0.15, num_seats= 200, rasm= 0.17, num_airplane=6)}

data= pd.read_excel("flight_schedule.xlsx")

#calculating operating cost
for fleet_type in list(fleet.keys()):
    operating_cost= fleet[fleet_type].casm * data['Distance'] * fleet[fleet_type].num_seats
    data['type{}_OC'.format(fleet_type)] = operating_cost.astype(int)

#calculate spill cost
def L(z):
    return stats.norm.pdf(z,0,1) - z*(1- stats.norm.cdf(z))

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

#for fleet_type in list(fleet.keys()):
#    data["type{}_assigned".format(fleet_type)]=0

data.to_csv("fleet_type_assignment_cost_added.csv", encoding='cp949')


# In[2]:


#define 'index sets'
I = data['Flight_number'].astype('str')
J = ['A', 'B']

#airline_node (airport 별)
import numpy as np

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


# In[3]:


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
for fleet_type in J:
    num_palne=0 
    for airport in airport_list:
        last_node= airport_nodes[airport_list[-1]].iloc[-1,:]
        num_palne += Y['{}-{}'.format(airport_nodes[airport].iloc[-1,0], airport)][fleet_type]
    prob+= (num_palne<= fleet[fleet_type].num_airplane )

prob.solve()

print("Status:", LpStatus[prob.status])

result_dict = {str(v): int(v.value()) for v in prob.variables()}


# In[4]:


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
print(data['Cost($)'].sum())
data= data.append({"Flight_number":"Total Cost", "Cost($)": data['Cost($)'].sum()},ignore_index=True)

data.to_csv("Case1_result.csv", encoding='cp949', index= False)

