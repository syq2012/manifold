import encoder
import numpy as np
import torch
# import torch.nn as nn
import torch.utils.data as data

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt;
import time 

# epochs = 10
batch_size = 128
# lr = 0.008
device= 'cpu'

# multiplicative weight update
def get_diff(autoencoder, dataset):
    res = None
    for x in dataset:
        x = x.to(device)
        code, output = autoencoder(x.float())
        temp = output.detach().numpy() - x.numpy()
        # temp_norm = np.linalg.norm(temp, 2, axis=0)
        temp_norm = np.linalg.norm(temp, 2, axis=0)**2
#         print(temp_norm.shape)
        if res is None:
            res = temp_norm
        else:
            res += temp_norm
    # match weigthed MSE loss
    # res = np.sqrt(res) 
#     print(res)
    return res
            
        
# input: current autoencoder and the pytorch dataset to evaulate on
# output new weight
def update_weight(autoencoder, data, prev_weight, step_size):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
    l = len(prev_weight)
#     initalize result to be a vector of 1
#     res = [1]*l
    diff_vec = get_diff(autoencoder, dataset) * (1/len(data))

    print(diff_vec)
    res = prev_weight*np.exp(-1 * step_size * diff_vec)
#     res[res > 1/3] = 1/3
    total = np.sum(res)
#     print(res)
    print(total)
#     print(res[1]/total)

# smoothing
#     temp = res/total.tolist()
#     cap = (1/3) * 1.1
#     index = [temp > cap] 
#     temp[index] = cap
#     result = temp * (1 - cap * np.sum(index))/ (np.sum(temp) - cap * np.sum(index))
#     result[index] = cap
    result = res/total
    return result

# =========================================================================================================================================

def relative_entropy(p, q):
    return p * np.log(p/q)

# =========================================================================================================================================

def reweight_data(data, w):
    return data * w[:, None]

# def termination_crit(weight, epsilon):
#     return np.sum([(x < epsilon) or (x > 1 - epsilon) for x in weight])

# def construct_dataset(data):
    
def multi_weight_init(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round, init_weight):
#     inputs = ['data-dim', 'max epoch', 'code-dim', 'first-layer-dim: ', 'step-size']
    print('training with ' + 'data-dim: ' + str(d) + 'max epoch: ' + str(epoch) + 
         'code-dim: ' + str(cod_dim) + 'first-layer-dim: ' + str( first_layer_dim) + 'step-size: ' + str(step_size))
    
#     temp_data = get_dataset_from_list(data, 0, 100)
#     temp_valid = get_dataset_from_list(data, 100, 150)
#     test_data= get_dataset_from_list(data, 200, 500)
#     init weight 



    # cur_weight = np.ones(d) * (1/d)
    
#     cur_weight = np.array([0.001]*14 + [0.09/6]*6)
#     cur_weight = np.array([0.9] + [0.1/(d - 1)] * (d - 1))
    
#     i = np.random.randint(0, d)
    # i = 0
    # cur_weight = np.array([0.1/(d- 1)]*d)
    # cur_weight[i] = 0.9
    cur_weight = init_weight
    
    cur_input = reweight_data(data, cur_weight)
    cur_ae = None
    loss = []
    test_error = []
    itr = 0
#     total_round = 20
    total_round = num_round
    average_weight = np.copy(cur_weight)
    
#     store the autoencoder and weight that gives the lowest test error
#     test_ae = None
    test_w = [] 
    min_test_error = np.inf
    while (itr <= total_round):
        print(cur_weight)
#         train on reweighted data
#         cur_data = reweight_data(temp_data, cur_weight)
#         cur_valid = reweight_data(temp_valid, cur_weight)
        cur_data = encoder.get_dataset_from_list(cur_input, 0, 500)
        cur_valid = encoder.get_dataset_from_list(cur_input, 500, 550)
        cur_ae, cur_loss = encoder.training(cur_data, cur_valid, epoch, d, cod_dim, first_layer_dim)
        
        cur_test = encoder.get_dataset_from_list(cur_input, 600, 700)
        cur_test_data = torch.utils.data.DataLoader(cur_test, batch_size, shuffle=True)
        cur_test_error = encoder.test_err(cur_ae, cur_test_data)
        
        test_error.append(cur_test_error)
        average_weight += np.copy(cur_weight)
#         print(cur_weight)
        if cur_test_error <= min_test_error:
            min_test_error = cur_test_error
#             test_ae = cur_ae            
            test_w = cur_weight
#             print('update test_w here')
#             print(test_w)
            
#         update weight
        cur_weight = update_weight(cur_ae, cur_data, cur_weight, step_size)
#         cur_weight = np.round(cur_weight, decimals=3)
#         cur_weight = cur_weight * (1/np.sum(cur_weight))
            
#         cur_weight = cur_weight * 0.5
        cur_input = reweight_data(data, cur_weight)
        
#         print(np.argsort(cur_weight)[-4:])
        
#         print(np.max(cur_weight))
#         print(np.max(cur_weight) - np.min(cur_weight))
        loss.append(cur_loss[-1])
        

#         print(np.sum(cur_weight))
        itr += 1
#     print(test_error)   
#     print("average_weight is")
#     print(1/total_round * (average_weight))
    # print(test_w)
    return cur_ae, cur_weight,1/total_round * average_weight, loss, test_error, test_w


# def construct_dataset(data):
    
def multi_weight_weightedMSE(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round, init_weight):
#     inputs = ['data-dim', 'max epoch', 'code-dim', 'first-layer-dim: ', 'step-size']
    print('training with ' + 'data-dim: ' + str(d) + 'max epoch: ' + str(epoch) + 
         'code-dim: ' + str(cod_dim) + 'first-layer-dim: ' + str( first_layer_dim) + 'step-size: ' + str(step_size))
    
#     temp_data = get_dataset_from_list(data, 0, 100)
#     temp_valid = get_dataset_from_list(data, 100, 150)
#     test_data= get_dataset_from_list(data, 200, 500)
#     init weight 
    cur_data = encoder.get_dataset_from_list(data, 0, 500)
    cur_valid = encoder.get_dataset_from_list(data, 500, 550)
    cur_test = encoder.get_dataset_from_list(data, 600, 700)

    #     add dynamic plot as data generates
    plt.ion()
    fig=plt.figure()
    # plt.axis([0,d,0,2/d])
    x = [i for i in range(d)]
#     cur_weight = np.ones(d) * (1/d)
    
#     cur_weight = np.array([0.001]*14 + [0.09/6]*6)
#     cur_weight = np.array([0.9] + [0.1/(d - 1)] * (d - 1))
    
#     i = np.random.randint(0, d)
    cur_weight = init_weight
    
#     cur_input = reweight_data(data, cur_weight)
    cur_ae = None
    loss = []
    test_error = []
    itr = 0
#     total_round = 20
    total_round = num_round
    average_weight = np.copy(cur_weight)
    
#     store the autoencoder and weight that gives the lowest test error
#     test_ae = None
    test_w = [] 
    min_test_error = np.inf
    while (itr <= total_round):
        print(cur_weight)
#         train on reweighted data
#         cur_data = reweight_data(temp_data, cur_weight)
#         cur_valid = reweight_data(temp_valid, cur_weight)
        
        cur_ae, cur_loss = encoder.training_weighted_MSE(cur_data, cur_valid, epoch, d, cod_dim, first_layer_dim, cur_weight)
        
        
        cur_test_data = torch.utils.data.DataLoader(cur_test, batch_size, shuffle=True)
        cur_test_error = encoder.test_err(cur_ae, cur_test_data)
        # print(cur_test_error)
        test_error.append(cur_test_error)
        average_weight += np.copy(cur_weight)
#         print(cur_weight)
        if cur_test_error <= min_test_error:
            min_test_error = cur_test_error
#             test_ae = cur_ae            
            test_w = cur_weight
#             print('update test_w here')
#             print(test_w)
        pl.plot(x, cur_weight, 'o--') 
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)
        
#         update weight
        cur_weight = update_weight(cur_ae, cur_data, cur_weight, step_size)
               
#         print(np.argsort(cur_weight)[-4:])
        
#         print(np.max(cur_weight))
#         print(np.max(cur_weight) - np.min(cur_weight))
        loss.append(cur_loss[-1])
        

#         print(np.sum(cur_weight))
        itr += 1
#     print(test_error)   
#     print("average_weight is")
#     print(1/total_round * (average_weight))
    # print(test_w)
    return cur_ae, cur_weight,1/total_round * average_weight, loss, test_error, test_w
    
    
 
def multi_weight(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round):
    # weight = np.ones(d) * (1/d)
    weight = np.array([0.1] + [0.9/199]*199)

        # weight = np.array([0.001]*14 + [0.09/6]*6)
#     weight = np.array([0.9] + [0.1/(d - 1)] * (d - 1))
    
#     i = np.random.randint(0, d)
    # i = 0
    # weight = np.array([0.1/(d- 1)]*d)
    # weight[i] = 0.9
    return multi_weight_init(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round, weight)

