import encoder
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import pylab as pl
from IPython import display
import time 

# epochs = 10
batch_size = 128
# lr = 0.008
device= 'cpu'

# multiplicative weight update
def get_diff(autoencoder, dataset):
    res = None
    for x, index in dataset:
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
    print('llendata is', len(data))
    diff_vec = get_diff(autoencoder, dataset) * (1/len(data))
    # diff_vec = get_diff(autoencoder, dataset)

    # print(diff_vec)
    res = prev_weight*np.exp(-1  * step_size * diff_vec)
    # print(res)
#     res[res > 1/3] = 1/3
    total = np.sum(res)
#     print(res)
    # print(total)
#     print(res[1]/total)


    # res[res > 1] = 1
    # total = np.sum(res)

    result = res/total
    return result

def get_diff_matrix(autoencoder, dataset):
    res = []
    # cell_index = []

    for x , index in dataset:
        x = x.to(device)
        # print(index)
        code, output = autoencoder(x.float())
        temp = output.detach().numpy() - x.numpy()
        # print(index)
        # print(len(temp))
        # print(temp.shape)
        res.append(np.array(temp)**2)
        # cell_index += index.tolist()
            # cell_index = np.concatenate((cell_index, index.numpy()), axis = 1)
    return np.array(res)

def update_weight_gene_cell(autoencoder, data, prev_weight, prev_weight_cell, step_size):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=False)

    diff_list = get_diff_matrix(autoencoder, dataset)
    diff = np.concatenate(tuple(diff_list), axis = 0)
    # print(np.sum(diff))
    reweight_cell = prev_weight_cell[:, None] * diff
    exp_gene = np.sum(reweight_cell, axis = 0)
    res_gene = prev_weight * np.exp((-1)*step_size * exp_gene)
    # print(exp_gene)
    tot_gene = np.sum(res_gene)
    # print(tot_gene)

    reweight_gene = prev_weight[None, :] * diff
    exp_cell = np.sum(reweight_gene, axis = 1)
    res_cell = prev_weight_cell * np.exp((-1)*step_size * exp_cell)
    tot_cell = np.sum(res_cell)

    return res_gene/tot_gene, res_cell/tot_cell

def update_weight_gene_or_cell(autoencoder, data, prev_weight, prev_weight_cell, step_size, ifgene):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=False)
    diff_list = get_diff_matrix(autoencoder, dataset)
    diff = np.concatenate(tuple(diff_list), axis = 0)
    if ifgene:
        reweight_cell = prev_weight_cell[:, None] * diff
        exp_gene = np.sum(reweight_cell, axis = 0)
        res_gene = prev_weight * np.exp((-1)*step_size * exp_gene)
        tot_gene = np.sum(res_gene)
        return res_gene/tot_gene
    else:
        reweight_gene = prev_weight[None, :] * diff
        exp_cell = np.sum(reweight_gene, axis = 1)
        res_cell = prev_weight_cell * np.exp((-1)*step_size * 100 * exp_cell)
        tot_cell = np.sum(res_cell)
        return res_cell/tot_cell


# =========================================================================================================================================

def update_weight_MSE(autoencoder, data, prev_weight, step_size):
    dataset = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
    var_weight = torch.tensor(prev_weight, requires_grad = True)
    d = len(prev_weight)
    criterion = nn.MSELoss()
    

    result  = np.zeros(d)
    for x, index in dataset:
        m, d = x.shape
        weighted_x = var_weight * x
        code, output = autoencoder(weighted_x.float())
        cur_loss = criterion(output, weighted_x.float())
        grad = torch.autograd.grad(cur_loss, var_weight)
        # print(result)
        result += grad[0].detach().numpy() 

    exps =  result
    res = prev_weight * np.exp((-1) * step_size * exps)
    # print(res)
    total = np.sum(res)
    return res/total
        # grad = [torch.autograd.functional.jacobian(lambda x: autoencoder(x.float())[1], weighted_x)]
        # diff_grad = [np.dot((np.array(grad[i]) - identity), np.diag(x[i, :].numpy())) for i in range(m)]
        # diff_output = 
             

# =========================================================================================================================================

def relative_entropy(p, q):
    return p * np.log(p/q)

# =========================================================================================================================================

def reweight_data(data, w):
    return data * w[:, None]

def reweight_data_cell(data, w):
    return data * w[None, :]

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
    plt.ion()
    fig=plt.figure()
    # plt.axis([0,d,0,2/d])
    x = [i for i in range(d)]

    cur_weight = init_weight
    
    # cur_input = reweight_data(data, cur_weight)
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
        index = np.mod(itr, 5)
        index2 = np.mod(itr + 1, 5)
        cur_train_input = reweight_data(data[index* 200:200*(index+ 1)], cur_weight)
        cur_valid_input = reweight_data(data[index2 * 200: 50 + index2 * 200], cur_weight)
        cur_test_input = reweight_data(data[50 + index2 * 200:100 + index2 * 200], cur_weight)

        # cur_data = encoder.get_dataset_from_list(cur_input, index* 200, 200*(index+ 1))
        # cur_valid = encoder.get_dataset_from_list(cur_input, index2 * 200, 50 + index2 * 200)
        # cur_test = encoder.get_dataset_from_list(cur_input, 50 + index2 * 200, 100 + index2 * 200)

        cur_data = encoder.get_dataset_from_list(cur_train_input, 0, len(cur_train_input))
        cur_valid = encoder.get_dataset_from_list(cur_valid_input, 0, len(cur_valid_input))
        cur_test = encoder.get_dataset_from_list(cur_test_input, 0, len(cur_test_input))
        
        cur_ae, cur_loss = encoder.training(cur_data, cur_valid, epoch, d, cod_dim, first_layer_dim)
        
        # cur_test = encoder.get_dataset_from_list(cur_input, 600, 700)
        cur_test_data = torch.utils.data.DataLoader(cur_test, batch_size, shuffle=False)
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
        cur_weight = update_weight_MSE(cur_ae, encoder.get_dataset_from_list(data,index* 200, 200*(index+ 1)), cur_weight, step_size)
        # cur_weight[cur_weight < 1/d ] = 0
        # cur_weight = (1/np.sum(cur_weight)) * cur_weight
        pl.plot(x, cur_weight, 'o--') 
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)
#         cur_weight = np.round(cur_weight, decimals=3)
#         cur_weight = cur_weight * (1/np.sum(cur_weight))
            
#         cur_weight = cur_weight * 0.5
        # cur_input = reweight_data(data, cur_weight)
        
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
    
def multi_weight_weightedMSE(data, d, epoch, cod_dim, first_layer_dim, step_size, num_round, init_weight, init_weight_cell):
#     inputs = ['data-dim', 'max epoch', 'code-dim', 'first-layer-dim: ', 'step-size']
    print('training with ' + 'data-dim: ' + str(d) + 'max epoch: ' + str(epoch) + 
         'code-dim: ' + str(cod_dim) + 'first-layer-dim: ' + str( first_layer_dim) + 'step-size: ' + str(step_size))
    

    # cur_data = encoder.get_dataset_from_list(data, 0, 500)
    # cur_valid = encoder.get_dataset_from_list(data, 500, 550)
    # cur_test = encoder.get_dataset_from_list(data, 600, 700)

    #     add dynamic plot as data generates
    plt.ion()
    fig=plt.figure()
    # plt.axis([0,d,0,2/d])
    x = [i for i in range(d)]
    cur_weight = init_weight
    cur_weight_cell = init_weight_cell
    threshold = 1 / len(init_weight[init_weight > 0]) * (0.99)
    cur_ae = None
    loss = []
    test_error = []
    itr = 0
#     total_round = 20
    total_round = num_round
    average_weight = np.copy(cur_weight)
    max_loss = np.inf
    
#     store the autoencoder and weight that gives the lowest test error
#     test_ae = None
    test_w = [] 
    min_test_error = np.inf
    while (itr <= total_round):
        print('cur iteration is ' + str(itr))
        # print(cur_weight)
#         train on reweighted data
        # index = np.mod(itr, 5)
        # index2 = np.mod(itr + 2, 5)
        index = 0
        index2 = 2
        cur_data = encoder.get_dataset_from_list(data, index* 200, 200*index+ 10)
        cur_valid = encoder.get_dataset_from_list(data, index2 * 200, 5 + index2 * 200)
        # cur_test = encoder.get_dataset_from_list(data, 450, 500)
        
        cur_ae, cur_loss = encoder.training_weighted_MSE(cur_data, cur_valid, epoch, d, cod_dim, first_layer_dim, cur_weight, cur_weight_cell, max_loss)
        # max_loss = cur_loss[-1]
        # print(max_loss)
        cur_dataset = torch.utils.data.DataLoader(cur_data, batch_size, shuffle=True)
        # cur_loss = encoder.test_err_weighted(cur_ae, cur_dataset, cur_weight, cur_weight_cell)
        loss.append(cur_loss[-1])
        # cur_test_data = torch.utils.data.DataLoader(cur_test, batch_size, shuffle=True)
        # cur_test_error = encoder.test_err_weighted(cur_ae, cur_test_data, cur_weight, cur_weight_cell)
        # print(cur_test_error)
        # test_error.append(cur_test_error)
        average_weight += np.copy(cur_weight)

#         print(cur_weight)
#         if cur_test_error <= min_test_error:
#             min_test_error = cur_test_error
# #             test_ae = cur_ae            
#             test_w = cur_weight
# #             print('update test_w here')
# #             print(test_w)

#         update weight
        # get_diff_matrix(cur_ae, cur_data)
        if np.mod(itr, 2) == 0 or np.mod(itr, 2) == 1:
            cur_weight = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell, step_size, True)
        else:
            cur_weight_cell = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell, step_size, False)
            # print(np.sum(cur_weight_cell[0:400000]))
        cur_loss = encoder.test_err_weighted(cur_ae, cur_dataset, cur_weight, cur_weight_cell)
        
        # new_weight = update_weight(cur_ae, cur_data, cur_weight, step_size)

        # new_weight = update_weight_gene_or_cell(cur_ae, cur_data, cur_weight, cur_weight_cell, step_size, True)
        # cur_loss = encoder.test_err_weighted(cur_ae, cur_dataset, new_weight, cur_weight_cell)
        # cur_weight = new_weight

        max_loss = cur_loss

        loss.append(cur_loss)
        pl.plot(x, cur_weight, 'o--') 
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)
        
        
        # print(np.var(cur_weight_cell))

        # if itr > 1:
        # threshold = ((np.max(cur_weight) + np.min(cur_weight[cur_weight > 0]))/2) * (1 - 0.01*itr)
        # threshold = np.mean(cur_weight[cur_weight > 0]) * 0.95
        # print(threshold)


        cur_weight = round(cur_weight, threshold)



#         print(np.argsort(cur_weight)[-4:])
        
#         print(np.max(cur_weight))
#         print(np.max(cur_weight) - np.min(cur_weight))
        # loss.append(cur_loss[-1])
        

#         print(np.sum(cur_weight))
        itr += 1
#     print(test_error)   
#     print("average_weight is")
#     print(1/total_round * (average_weight))
    # print(test_w)
    return cur_ae, cur_weight,1/total_round * average_weight, loss, test_error, test_w, cur_weight_cell
    
    
 
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

# =========================================================================================================================================

def round(w, threshold):
    # r = np.max(w) - np.min(w[w > 0])
    # threshold = 1/len(w[w > 0]) - r * 0.1
    w[w < threshold] = 1/len(w) * 0.01
    return w/np.sum(w)


def find_subset(w):
    threshold = np.mean(w[w > 0.001]) * 0.95
    return np.array([w > threshold], dtype = int)

def True_positive(w, true_w):
    real = np.sum(true_w)
    computed = np.sum(w*true_w)
#     print(w*true_w)
    return computed/real

def False_positive(w, true_w):
    n = len(w)
    complement = np.ones(n) - true_w
    return True_positive(w, complement)


