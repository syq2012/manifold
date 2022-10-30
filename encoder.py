import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import dataset_gen

import random

seed = 121
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Settings
epochs = 30
batch_size = 64
# batch_size = 10

# lr = 0.008
device = "cuda" if torch.cuda.is_available() else "cpu"
# DataLoader
# read the first l set of samples from file, 
# each set of sample is of size d by n
def get_dataset(file, d, n, l, h):
    raw_data = dataset_gen.read(file, d, n)
    # print(raw_data[0].shape)
    raw_inputs = np.concatenate(tuple([raw_data[i] for i in range(l, h)]), axis = 1)
    dataset = dataset_gen.cellDataset(raw_inputs)
    return dataset
# input is a list of numpy matrices and the range of data to read
def get_dataset_from_list(raw_data, l, h):
#     raw_data = dataset_gen.read(file, d, n)
    # print(raw_data[0].shape)
    raw_inputs = np.concatenate(tuple([raw_data[i] for i in range(l, h)]), axis = 1)
    dataset = dataset_gen.cellDataset(raw_inputs)
    return dataset  
# load datas as pytorch dataset
def get_torchdataset(datas):
    return torch.utils.data.DataLoader(datas, batch_size,
            shuffle=True)
# =========================================================================================================================================


class custom_discrete(nn.Module):
    #the init method takes the parameter:
    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim

    #the forward calls it:
    def forward(self, x):
        multiplier = torch.tensor([2 ** i for i in range(self.dim)]).to(device)
        temp = torch.sum(x * multiplier, dim = 1)
        return torch.stack([temp**i for i in range(self.output_dim)], dim = 1).to(device)

class custom_sin(nn.Module):
    #the init method takes the parameter:
    def __init__(self):
        super().__init__()
#         self.dim = dim
#         self.output_dim = output_dim

    #the forward calls it:
    def forward(self, x):
        
        return torch.sin(x)
# =========================================================================================================================================

# Model structure
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
        data_dim = kwargs["input_shape"]
        code_dim = kwargs["code_dim"]
        list_dim = kwargs["list_dim"]
        complete_layer_dim = [data_dim]+list_dim
        layer_list = []
        for i in range(len(complete_layer_dim) - 1):
          layer_list.append(nn.Linear(complete_layer_dim[i], complete_layer_dim[i + 1]))
          layer_list.append(nn.ReLU6())
#           layer_list.append(nn.LeakyReLU(0.1))
#           layer_list.append(nn.Dropout(p=0.5 - 0.1*i))
#           layer_list.append(nn.Tanh())
#           layer_list.append(torch.nn.AlphaDropout(0.6))
        layer_list.append(nn.Linear(complete_layer_dim[-1], code_dim))
#         layer_list.append(nn.Sigmoid())

        decoder_layers = [nn.Linear(code_dim, complete_layer_dim[-1])]
#         decoder_layers = [custom_discrete(code_dim, 3), nn.Linear(3, complete_layer_dim[-1])]
#         decoder_layers = [nn.Linear(code_dim, code_dim), custom_sin(), nn.Linear(code_dim, complete_layer_dim[-1])]
        for i in range(1, len(complete_layer_dim)):
          decoder_layers.append(nn.ReLU6())
#           decoder_layers.append(nn.Tanh())
#           decoder_layers.append(nn.Sigmoid())
#           decoder_layers.append(nn.Dropout(p=0.5 - 0.1*i))
          decoder_layers.append(nn.Linear(complete_layer_dim[-i], complete_layer_dim[-(i + 1)]))
#           decoder_layers.append(nn.ReLU6())
#         decoder_layers.append(nn.ReLU6())
        
#         decoder_layers = [nn.Linear(code_dim, data_dim)]
#         decoder_layers.append(custom_sin())
        
                               
        self.encoder = nn.Sequential(*layer_list)
        self.decoder = nn.Sequential(*decoder_layers)
#        print(self.encoder)
#        print(self.decoder)
        # first_layer_dim = kwargs["first_layer_dim"]
        # mid_layer_dim = int(first_layer_dim / 2)
        # final_layer_dim = int(mid_layer_dim / 2)
        # print(mid_layer_dim)
        # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(data_dim, first_layer_dim),
# #             nn.Tanh(),
# #             nn.Linear(128, 64),
#             nn.ReLU6(),
#             nn.Linear(first_layer_dim, mid_layer_dim),
#             nn.ReLU6(),
#             # nn.Linear(mid_layer_dim, final_layer_dim),
#             # nn.ReLU6(),
# #             nn.Linear(64, 16),
# #             nn.ReLU6(),
            
#             nn.Linear(mid_layer_dim, code_dim),
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(code_dim, mid_layer_dim),
#             nn.ReLU6(),
# #             nn.Linear(16, 64),
# #             nn.ReLU6(),
# #             nn.Linear(64, 128),
# #             nn.Tanh(),
#             # nn.Linear(final_layer_dim, mid_layer_dim),
#             # nn.ReLU6(),
#             nn.Linear(mid_layer_dim, first_layer_dim),
#             nn.ReLU6(),
#             nn.Linear(first_layer_dim, data_dim),
# #             nn.Sigmoid()
#         )

    def forward(self, inputs):
#         print(inputs.shape)
        codes = self.encoder(inputs)
        decoded = self.decoder(codes *3)

        return codes, decoded
#         activation = self.encoder_hidden_layer(inputs)
#         activation = torch.relu(activation)
#         code = self.encoder_output_layer(activation)
#         code = torch.relu(code)
#         activation = self.decoder_hidden_layer(code)
#         activation = torch.relu(activation)
#         activation = self.decoder_output_layer(activation)
#         reconstructed = torch.relu(activation)
#         return reconstructed


# =========================================================================================================================================

def MSE_transpose(output, target):
    return (1/batch_size) * torch.sum((output - target)**2)

def training(train_dataset, valid_dataset, epochs, input_length, code_dim, first_layer_dim):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  get dataloader 
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    list_dimension = [first_layer_dim, int(first_layer_dim / 2)]
    model = AE(input_shape=input_length, code_dim = code_dim, first_layer_dim = first_layer_dim, list_dim = list_dimension).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    learning_rate= 1e-5
#     learning_rate= 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    # mean-squared error loss
    criterion = nn.MSELoss()
#     criterion = weighted_MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
#     res_loss = None
    for epoch in range(epochs):
        loss = 0
        for x, index in train_data:
            x = x.to(device)
            
            code, outputs = model(x.float())
            train_loss = criterion(outputs, x.float())
#             train_loss = weighted_MSELoss(outputs, x.float(), weight)
            
            train_loss.backward()
            optimizer.step()
        
            loss += train_loss.item()
        # loss = loss / len(train_data)
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss)
        loss_list.append(loss)
#         loss_list.append(train_loss.item())
        cur_valid_err = test_err(model, valid_data)
        if cur_valid_err >= valid_error:
            break
        else:
            valid_error = cur_valid_err
#             print(test_err(model, valid_data))
#             if res_ae != None:
#                 print(test_err(res_ae, valid_data))
            res_ae = model
            loss_list.append(train_loss.item())
#             res_loss = loss_list
    return res_ae, loss_list


def weighted_MSELoss(output, target, weight, weight_cell):
    weight_tensor = torch.tensor(weight).to(device)
#     print("cur target is")
#     print(output - target)
    
    # temp = (1/batch_size) * (output - target) ** 2
    # print(torch.sum(temp, dim = 0).size())
    (m, n) = output.size()
#     print(torch.tensor(weight_cell).to(device))
#     print(output.size())
#     print(weight_tensor.size())
#     print((weight_tensor *  (output - target) ** 2).shape)
    return torch.sum(torch.tensor(weight_cell).to(device)[:, None] * (weight_tensor *  ((output - target) ** 2)))
    # return (1/m) * torch.sum(weight_tensor *  (output - target) ** 2)
    
    
def weighted_MSELoss_ignore0(output, target, weight, weight_cell):
    weight_tensor = torch.tensor(weight).to(device)
    # temp = (1/batch_size) * (output - target) ** 2
    # print(torch.sum(temp, dim = 0).size())
#     (m, n) = output.size()
    temp = (output - target)
    temp[target == 0] = 0
#     ms = torch.sum([target != 0][0], axis = 1)
#     num = torch.sum([target > 0][0])
#     print(torch.tensor(weight_cell).to(device))
#     print((weight_tensor *  (output - target) ** 2).shape)
    return torch.sum(torch.tensor(weight_cell).to(device)[:, None] * (weight_tensor *  temp ** 2))
#     return torch.sum(weight_tensor *  temp ** 2)

# 
# def time_avg_loss(output, avg_target, time_avg_dic, weight, weight_cell):
    

import torch.nn.functional as F

def sparse_loss(model, data):
    model_children = list(model.children())
    loss = 0
    values = data
    for i in range(len(model_children)):
        values = (model_children[i](values))
        loss += torch.mean(torch.abs(values))
    return loss * (1/len(model_children))

def training_weighted_MSE(train_dataset, valid_dataset, epochs, input_length, code_dim, list_dims, weight, weight_cell, max_loss, early_stop, init_ae):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  get dataloader 
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    # valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    # list_dims = [first_layer_dim, int(first_layer_dim / 2)]
    # model = AE(input_shape=input_length, code_dim = code_dim, first_layer_dim = first_layer_dim, list_dim = list_dims).to(device)
    if init_ae == None:
        model = AE(input_shape=input_length, code_dim = code_dim, list_dim = list_dims).to(device)
    else:
        model = init_ae

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=1e-08)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,  weight_decay=1e-07)
    loss_list = []
    epoch_loss = []
    # mean-squared error loss
#     criterion = nn.MSELoss()
#     criterion = weighted_MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
    min_train_loss = np.inf
    patience = 0
    print('max_loss is' + str(max_loss))
#     res_loss = None
    for epoch in range(epochs):
        loss = 0
        for x, index in train_data:
            x = x.to(device)
            # print(index)
            code, outputs = model(x.float())
#             train_loss = criterion(outputs, x.float())
            
#             train_loss = weighted_MSELoss_ignore0(outputs.to(device), x.float(), weight, weight_cell[index]) 
            train_loss = weighted_MSELoss(outputs.to(device), x.float(), weight, weight_cell[index])
            
            
            train_loss.backward()
            optimizer.step()
        
            loss += train_loss.item()
        # print(len(train_data))
        # loss = loss 
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss)
        epoch_loss.append(loss)
        # loss = train_loss.data.item()
#         loss_list.append(train_loss.item())
        weight_cell_valid = np.array([1/len(valid_dataset)] * len(valid_dataset))
        cur_valid_err = test_err_weighted(model, valid_data, weight, weight_cell_valid)
        print("current validation error is")
        print(cur_valid_err)   
#         if cur_valid_err >= valid_error:
#             # print('here')
#             if max_loss < np.inf:
#                 if train_loss.item() <= max_loss:
#                     rea_ae = model
#                     loss_list.append(train_loss.item())
#                     return res_ae, loss_list
#             else:
#                 break
                 
#         else:
#             valid_error = cur_valid_err
# #             print(test_err(model, valid_data))
# #             if res_ae != None:
# #                 print(test_err(res_ae, valid_data))
#             res_ae = model
#             loss_list.append(train_loss.item())
# #             res_loss = loss_list
        if early_stop:
          if max_loss < np.inf:
              # record the best model so far. min_train_loss > max_loss
              if loss <= min_train_loss:
                  print(min_train_loss)
                  min_train_loss = loss
                  res_ae = model
                  loss_list.append(loss)
              if loss < max_loss and patience > 20:
                  # print(res_ae == None)
                  return res_ae, loss_list, epoch_loss
              else:
                patience += 1
          else:
            # res_ae = model
            # loss_list.append(loss)
            if cur_valid_err >= valid_error:
              if patience > 20:
                return res_ae, loss_list, epoch_loss
              else:
                patience += 1 
            else:
                valid_error = cur_valid_err
  #             print(test_err(model, valid_data))
  #             if res_ae != None:
  #                 print(test_err(res_ae, valid_data))
                res_ae = model
                loss_list.append(loss)
        else:
          res_ae = model
          loss_list.append(loss)
#         # print('here')
    return res_ae, loss_list, epoch_loss

# =========================================================================================================================================


# compute test error of a given autoencoder 
def test_err(autoencoder, data_test):
    criterion = nn.MSELoss()
    res = []
#     counter = 0
    for x, index in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        test_loss = criterion(outputs, x.float())
#         print(type(test_loss))
        res.append(test_loss.item())
#         counter += batch_size
    return np.mean(res)

def test_err_weighted(autoencoder, data_test, weight, weight_cell):
    # dataset = torch.utils.data.DataLoader(data_test, batch_size, shuffle=False)
    # criterion = nn.MSELoss()
    res = []
#     counter = 0
    for x, index in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        # print(outputs.size())
#         print(index)
#         test_loss = weighted_MSELoss_ignore0(outputs, x.float(), weight, weight_cell[index])
        test_loss = weighted_MSELoss(outputs, x.float(), weight, weight_cell[index])
#         print(type(test_loss))
        res.append(test_loss.item())
        # print(res)
#         counter += batch_size
    # print('len of test res' + str(len(res)))
    # print(res)
    return np.sum(res)

# def check_encod(autoencoder, data_train):
#     for x, index in 
# Save
def save_autoencoder(model):
    torch.save(model, 'autoencoder.pth')

