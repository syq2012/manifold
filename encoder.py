import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import dataset_gen


# Settings
# epochs = 10
batch_size = 128
# lr = 0.008
device = 'cpu'
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

# Model structure
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
#         self.encoder_hidden_layer = nn.Linear(
#             in_features=kwargs["input_shape"], out_features=128
#         )
#         self.encoder_output_layer = nn.Linear(
#             in_features=128, out_features=2
#         )
#         self.decoder_hidden_layer = nn.Linear(
#             in_features=2, out_features=128
#         )
#         self.decoder_output_layer = nn.Linear(
#             in_features=128, out_features=kwargs["input_shape"]
#         )
        data_dim = kwargs["input_shape"]
        code_dim = kwargs["code_dim"]
        first_layer_dim = kwargs["first_layer_dim"]
        mid_layer_dim = int(first_layer_dim / 2)
        print(mid_layer_dim)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, first_layer_dim),
#             nn.Tanh(),
#             nn.Linear(128, 64),
            nn.ReLU6(),
            nn.Linear(first_layer_dim, mid_layer_dim),
            nn.ReLU6(),
#             nn.Linear(64, 16),
#             nn.ReLU6(),
            
            nn.Linear(mid_layer_dim, code_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, mid_layer_dim),
            nn.ReLU6(),
#             nn.Linear(16, 64),
#             nn.ReLU6(),
#             nn.Linear(64, 128),
#             nn.Tanh(),
            nn.Linear(mid_layer_dim, first_layer_dim),
            nn.ReLU6(),
            nn.Linear(first_layer_dim, data_dim),
#             nn.Sigmoid()
        )

    def forward(self, inputs):
#         print(inputs.shape)
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

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
    model = AE(input_shape=input_length, code_dim = code_dim, first_layer_dim = first_layer_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss_list = []
    # mean-squared error loss
    criterion = nn.MSELoss()
#     criterion = weighted_MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
#     res_loss = None
    for epoch in range(epochs):
#         loss = 0
        for x in train_data:
            x = x.to(device)
            
            code, outputs = model(x.float())
            train_loss = criterion(outputs, x.float())
#             train_loss = weighted_MSELoss(outputs, x.float(), weight)
            
            train_loss.backward()
            optimizer.step()
        
#             loss += train_loss.item()
#         loss = loss / len(train_data)
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
        print('[{}/{}] Loss:'.format(epoch+1, epochs), train_loss.item())
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
    weight_tensor = torch.tensor(weight)
    # temp = (1/batch_size) * (output - target) ** 2
    # print(torch.sum(temp, dim = 0).size())
    (m, n) = output.size()
    # return torch.sum(torch.tensor(weight_cell)[:, None] * (weight_tensor *  (output - target) ** 2))
    return (1/m) * torch.sum(weight_tensor *  (output - target) ** 2)


def training_weighted_MSE(train_dataset, valid_dataset, epochs, input_length, code_dim, first_layer_dim, weight, weight_cell):
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #  get dataloader 
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size, shuffle=True)
    
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=input_length, code_dim = code_dim, first_layer_dim = first_layer_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_list = []
    # mean-squared error loss
#     criterion = nn.MSELoss()
#     criterion = weighted_MSELoss()
    
#     for early stopping
    valid_error = np.inf
    res_ae = None
#     res_loss = None
    for epoch in range(epochs):
#         loss = 0
        for x, index in train_data:
            x = x.to(device)
            # print(index)
            code, outputs = model(x.float())
#             train_loss = criterion(outputs, x.float())
            train_loss = weighted_MSELoss(outputs, x.float(), weight, weight_cell[index])
            
            train_loss.backward()
            optimizer.step()
        
#             loss += train_loss.item()
#         loss = loss / len(train_data)
#         print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss)) 
        print('[{}/{}] Loss:'.format(epoch+1, epochs), train_loss.item())
#         loss_list.append(train_loss.item())
        cur_valid_err = test_err_weighted(model, valid_data, weight, weight_cell)
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

# =========================================================================================================================================


# compute test error of a given autoencoder 
def test_err(autoencoder, data_test):
    criterion = nn.MSELoss()
    res = []
#     counter = 0
    for x in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        test_loss = criterion(outputs, x.float())
#         print(type(test_loss))
        res.append(test_loss.item())
#         counter += batch_size
    return np.mean(res)

def test_err_weighted(autoencoder, data_test, weight, weight_cell):
    # criterion = nn.MSELoss()
    res = []
#     counter = 0
    for x, index in data_test:
#         if counter <= r:
        x = x.to(device)
        code, outputs = autoencoder(x.float())
        test_loss = weighted_MSELoss(outputs, x.float(), weight, weight_cell[index])
#         print(type(test_loss))
        res.append(test_loss.item())
#         counter += batch_size
    return np.mean(res)

# Save
def save_autoencoder(model):
    torch.save(model, 'autoencoder.pth')
