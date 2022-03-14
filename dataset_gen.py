import torch
import numpy as np
from numpy import savetxt
from numpy import loadtxt
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from torch.utils.data import TensorDataset, DataLoader

# get data generating function
import sample
# this is to generate dataset for future testing

# e.g. write(lambda: fun(arg), n)
def write(gen_fun, num_sample, name):
	result = [gen_fun().flatten() for i in range(num_sample)]
	f_name = name + '.csv'
	savetxt(f_name, result, delimiter= ',')
	# tensor_result = torch.from_numpy(result)
	# cur_dataset = TensorDataset(tensor_result)

def read(file_name, d, n):
	data = loadtxt(file_name, delimiter = ',')
	result = [data[i].reshape((d, n)) for i in range(len(data))]
	return result

# generate num_sample set of samples, each set contains num_cell number of data point of 
# dimension k
def gen_sphereical(num_sample, num_cell,  k):
	name = 'sphere' + '_' + str(num_sample) + '_'  + str(num_cell) + '_'  +str(k)
	write(lambda: sample.sample_spherical(num_cell, k), num_sample, name)


# generate torus of outer radius R and inner radius r
def gen_torus(num_sample, num_cell_toru, R, r):
	name = 'torus' + '_'  + str(num_cell_toru) + '_'  + str(num_sample) + '_'  +str(R) + '_'  +str(r)
	write(lambda: sample.sample_torus(num_cell_toru, R, r), num_sample, name)

# generate disk of random radius
def gen_disk( num_sample, num_cell):
	name = 'disk' + '_'  + str(num_sample) + '_'  + str(num_cell)
	write(lambda: sample.sample_disk(num_cell), num_sample, name)

def gen_noise(num_sample, m, n):
	name = 'uniformnoise' + '_' + str(num_sample) + '_' + str(m) + '_' + str(n)
	write(lambda: sample.sample_noise(m, n), num_sample, name)


def gen_sphere_with_noise(num_sample, num_cell, k, noise_dimension, noise_intensity):
	lst = ['spherewithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda: sample.sample_spherical(num_cell, k), noise_dimension, noise_intensity, 1), num_sample, name)

def gen_torus_with_noise(num_sample, num_cell, R, r, noise_dimension, noise_intensity):
	lst = ['toruswithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda: sample.sample_torus(num_cell, R, r), noise_dimension, noise_intensity, 1), num_sample, name)

def gen_disk_with_noise(num_sample, num_cell, noise_dimension, noise_intensity):
	lst = ['diskwithnoise', 'numsample', str(num_sample), 'num_cell', str(num_cell), 'noise_dim', str(noise_dimension), 'noise_intensity', str(noise_intensity)]
	name = '_'.join(lst)
	write(lambda: sample.pad_noise(lambda:sample.sample_disk(num_cell), noise_dimension, noise_intensity, 1), num_sample, name)

# generate data valuated on d2 random polynomial of degree k
# original data of dimension d 
# returns coefficients of the polynomials (each row represent the correspoinding poly)
def gen_polynomial_data(file_name, d, n, k, d2):
	name = file_name + '.csv'
	cur_data = read(name, d, n)
	lifted, values, coeff = sample.random_polymonial(cur_data, k, d2)
	x_file = file_name + '_' + 'degree' + '_' + str(k) + '.csv'
	savetxt(x_file , [cur.flatten() for cur in lifted], delimiter= ',')
	new_name = file_name + '_num_poly_' + str(d2) + '_degree_' + str(k) + '.csv'
	savetxt(new_name, [cur.flatten() for cur in values], delimiter = ',')
	return coeff


# =========================================================================================================================================

# functions for combine data from two files, i.e. stacking or padding 
def get_dim(l):
    res = 3
    if l[0] == 'sphere':
        res = int(l[3])
    return res
# assume input datasets have the same number of samples 
# concat and add noise to two datasets

# change name
def combine_dataset(f1, f2, noise):
    l1 = f1.split('_')
    l2 = f2.split('_')
    l3 = noise.split('_')
    
    dim1 = get_dim(l1)
    dim2 = get_dim(l2)
#     print(dim1)
#     print(dim2)
#     print(l3[2] != l1[1])
    if (l1[1] != l2[1]) or (l1[2] != l2[2]) or (int(l3[2]) != (dim1 + dim2)) or l3[1] != l1[1]:
        print("data number not matched")
        return
    else:        
        raw1 = dataset_gen.read(f1 + '.csv', dim1, int(l1[2]))  
        raw2 = dataset_gen.read(f2+ '.csv', dim2, int(l2[2]))
        rawn = dataset_gen.read(noise+ '.csv', int(l3[2]), int(l3[3]))
        
        temp1 = [np.pad(x, ((0, dim2), (0, 0)), 'constant') for x in raw1]
        temp2 = [np.pad(x, ((dim1, 0), (0, 0)), 'constant') for x in raw2]
        res = [np.concatenate((t1, t1), axis = 1) for t1, t2 in zip(temp1, temp2)]
        return res, rawn
def stack_dataset(f1, f2, noise):
    l1 = f1.split('_')
    l2 = f2.split('_')
    l3 = noise.split('_')
    
    dim1 = get_dim(l1)
    dim2 = get_dim(l2)
    
    if (l1[1] != l2[1]) or (l1[2] != l2[2]) or (int(l3[2]) != dim1 + dim2) or l3[1] != l1[1]:
        print("data number not matched")
        return
    else:        
        raw1 = read(f1 + '.csv', dim1, int(l1[2]))  
        raw2 = read(f2+ '.csv', dim2, int(l2[2]))
        rawn = read(noise+ '.csv', int(l3[2]), int(l3[3]))
        
#         temp1 = [np.pad(x, ((0, dim2), (0, 0)), 'constant') for x in raw1]
#         temp2 = [np.pad(x, ((dim1, 0), (0, 0)), 'constant') for x in raw2]
        res = [np.concatenate((t1, t2), axis = 0) for t1, t2 in zip(raw1, raw2)]
        return res, rawn

# =========================================================================================================================================
# construct dataset for pytorch 
# input: matrix with column being a cell 
class cellDataset(torch.utils.data.Dataset):
	# data is a array of matrices 
	def __init__(self, data):
		# parse data into tensor 
		d, n = data.shape
		self.data = [torch.from_numpy(data[:,i]) for i in range(n)]

	def __len__(self):
		return len(self.data)
	def __getitem__(self, index):
		return self.data[index]


def construct_cell_dataset(data):
	return cellDataset(data)


	
	
