import sudo_algo2
import dataset_gen
import numpy as np


def print_weight(w):
	print("weight is " + str(w))

if __name__ == '__main__':
	stack_data, rawns = dataset_gen.stack_dataset('sphere_1000_1000_10', 'torus_1000_1000_2_1', 'uniformnoise_1000_13_1000')
	stack_data_43, rawn7 = dataset_gen.stack_dataset('sphere_1000_1000_4', 'torus_1000_1000_1_0.5', 'uniformnoise_1000_7_1000')

	name = "disk_1000_1000"
	l = name.split('_')
	raw = dataset_gen.read(name + '.csv', 2, int(l[2]))
	stack_data_with_disk = [np.concatenate((t1, t2), axis = 0) for t1, t2 in zip(stack_data_43, raw)]

	rawn9 = dataset_gen.read('uniformnoise_1000_9_1000.csv', 9, 1000)
	print("loading finished")


	noise_scale = 0.01
	mix_raw_stack = np.add(stack_data, np.multiply(rawns, noise_scale))
	mix_raw_stack_43 = np.add(stack_data_43, np.multiply(rawn7, noise_scale))
	mix_raw_stack_disk = np.add(stack_data_with_disk, np.multiply(rawn9, noise_scale))



	# ae, w, w_avg, loss, test = sudo_algo2.multi_weight(mix_raw_stack, 13, 10, 10, 128, 0.05)
	# print_weight(w)
	# ae3, w3, w3_avg,loss3, test3 = sudo_algo2.multi_weight(mix_raw_stack, 13, 10, 3, 128, 0.05)
	# print_weight(w3)
	ae7, w7, w7_avg,loss7, test7 = sudo_algo2.multi_weight(mix_raw_stack_43, 7, 10, 4, 128, 0.05)
	print_weight(w7)
	# ae_d, w_d, wd_avg,loss_d, test_d = sudo_algo2.multi_weight(mix_raw_stack_disk, 9, 10, 5, 128, 0.05)
	# print_weight(w_d)
