import sudo_algo2
import dataset_gen
import numpy as np
import os.path

def print_weight(w):
	print("weight is " + str(w))

# run multiple experiments and plot average result
def get_average(num_round, round_size, train_data, data_dim, max_epoch, code_size, first_layer_dim, step_size, round_num):
    res_avg = []
    res = []
    test_avg = []
    test = []
    
#     add dynamic plot as data generates
    plt.ion()
    fig=plt.figure()
    plt.axis([0,data_dim,0,0.2])
    x = [i for i in range(data_dim)]
#  plt.figure()
    
#     start testing
    for j in range(num_round):
        temp = []
        temp_test = []
        for i in range(round_size):
            print('round' + str(i))
            ae, w, avg_w, loss, test_err, test_w = sudo_algo2.multi_weight(train_data, data_dim, max_epoch, code_size, first_layer_dim, step_size, round_num)
    #         res_avg.append(avg_w7)
            temp.append(w)
#             print(test_w)
            temp_test.append(test_w)
        res_avg.append(sum(temp)* (1/round_size))
        res.append(temp)
        cur_test_avg = sum(temp_test) * (1/round_size)
        test_avg.append(cur_test_avg)
        test.append(temp_test)
        temp= []
        temp_test = []

        pl.plot(x, cur_test_avg, 'o--') 
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)
    return res_avg, res, test_avg, test


if __name__ == '__main__':

	l_name = ['sphere_1000_1000_10', 'torus_1000_1000_2_1', 'uniformnoise_1000_13_1000', 'sphere_1000_1000_4', 'torus_1000_1000_1_0.5', 'uniformnoise_1000_7_1000']
	for l in l_name:
		if not os.path.isfile(l + '.csv'):
			print('missing data file' + l + 'csv')

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
	ae7, w7, w7_avg,loss7, test7, test7_w = sudo_algo2.multi_weight(mix_raw_stack_43, 7, 10, 4, 128, 0.05, 50)
	print_weight(w7)
	# ae_d, w_d, wd_avg,loss_d, test_d = sudo_algo2.multi_weight(mix_raw_stack_disk, 9, 10, 5, 128, 0.05)
	# print_weight(w_d)
