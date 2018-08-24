import psutil
import os

import tensorly as tl
import numpy as np
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker

import argparse


### Receive Args ###
parser = argparse.ArgumentParser()
parser.add_argument('--time', type=int, default=1)
parser.add_argument('OPERATION', type=str)
parser.add_argument('--new_tensor', type=bool, default=False)
parser.add_argument('--origin', nargs=3, type=int, default=[10, 10, 64])
parser.add_argument('--rank', type=int, default=4)
parser.add_argument('--core', nargs=3, type=int, default=[10, 10, 64])
args = parser.parse_args()


### Get Memory Usage ###
pid = os.getpid()
p = psutil.Process(pid)
def Get_Mem():
	info = p.memory_full_info()	
	memory = info.uss / 1024. / 1024.
	print ('Memory used: {:.2f} MB'.format(memory))


### SETTING ###
## CP ##
# Rank of the CP decomposition
cp_rank = args.rank


## TUCKER ##
random_state = 12345
# Rank of the Tucker decomposition
tucker_rank = args.core

## ORIGIN_TENSOR ##
DIM1 = args.origin[0]
DIM2 = args.origin[1]
DIM3 = args.origin[2]
size = [DIM1,DIM2,DIM3]
#current_path = os.getcwd()
save_path = os.path.join('..','save', '{}'.format(size))
if args.new_tensor:
	origin_tensor = np.random.randn(DIM1, DIM2, DIM3)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	np.save(os.path.join(save_path, 'cp_origin_tensor.npy'), origin_tensor)
	origin_tensor = np.random.randn(DIM1, DIM2, DIM3)
	np.save(os.path.join(save_path, 'tucker_origin_tensor.npy'), origin_tensor)
else:
	cp_origin_tensor = np.load(os.path.join(save_path, 'cp_origin_tensor.npy'))
	tucker_origin_tensor = np.load(os.path.join(save_path, 'tucker_origin_tensor.npy'))

#origin_tensor = np.random.randn(7, 8, 5)

#origin_tensor = np.load('origin_tensor.npy')
#np.save('origin_tensor.npy', origin_tensor)
#print(origin_tensor[:, :, 0])




if (args.OPERATION == 'CP') or (args.OPERATION == 'cp'):
	# Perform the CP decomposition
	factors = parafac(cp_origin_tensor, rank=cp_rank, init='random', tol=10e-6)
	# Reconstruct the image from the factors
	cp_reconstruction = tl.kruskal_to_tensor(factors)
	name = 'py_CP_rec_' + str(args.time) + '.npy'
	np.save(os.path.join(save_path, name), cp_reconstruction)

if (args.OPERATION == 'TUCKER') or (args.OPERATION == 'tucker'):
	# Tucker decomposition
	core, tucker_factors = tucker(tucker_origin_tensor, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
	tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)
	#new_tucker = tg.multi_mode_dot(core, [tucker_factors[0], tucker_factors[1], tucker_factors[2]])
	name = 'py_TUCKER_rec_' + str(args.time) + '.npy'
	np.save(os.path.join(save_path, name), tucker_reconstruction)


Get_Mem()

