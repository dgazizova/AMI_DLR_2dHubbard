from common.functions import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
import os
load_dotenv()
import time
import sys

# parameters
# L = int(sys.argv[1])
L = 201  # number of kx points and ky points (whole grid L**2)
print(f"Size of the grid L={L}")
ord = 0  # order of the graph
beta = 5  # inverse temperature
E_max = 5.  # range for DLR
n = 2*ord + 2  # number of GF in the graph
eps = 1e-9  # eps parameter for DLR
n_iw = 5  # number of matsubara freq to calculate

# DLR representation
d = multiple_DLR(n_dlr=n, beta=beta, E_max=E_max, eps=eps, delta_range=(0, 1))
iw_q_list = d.iw_q_list
print(f"Number of poles: {d.r_list}")

# external k grid
kx, ky = np.linspace(0, 2 * np.pi, L), np.linspace(0, 2 * np.pi, L)
kx, ky = np.meshgrid(kx, ky)
kx, ky = np.ravel(kx), np.ravel(ky)


# Find G0 in iw_DLR points
G0_list = []
for i, iw_q in enumerate(iw_q_list):
    G0_list.append(get_G0(iw_q, kx, ky))
G0_list = transpose_lits(G0_list)
# get the DLR poles weights for every kx, ky point
poles_w = [d.get_poles_weights_list(G0_list[i]) for i in range(len(kx))]
poles_w = [item for sublist in poles_w for item in sublist]

# fill Data Frame for poles
kx_df = np.repeat(kx, n)
ky_df = np.repeat(ky, n)
ind = np.tile(np.arange(n), len(kx))
poles_weights_kx_ky = pd.DataFrame({
    'kx': kx_df,
    'ky': ky_df,
    'indx': ind,
    'poles': poles_w
})

# Find SE using energy dispersion and poles
Pi = CHI_AMI(beta, L, ord, 0, 0, os.getenv('GRAPH_PATH_calc_chi'), torch.device('cuda'))
iw = matsubara("B", np.arange(0, 5), beta=beta)

N = 21
kexx, kexy = get_k_ext_cut(N)

k_mesh = Pi.get_k_mesh(kexx, kexy)
start_time = time.time()
res = Pi.get_SE_AMI_from_poles(iw, k_mesh, poles_weights_kx_ky, poles_locs=d.w_list)
# print(res)
end_time = time.time()
print("Execution time: {} seconds".format(end_time - start_time))


# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
CHI_res = np.ravel(res)
df = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'CHI_res_re': np.real(CHI_res), 'CHI_res_im': np.imag(CHI_res)})
print(df)
df.to_csv(f'CHI_res_ord{ord}_L{L}_DLR_cut.csv', index=False)
