from common.functions import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
import os
import time
import pandas as pd
load_dotenv()
import sys

# parameters
L = int(sys.argv[1])
# L = 5  # number of kx points and ky points (whole grid L**2) better be odd
ord = 2  # order of the graph
beta = 5  # inverse temperature
E_max = 5.  # range for DLR
n = 2*ord - 1  # number of GF in the graph
eps = 1e-6  # eps parameter for DLR
n_iw = 5  # number of matsubara freq to calculate
U = 1

# DLR representation
d = multiple_DLR(n_dlr=n, beta=beta, E_max=E_max, eps=eps, delta_range=(0, 1))
iw_q_list = d.iw_q_list
print(f"Number of poles: {[len(iw_q_list[i]) for i in range(n)]}")

# external k grid
kx, ky = np.linspace(0, 2 * np.pi, L), np.linspace(0, 2 * np.pi, L)
kx, ky = np.meshgrid(kx, ky)
kx, ky = np.ravel(kx), np.ravel(ky)

# SE energy class and k mesh for kx and ky points
SE = SE_AMI(beta, L, ord, 0, 0, os.getenv('GRAPH_PATH_calc'), torch.device('cuda'))
k_mesh = SE.get_k_mesh(kx, ky)

# Find G0 in iw_DLR points and SE from energy dispersion
G_list = []
for i, iw_q in enumerate(iw_q_list):
    G0 = get_G0(iw_q, kx, ky)
    SE_disp = SE.get_SE_AMI_from_dispersion(iw_q, k_mesh)
    G_list.append(G0 + G0**2 * SE_disp * U**2)

    # G_list.append(dyson(G0, SE_disp*U**2))
G_list = list(map(list, zip(*G_list)))
# get the DLR poles weights for every kx, ky point
poles_w = [d.get_poles_weights_list(G_list[i]) for i in range(len(kx))]
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


iw = matsubara("F", np.arange(0, 5), beta=beta)
N = 11  # should be odd and equal to (L // 2 + 1) to much with the k grid that exist
# kexx, kexy = get_k_ext_cut(N)
kexx, kexy = [0, np.pi], [np.pi, 0]
k_mesh = SE.get_k_mesh(kexx, kexy)
start_time = time.time()
print(len(d.w_list[0]))
SE_res = SE.get_SE_AMI_from_poles(iw, k_mesh, poles_weights_kx_ky, poles_locs=d.w_list)
end_time = time.time()
total_time = end_time - start_time
print("Execution time: {} seconds".format(total_time))

# if needed to save time
with open('time_log.txt', 'a') as f:
    np.savetxt(f, np.column_stack((L, total_time)))

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
SE_res = np.ravel(SE_res)
df = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'SE_res_re': np.real(SE_res), 'SE_res_im': np.imag(SE_res)})
print(df)
# df.to_csv(f'SE_res_ord{ord}_L{L}_DLR_1.csv', index=False)
# df.to_csv(f'SE_res_ord{ord}_L{L}_DLR_1_dyson.csv', index=False)
