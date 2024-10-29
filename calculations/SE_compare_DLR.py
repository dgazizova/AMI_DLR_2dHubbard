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
L = 11  # number of kx points and ky points (whole grid L**2)
print(f"Size of the grid L={L}")
ord = 2  # order of the graph
beta = 5  # inverse temperature
E_max = 5.  # range for DLR
n = 2*ord - 1  # number of GF in the graph
eps = 1e-6  # eps parameter for DLR
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
    'polls': poles_w
})

# Find SE using energy dispersion and poles
SE = SE_AMI(beta, L, ord, 0, 0, os.getenv('GRAPH_PATH_calc'), torch.device('cuda'))
iw = matsubara("F", np.arange(0, 5), beta=beta)
kexx, kexy = [0], [0]
k_mesh = SE.get_k_mesh(kexx, kexy)
res_AMI = SE.get_SE_AMI_from_dispersion(iw, k_mesh)
print(res_AMI)
start_time = time.time()
res = SE.get_SE_AMI_from_poles(iw, k_mesh, poles_weights_kx_ky, poles_locs=d.w_list)
print(res)
end_time = time.time()
print("Execution time: {} seconds".format(end_time - start_time))

# compare two results
re_diff = np.abs(np.real(res[0]) - np.real(res_AMI[0]))
im_diff = np.abs(np.imag(res[0]) - np.imag(res_AMI[0]))
print(re_diff, im_diff)

# save differences
# with open('diff.txt', 'a') as f:
#     np.savetxt(f, np.column_stack((np.full(5, L), np.full(5, len(iw_q_list[0])), re_diff, im_diff)))


# plot real and imag diff
plt.subplot(1, 2, 1)
plt.plot(iw, np.real(res_AMI[0]), marker=".", label='AMI')
plt.plot(iw, np.real(res[0]), marker=".", label="AMI_DLR")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(iw, np.imag(res_AMI[0]), marker=".", label='AMI')
plt.plot(iw, np.imag(res[0]), marker=".", label="AMI_DLR")
plt.legend()
plt.show()


