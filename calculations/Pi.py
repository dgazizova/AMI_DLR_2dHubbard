from common.functions import *
from common import vk as vk
import numpy as np
from dotenv import load_dotenv
import os
import torch
# Load environment variables from .env file
load_dotenv()


L = 13
ord, group, num = 2, 0, 0
beta = 5

# create external k mesh
N = 21
# kexx, kexy = get_k_ext_cut(N)
kexx, kexy = [np.pi], [np.pi]

# calculate Polarization bubble
CHI = CHI_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc_chi'), torch.device('cuda'))
iw = matsubara("B", np.arange(0, 5), beta=beta)
# print([i.shape for i in CHI.kx_mesh])
k_mesh = CHI.get_k_mesh(kexx, kexy)
print(k_mesh.shape)
# print(k_mesh.shape)
vk_inds = vk.get_index_vk(os.getenv('GRAPH_PATH_calc_chi'), ord, group, num)
vk_pref = CHI.get_vk(vk_inds, k_mesh)

CHI_res = CHI.get_SE_AMI_from_dispersion(iw, k_mesh)
print(CHI_res)

CHI_random = CHI_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc_chi'), torch.device('cuda'),
                     random_sample=True, N_integr=1*1e+6, N_batch=1)
CHI_res1 = CHI_random.batch_calc_random(iw, kexx, kexy)
print(CHI_res1)

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
CHI_res = np.ravel(CHI_res)
CHI_res_random = np.ravel(CHI_res1)
df = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'CHI_res_re': np.real(CHI_res), 'CHI_res_im': np.imag(CHI_res)})
print(df)

df1 = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'CHI_res_re': np.real(CHI_res_random), 'CHI_res_im': np.imag(CHI_res_random)})
print(df1)
# df.to_csv(f'CHI_res_ord{ord}_L{L}_new.csv', index=False)
# df1.to_csv(f'CHI_res_ord{ord}_random.csv', index=False)
