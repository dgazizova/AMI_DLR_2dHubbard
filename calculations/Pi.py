from common.functions import *
import numpy as np
from dotenv import load_dotenv
import os
import torch
# Load environment variables from .env file
load_dotenv()


L = 1001
ord, group, num = 0, 0, 0
beta = 5

# create external k mesh
N = 21
kexx, kexy = get_k_ext_cut(N)

# calculate Polarization bubble
CHI = CHI_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc_chi'), torch.device('cuda'))
iw = matsubara("B", np.arange(0, 5), beta=beta)
k_mesh = CHI.get_k_mesh(kexx, kexy)
CHI_res = CHI.get_SE_AMI_from_dispersion(iw, k_mesh)

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
CHI_res = np.ravel(CHI_res)
df = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'CHI_res_re': np.real(CHI_res), 'CHI_res_im': np.imag(CHI_res)})
print(df)
df.to_csv(f'CHI_res_ord{ord}_L{L}.csv', index=False)
