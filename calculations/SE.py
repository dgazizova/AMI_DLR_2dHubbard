from common.functions import *
import numpy as np
from dotenv import load_dotenv
import os
import torch
# Load environment variables from .env file
load_dotenv()

# L = int(sys.argv[1])
L = 21
ord, group, num = 2, 0, 0
beta = 5

# create external k mesh
N = 21
kexx, kexy = get_k_ext_cut(N)

# calculate SE
SE = SE_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc'), torch.device('cuda'))
iw = matsubara("F", np.arange(0, 10), beta=beta)
k_mesh = SE.get_k_mesh(kexx, kexy)
SE_res = SE.get_SE_AMI_from_dispersion(iw, k_mesh)

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
SE_res = np.ravel(SE_res)
df = pd.DataFrame({'kx': kx, 'ky': ky, 'iw': iw, 'SE_res_re': np.real(SE_res), 'SE_res_im': np.imag(SE_res)})
print(df)
df.to_csv(f'SE_res_ord{ord}_L{L}.csv', index=False)
