from common.functions import *
import numpy as np
from dotenv import load_dotenv
import os
import torch
# Load environment variables from .env file
load_dotenv()
import sys
import pandas as pd


L = int(sys.argv[1])
# L = 5
ord, group, num = 0, 0, 0
beta = 5

# calculate for 3 kx ky points
kexx, kexy = [np.pi, np.pi/2], [np.pi, np.pi/2]

# calculate SE
Pi = CHI_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc_chi'), torch.device('cuda'))
iw = matsubara("B", np.arange(1, 2), beta=beta)
k_mesh = Pi.get_k_mesh(kexx, kexy)
Pi_res = Pi.get_SE_AMI_from_dispersion(iw, k_mesh)

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
Pi_res = np.ravel(Pi_res)

try:
    df = pd.read_csv(f"Pi_ord{ord}_L_dep_1st_freq.csv")
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty DataFrame
    df = pd.DataFrame(columns=['kx', 'ky', 'iw', 'L', 'SE_re', 'SE_im'])

for i in range(len(kx)):
    df.loc[len(df)] = [kx[i], ky[i], iw[i], L, np.real(Pi_res[i]), np.imag(Pi_res[i])]
print(df)
df.to_csv(f"Pi_ord{ord}_L_dep_1st_freq.csv", index=False)

