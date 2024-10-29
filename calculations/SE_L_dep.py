from common.functions import *
import numpy as np
from dotenv import load_dotenv
import os
import torch
# Load environment variables from .env file
load_dotenv()
import sys
import pandas as pd


# L = int(sys.argv[1])
L = 5
ord, group, num = 2, 0, 0
beta = 5

# calculate for 3 kx ky points
kexx, kexy = [0, np.pi, np.pi/2], [0, np.pi, np.pi/2]

# calculate SE
SE = SE_AMI(beta, L, ord, group, num, os.getenv('GRAPH_PATH_calc'), torch.device('cuda'))
iw = matsubara("F", np.arange(0, 1), beta=beta)
k_mesh = SE.get_k_mesh(kexx, kexy)
SE_res = SE.get_SE_AMI_from_dispersion(iw, k_mesh)

# save into df
kx = np.repeat(kexx, len(iw))
ky = np.repeat(kexy, len(iw))
iw = np.tile(iw, len(kexx))
SE_res = np.ravel(SE_res)

try:
    df = pd.read_csv(f"SE_ord{ord}_L_dep.csv")
except FileNotFoundError:
    # If the file doesn't exist, initialize an empty DataFrame
    df = pd.DataFrame(columns=['kx', 'ky', 'iw', 'L', 'SE_re', 'SE_im'])

for i in range(len(kx)):
    df.loc[len(df)] = [kx[i], ky[i], iw[i], L, np.real(SE_res[i]), np.imag(SE_res[i])]
print(df)
df.to_csv(f"SE_ord{ord}_L_dep.csv", index=False)

