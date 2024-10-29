import pandas as pd
import torch
import sys
import pathlib
from dotenv import load_dotenv
import os
import numpy as np
# Load environment variables from .env file
load_dotenv()
sys.path.append(str(pathlib.Path(__file__).parent.joinpath(os.getenv('PYTAMI_PATH')).resolve()))
import pytami


def torch_energy_prefactor_tensors_kx_ky(energy_list, usedevice):
    e_tensor = [torch.tensor(en, device=usedevice) for en in energy_list]
    e_tensor = torch.column_stack(e_tensor)
    return e_tensor

def torch_prefactor_tensors_kx_ky_poles(k_mesh: np.array, poles_weights: pd.DataFrame, usedevice):
    prefactors = []
    poles_weights_kx_ky = [torch.tensor(
        poles_weights[np.isclose(poles_weights['kx'], k_mesh[0, j]) &
                      np.isclose(poles_weights['ky'], k_mesh[1, j]) & (poles_weights['indx'] == j)]['poles'].values[0], device=usedevice)
        for j in range(k_mesh.shape[1])]
    prefactors.append(torch.prod(torch.cartesian_prod(*poles_weights_kx_ky), dim=1))
    prefactors_tensor = torch.cat(prefactors, dim=0)
    return prefactors_tensor

def torch_energy_tensors_kx_ky_poles(poles_locs, usedevice):
    poles_locs_t = [torch.tensor(l, device=usedevice) for l in poles_locs]
    e_tensor = torch.cartesian_prod(*poles_locs_t)
    return e_tensor

def get_sigma_torchami_from_dispersion(wn, beta, energy_list, R0, pref_in, rfswitch, gamma, device):
    ami = pytami.TamiBase(device)
    fbatchsize = len(wn)
    order = len(R0[0].alpha_) - 1

    int_cols = torch.zeros((fbatchsize, order), dtype=torch.cfloat, device=device)
    ext_col = torch.zeros(fbatchsize, dtype=torch.cfloat, device=device)
    if rfswitch:
        ext_col += gamma*1j
    else:
        ext_col += torch.from_numpy(wn).to(device)*1j
    frequency = torch.column_stack((int_cols, ext_col))

    ftout = pytami.TamiBase.ft_terms()

    # Integration/Evaluation parameters
    E_REG = 0  # numerical regulator for small energies.  If inf/nan results try E_REG=1e-8
    N_INT = int(order)  # number of matsubara sums to perform
    test_amiparms = pytami.TamiBase.ami_parms(N_INT, E_REG)
    ami.construct(N_INT, R0, ftout)

    energy = torch_energy_prefactor_tensors_kx_ky(energy_list, ami.getDevice())

    external = pytami.TamiBase.ami_vars(energy, frequency, beta)
    calc_res = ami.evaluate(test_amiparms, ftout, external) * pref_in
    total_res = calc_res.nansum(dim=1, keepdim=False)
    return total_res.detach().cpu().numpy()



def get_sigma_torchami_from_poles(wn, beta, k_mesh, poles_weights, poles_locs, R0, pref_in, rfswitch, gamma, device):
    ami = pytami.TamiBase(device)
    fbatchsize = len(wn)
    order = len(R0[0].alpha_) - 1

    int_cols = torch.zeros((fbatchsize, order), dtype=torch.cfloat, device=device)
    ext_col = torch.zeros(fbatchsize, dtype=torch.cfloat, device=device)
    if rfswitch:
        ext_col += gamma*1j
    else:
        ext_col += torch.from_numpy(wn).to(device)*1j
    frequency = torch.column_stack((int_cols, ext_col))

    # Integration/Evaluation parameters
    E_REG = 0  # numberical regulator for small energies.  If inf/nan results try E_REG=1e-8
    N_INT = int(order)  # number of matsubara sums to perform
    test_amiparms = pytami.TamiBase.ami_parms(N_INT, E_REG)  # SHOULD BE (0, 0) ?
    ftout = pytami.TamiBase.ft_terms()
    ami.construct(N_INT, R0, ftout)

    energy = torch_energy_tensors_kx_ky_poles(poles_locs, device)
    external = pytami.TamiBase.ami_vars(energy, frequency, beta)
    calc_res = ami.evaluate(test_amiparms, ftout, external) * pref_in

    total_res = 0
    for i in range(k_mesh.shape[1]):
        prefactor = torch_prefactor_tensors_kx_ky_poles(k_mesh[:, i, :], poles_weights, device)
        mult_res = calc_res * prefactor
        total_res = total_res + mult_res.nansum(dim=1, keepdim=False)
    return total_res.detach().cpu().numpy()