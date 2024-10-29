import numpy as np
import pandas as pd

from common import pytami_func as ami
from common import graph_info as gr
from pydlr import dlr

def transpose_lits(l):
    return list(map(list, zip(*l)))


def first_order_truncation(G, SE, U):
    return G + G ** 2 * SE * U ** 2


def dyson(G, SE):
    """
    Calculation bold GF in Dyson equation
    :param G: GF function usually bare
    :param SE: Self energy
    :return: bold GF
    """
    res = G / (1.0 + 0.0*1j - G*SE)
    return res

def get_k_ext_cut(N):
    """
    Function that find cut for kx ky
    :param N: Number of point in one way
    :return: cut for kx ky
    """
    kexx, kexy = np.zeros(N - 1), np.linspace(0, np.pi, N)[:-1]
    kexx, kexy = np.append(kexx, np.linspace(0, np.pi, N)[:-1]), np.append(kexy, np.ones(N - 1) * np.pi)
    kexx, kexy = np.append(kexx, np.linspace(np.pi, 0, N)[:-1]), np.append(kexy, np.linspace(np.pi, 0, N)[:-1])
    return kexx, kexy

def get_energy_dispersion_AMI(kx, ky, t=1):
    """
    Energy dispersion for 2d Hubbard model
    :param kx: momentum in x direction, can be either float or numpy array
    :param ky: momentum in y direction, can be either float or numpy array
    :param t: hopping parameter, generally = 1
    :return: negative energy dispersion [-e(kx, ky)]
    """
    return 2 * t * (np.cos(kx) + np.cos(ky))

def get_GF_from_DLR_iw(iw, g_k, w_k):
    """
    Function that recover Green function from pole representation
    :param iw: matsubara frequencies of Green Function
    :param g_k: pole weights, numpy array
    :param w_k: pole locations, numpy array
    :return: Green function in matsubara frequency, works same if using iw as real frequencies with small imaginary part
    """
    res = 0
    for pole_weight, pole_location in zip(g_k, w_k):
        res = res + pole_weight/(iw - pole_location)
    return res

def get_G0(iw, kx, ky, t=1):
    """
    Finds non-interacting Green function for 2d Hubbard model
    :param iw: matsubara frequencies
    :param kx: momentum in x direction
    :param ky: momentum in y direction
    :param t: hopping parameter, generally = 1
    :return: Non-interacting GF
    """
    return 1/(np.add.outer(get_energy_dispersion_AMI(kx, ky, t), iw))

def matsubara(statistics: str, n, beta):
    """
    Finds imaginary parts of Matsubara frequencies
    :param statistics: Fermi or Bose statistics
    :param n: number of matsubara frequency, can be int or numpy array of numbers
    :param beta: inverse temperature
    :return: imaginary part of Matsubara frequencies
    """
    omega = 0
    if statistics == "F":
        omega = (2*n + 1) * np.pi / beta
    if statistics == "B":
        omega = 2*n * np.pi / beta
    return omega

def get_fermi_bose(statistics: str, dispersion, beta):
    """
    Finds Fermi or Bose distribution
    :param statistics: which statistics 'F' or 'B'
    :param dispersion: dispersion relationship (energy)
    :param beta: inverse temperature
    :return: Fermi or Bose distribution
    """
    if statistics == "F":
        return 1 / (np.exp(dispersion * beta) + 1)
    elif statistics == "B":
        return 1 / (np.exp(dispersion * beta) - 1)
    else:
        return

def get_self_energy_o2(iomega, z1, z2, z3, beta):
    """
    Finds second order self energy
    :param iomega: matsubara frequencies of SE
    :param z1: energy dispersion of first GF
    :param z2: energy dispersion of second GF
    :param z3: energy dispersion of third GF
    :param beta: inverse temperature
    :return: Second order SE for frequencies iomega
    """
    num = ((get_fermi_bose("F", z1, beta) - get_fermi_bose("F", z2, beta))
           * (get_fermi_bose("B", z2 - z1, beta) + get_fermi_bose("F", -z3,beta)))
    den = iomega + z1 - z2 - z3
    return num / den

class SE_AMI:
    """
    Class that find SE for certain type of diagram that represented by ord group and number and has L*L
    2d grid for every GF
    """
    def __init__(self, beta, L, order, group, num, path_to_graph, device):
        """
        :param beta: inverse temperature
        :param L: kx*ky grid size in one direction, whole grid L**2
        :param ord: order of SE diagram
        :param group: group of SE diagram
        :param num: num of SE diagram
        :param path_to_graph: path to the diagram
        :param device: torch device cuda or cpu
        """
        self.beta = beta
        self.L = L
        self.order = order
        self.group = group
        self.num = num
        self.path_to_graph = path_to_graph
        self.device = device
        self.device = device
        self.N_GF = 2 * self.order - 1
        self.R0, self.prefactor = gr.get_R0_prefactor_specific(
            self.path_to_graph, ord=self.order, group=self.group, num=self.num)
        self.kx_mesh, self.ky_mesh = self._get_ind_kx_ky_mesh(self.order, self.L)

    def _get_ind_kx_ky_mesh(self, order, L):
        """
        Function to find kx * ky grid for n = ord number of internal momenta
        :param order: order of SE
        :param L: kx*ky grid size in one direction, whole grid L**2
        :return: kx and ky mesh both have shape (L**(2*ord), ord)
        """
        kx_ind = np.zeros((order, L ** 2))
        ky_ind = np.zeros((order, L ** 2))
        for i in range(order):
            cur_x = np.linspace(0, 2 * np.pi, L)
            cur_y = np.linspace(0, 2 * np.pi, L)
            cur_x, cur_y = np.meshgrid(cur_x, cur_y)
            cur_x, cur_y = np.ravel(cur_x), np.ravel(cur_y)
            kx_ind[i, :] = cur_x
            ky_ind[i, :] = cur_y

        kx_mesh = np.meshgrid(*kx_ind)
        kx_mesh = list(kx_mesh)
        for i in range(len(kx_mesh)):
            kx_mesh[i] = np.ravel(kx_mesh[i])

        ky_mesh = np.meshgrid(*ky_ind)
        ky_mesh = list(ky_mesh)
        for i in range(len(ky_mesh)):
            ky_mesh[i] = np.ravel(ky_mesh[i])
        return kx_mesh, ky_mesh

    def get_k_mesh(self, kexx: np.array, kexy: np.array):
        """
        Function that takes external kx, ky grid and calculates every GF kx ky mesh depending on topology of diagram
        :param kexx: external kx should be same length as kexy
        :param kexy: external ky should be same length as kexx
        :return: k mesh
        """
        kx_mesh, ky_mesh = self.kx_mesh.copy(), self.ky_mesh.copy()

        k_mesh = np.zeros((len(kexx), 2, self.N_GF, self.L**(2 * self.order)))
        for ind, (kexx_, kexy_) in enumerate(zip(kexx, kexy)):
            for idx, v in enumerate(self.R0):
                currx = 0
                curry = 0
                for a, kx, ky in zip(v.alpha_, kx_mesh + [kexx_], ky_mesh + [kexy_]):
                    if a == 0:
                        continue
                    currx += a * kx
                    curry += a * ky
                i = list(v.eps_).index(1)
                k_mesh[ind, 0, i, :] = currx % (2*np.pi)
                k_mesh[ind, 1, i, :] = curry % (2*np.pi)
        return k_mesh.transpose((0, 1, 3, 2))

    def get_SE_AMI_from_dispersion(self, iw, k_mesh):
        """
        Function that calculates energy dispersions for k_mesh and uses them to find SE sigma
        :param iw: matsubara frequencies
        :param k_mesh: k_mesh for every k dependence for certain topology of diagram
        :return: SE sigma for matsubara frequencies and external momenta size of length of kext * length of matsubara
        """
        energy_list = np.zeros((self.N_GF, self.L**(2 * self.order)))
        sigma = np.zeros((k_mesh.shape[0], iw.shape[0]), dtype=complex)
        for i_ext in range(k_mesh.shape[0]):
            for i in range(self.N_GF):
                energy_list[i, :] = get_energy_dispersion_AMI(k_mesh[i_ext, 0, :, i], k_mesh[i_ext, 1, :, i])
            sigma[i_ext, :] = ami.get_sigma_torchami_from_dispersion(
                iw, self.beta, energy_list, self.R0, self.prefactor,
                False, 0, self.device) / (self.L ** (2 * self.order))
        return sigma

    def get_SE_AMI_from_poles(self, iw: np.array, k_mesh: np.array, poles_weights_kx_ky: pd.DataFrame, poles_locs: list):
        """
        Function that uses k_mesh and poles weight and locations to calculate SE sigma
        :param iw: matsubara frequencies
        :param k_mesh: k mesh of momenta for every topology of diagram
        :param poles_weights_kx_ky: table for every choice of kx, ky and number of DLR pole representation,
        that equals to number of GF in the diagram
        :param poles_locs: list for every choice of number of DLR pole representation,
        equals to number of GF in the diagram
        :return: SE sigma for matsubara frequencies and external momenta size of length of kext * length of matsubara
        """
        sigma = np.zeros((k_mesh.shape[0], iw.shape[0]), dtype=complex)
        for i_ext in range(k_mesh.shape[0]):
            sigma[i_ext, :] = ami.get_sigma_torchami_from_poles(
                iw, self.beta, k_mesh[i_ext], poles_weights_kx_ky, poles_locs, self.R0, self.prefactor,
                False, 0, self.device) / (self.L ** (2 * self.order))
        return sigma

class multiple_DLR:
    """
    Class that creates multiple different DLR representations
    """
    def __init__(self, n_dlr, beta, E_max, eps, delta_range: tuple):
        """
        :param n_dlr: number of DLR representations needed
        :param beta: beta temperature
        :param E_max: range for DLR
        :param eps: DLR eps
        :param delta_range: tuple of boarders for additional delta to the E_max
        """
        self.n_dlr = n_dlr
        self.beta = beta
        self.E_max = E_max
        self.eps = eps
        self.delta_range = delta_range
        self.d_list = []
        for n_dlr_ in range(self.n_dlr):
            delta = np.random.uniform(self.delta_range[0], self.delta_range[1])
            self.d_list.append(dlr(lamb=self.beta * (self.E_max + delta), eps=self.eps))
        self.w_list = []
        self.iw_q_list = []
        self.iw_q_re_list = []
        self.r_list = []
        self.r_list_un = []
        for d in self.d_list:
            self.w_list.append(-d.dlrrf / beta)  # negative because of how AMI made
            self.r_list.append(len(d.dlrrf))
            self.iw_q_list.append(d.get_matsubara_frequencies(beta))
            self.iw_q_re_list.append(np.imag(d.get_matsubara_frequencies(beta)))

    def get_poles_weights_list(self, G_iwq_list):
        '''
        Finds weights of the DLR poles for multiple DLR representations
        :param G_iwq_list: List of GF for multiple DLR representations
        :return: list of poles weights for multiple DLR representations
        '''
        poles_weights_list = []
        for G_iwq, d in zip(G_iwq_list, self.d_list):
            poles_weights_list.append(d.dlr_from_matsubara(G_iwq, self.beta)*(-1))
        return poles_weights_list
