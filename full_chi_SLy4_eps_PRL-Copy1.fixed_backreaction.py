import sys, os, pickle, h5py
import numpy as np
from numba import njit,prange
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.special import sph_harm
from tqdm import tqdm

from seins.eigenmode import Eigenmode, Gyremode
from seins.background import Background
from seins.orbit import Evolver_D
from seins.constants import *

from time import time
from copy import copy, deepcopy

N_NS = 2

BKG_NAME = 'SLy4.2'

COWLING = True
L_TIDE = 2
if COWLING:
    MODE_DIR = os.path.join(os.environ.get('SEINS_DATA'), 'mode_profiles', 'cowling')

    k3_l0 = {}
    k3_l2 = np.load(f'/home/kgb0255/data/seins/coupling/cowling/{BKG_NAME}/k3.l024.npg5.npy', allow_pickle=True)[()]
    k3_l4 = {}
    
    J_l0 = {}#
    J_l2 = np.load(f'/home/kgb0255/data/seins/coupling/cowling/{BKG_NAME}/J2.l024.npg5.npy', allow_pickle=True)[()]
    J_l4 = {}#np.load('/home/kgb0255/data/seins/coupling/cowling/SLy4/J.l4.npg1.npy', allow_pickle=True)[()]

    k4chi1_dict = np.load(f'/home/kgb0255/data/seins/coupling/cowling/{BKG_NAME}/k4.l024.npg5.npy', allow_pickle=True)[()]
else:
    pass
    # MODE_DIR = os.path.join(os.environ.get('SEINS_DATA'), 'mode_profiles', 'non_cowling')
    # GYRE_DIR = os.path.join('/home/kgb0255/work/poly_ns/LE2W16/modes/non_cowling')

    # k3_l0 = {}
    # k3_l2 = np.load('/home/kgb0255/data/seins/coupling/non_cowling/SLy4/k3.l2.npg1.npy', allow_pickle=True)[()]
    # k3_l4 = {}
    
    # J_l0 = {}
    # J_l2 = np.load('/home/kgb0255/data/seins/coupling/non_cowling/SLy4/J.l2.npg1.npy', allow_pickle=True)[()]
    # J_l4 = np.load('/home/kgb0255/data/seins/coupling/non_cowling/SLy4/J.l4.npg1.npy', allow_pickle=True)[()]

    # k4chi1_dict = np.load('/home/kgb0255/data/seins/coupling/non_cowling/SLy4/k4.l2.npg1.npy', allow_pickle=True)[()]
k3_l2_f = np.load(f'/home/kgb0255/data/seins/coupling/non_cowling/{BKG_NAME}/k3.l2.npg0.npy', allow_pickle=True)[()]
J_l2_f = np.load(f'/home/kgb0255/data/seins/coupling/non_cowling/{BKG_NAME}/J2.l2.npg0.npy', allow_pickle=True)[()]

k3_dict = {**k3_l0, **k3_l2, **k3_l4}
for k, v in k3_l2_f.items():
    assert k in list(k3_dict.keys())
    k3_dict[k] = v
J_dict = {**J_l0, **J_l2, **J_l4}
for k, v in J_l2_f.items():
    assert k in list(J_dict.keys())
    J_dict[k] = v

bkg = Background(load=os.path.join(os.environ.get('SEINS_DATA'), 'background', f'{BKG_NAME}.npy'))

get_W = lambda l_sph, m_sph: np.real(4*np.pi/(2*l_sph+1)*sph_harm(m_sph, l_sph, 0, np.pi/2))
W22 = get_W(2,2)
W20 = get_W(2,0)
W40 = get_W(4,0)
W42 = get_W(4,2)
W44 = get_W(4,4)
W00 = get_W(0,0)
W_dict = {(0,0): W00,
          (2,2): W22, (2, -2): W22, (2,0): W20,
          (4,4): W44, (4,-4): W44, (4,2): W42, (4,-2): W42, (4,0): W40}

'''
************************************************************************************************************
Load in chi1 modes
************************************************************************************************************
'''
print('Loading modes')
NON_COWLING_DIR = os.path.join(os.environ.get('SEINS_DATA'), 'mode_profiles', 'non_cowling')
f_mode = np.load(os.path.join(NON_COWLING_DIR, f'f.0.2.{BKG_NAME}.npy'), allow_pickle=True)[()]
f_afreq = f_mode['mode_afreq']
f_Q = f_mode['Q']
chi_afreq_list = [f_mode['mode_afreq'], f_mode['mode_afreq'], f_mode['mode_afreq']]
# chi_afreq_list = []
Q_list = [f_mode['Q'], f_mode['Q'], f_mode['Q']]
# Q_list = []
m_chi_list = [0,2,-2]
# m_chi_list = []
chi_surf_disp = [f_mode['xir_arr'][-1], f_mode['xir_arr'][-1], f_mode['xir_arr'][-1]]
# chi_surf_disp = []
chi_hdr = [f'f.0.2.{m_chi}' for m_chi in m_chi_list]
# chi_hdr = []

'''
g-mode
'''
_g1_data = np.load(os.path.join(MODE_DIR, f'g.1.2.{BKG_NAME}.npy'), allow_pickle=True)[()]
_g1_afreq = _g1_data['mode_afreq']
_g1_Q = _g1_data['Q']

for i in range(3):
    chi_afreq_list.append(_g1_afreq)
    Q_list.append(_g1_Q)
    chi_surf_disp.append(_g1_data['xir_arr'][-1])
m_chi_list.append(0)
m_chi_list.append(2)
m_chi_list.append(-2)

chi_hdr.append('g.1.2.0')
chi_hdr.append('g.1.2.2')
chi_hdr.append('g.1.2.-2')

G120_IDX = chi_hdr.index('g.1.2.0')
G122_IDX = chi_hdr.index('g.1.2.2')
G12N2_IDX = chi_hdr.index('g.1.2.-2')

N_CHI_L2 = len(chi_surf_disp)
W_list = [W_dict[(2,m_chi)] for m_chi in m_chi_list]

# n_l0_max = n_max
# n_l4_max = n_max

# print('loading l=0 modes')
# f_l0_mode = np.load(os.path.join(MODE_DIR, f'f.0.0.{bkg_name}.npy'), allow_pickle=True)[()]
# chi_hdr.append('f.0.0.0')
# chi_afreq_list.append(f_l0_mode['mode_afreq'])
# Q_list.append(0)
# m_chi_list.append(0)
# chi_surf_disp.append(f_l0_mode['xir_arr'][-1])
# W_list.append(W_dict[(0,0)])


# for n_pg_l0 in tqdm(range(1, n_l0_max+1)):
#     p_n = np.load(os.path.join(MODE_DIR, f'p.{n_pg_l0}.0.{bkg_name}.npy'), allow_pickle=True)[()]
#     chi_hdr.append(f'p.{n_pg_l0}.0.0')
#     chi_afreq_list.append(p_n['mode_afreq'])
#     Q_list.append(0)
#     m_chi_list.append(0)
#     chi_surf_disp.append(p_n['xir_arr'][-1])
#     W_list.append(W_dict[(0,0)])

N_CHI_L0 = len(chi_surf_disp) - N_CHI_L2

# print('loaindg l=4 modes')
# _m_chi_4 = [0,2,-2,4,-4]
# f_l4_mode = np.load(os.path.join(MODE_DIR, f'f.0.4.{bkg_name}.npy'), allow_pickle=True)[()]
# for m_chi_4 in _m_chi_4:
    # chi_hdr.append(f'f.0.4.{m_chi_4}')
    # chi_afreq_list.append(f_l4_mode['mode_afreq'])
    # Q_list.append(0)
    # m_chi_list.append(m_chi_4)
    # chi_surf_disp.append(f_l4_mode['xir_arr'][-1])
    # W_list.append(W_dict[(4,m_chi_4)])


# for n_pg_l4 in tqdm(range(1, n_l4_max+1)):
#     p_n = np.load(os.path.join(MODE_DIR, f'p.{n_pg_l4}.4.{bkg_name}.npy'), allow_pickle=True)[()]
#     # g_n = np.load(os.path.join(MODE_DIR, f'g.{n_pg_l4}.4.{bkg_name}.npy'), allow_pickle=True)[()]

#     for m_chi_4 in _m_chi_4:
#         chi_hdr.append(f'p.{n_pg_l4}.4.{m_chi_4}')
#         chi_afreq_list.append(p_n['mode_afreq'])
#         Q_list.append(0)
#         m_chi_list.append(m_chi_4)
#         chi_surf_disp.append(p_n['xir_arr'][-1])
#         W_list.append(W_dict[(4,m_chi_4)])
    
    # for m_chi_4 in _m_chi_4:
    #     chi_hdr.append(f'g.{n_pg_l4}.4.{m_chi_4}')
    #     chi_afreq_list.append(g_n['mode_afreq'])
    #     Q_list.append(0)
    #     m_chi_list.append(m_chi_4)
    #     chi_surf_disp.append(g_n['xir_arr'][-1])
    #     W_list.append(W_dict[(4,m_chi_4)])

N_CHI_L4 = len(chi_surf_disp) - N_CHI_L2 - N_CHI_L0

Q_list = np.array(Q_list)
W_list = np.array(W_list)
QW_list = Q_list*W_list 

print('Constructing 3-mode coupling matrix and nonlinear tide matrix')
k4chi_tensor = np.zeros((len(chi_hdr), len(chi_hdr), len(chi_hdr), len(chi_hdr)), np.complex128)
k3_mat_list = np.zeros((len(chi_hdr), len(chi_hdr), len(chi_hdr)), dtype=np.complex128)
JW_mat = np.zeros((len(chi_hdr), len(chi_hdr)), dtype=np.complex128)
mJW_mat = np.zeros((len(chi_hdr), len(chi_hdr)), dtype=np.complex128)
m5JW_mat = np.zeros((len(chi_hdr), len(chi_hdr)), dtype=np.complex128)

def test_triangle(a, b, c):
    tri = True
    if (np.abs(a-c) <= b and b <= np.abs(a+c)) and (np.abs(a-b) <= c and c <= np.abs(a+b)) and (np.abs(b-c) <= a and a <= np.abs(b+c)):
        tri = True
    else:
        tri = False
    return tri

for i, chi_i in enumerate(chi_hdr):
    for j, chi_j in enumerate(chi_hdr):
        for k, chi_k in enumerate(chi_hdr):
            for h, chi_h in enumerate(chi_hdr):
                m1, m2, m3, m4 = int(chi_i.split('.')[-1]), int(chi_j.split('.')[-1]), int(chi_k.split('.')[-1]), int(chi_h.split('.')[-1])
                if m1 + m2 + m3 + m4 == 0:
                    key = tuple(sorted((chi_i, chi_j, chi_k, chi_h)))
                    # if key == tuple(sorted(('g.1.2.2', 'g.1.2.2', 'g.1.2.-2', 'g.1.2.-2'))) or key == tuple(sorted(('g.1.2.2', 'g.1.2.2', 'g.1.2.-2', 'f.0.2.-2'))) or key == tuple(sorted(('g.1.2.2', 'g.1.2.-2', 'g.1.2.-2', 'f.0.2.2'))) or key == tuple(sorted(('g.1.2.2', 'g.1.2.2', 'f.0.2.-2', 'f.0.2.-2'))) or key == tuple(sorted(('g.1.2.-2', 'g.1.2.-2', 'f.0.2.2', 'f.0.2.2'))):
                    k4chi_tensor[i,j,k,h] = k4chi1_dict[key] #* factor
                        
del i,j,k,h,chi_i,chi_j,chi_k,chi_h

for i, chi_i in enumerate(chi_hdr):
    for j, chi_j in enumerate(chi_hdr):
        for k, chi_k in enumerate(chi_hdr):
            l1, l2, l3 = int(chi_i.split('.')[-2]), int(chi_j.split('.')[-2]), int(chi_k.split('.')[-2])
            m1, m2, m3 = int(chi_i.split('.')[-1]), int(chi_j.split('.')[-1]), int(chi_k.split('.')[-1])
                
            if test_triangle(l1, l2, l3) and (m1 + m2 + m3 == 0):
                key = tuple(sorted((chi_i, chi_j, chi_k)))
                k3_mat_list[i,j,k] = k3_dict[key] * (1+0j) 
            else:
                k3_mat_list[i,j,k] = 0 + 0j
del i,j,k,chi_i,chi_j,chi_k

for i, chi_i in enumerate(chi_hdr):
    for j, chi_j in enumerate(chi_hdr):
        m1, m2 = int(chi_i.split('.')[-1]), int(chi_j.split('.')[-1])
        m_tide = -(m1 + m2)
        if np.abs(m_tide) <= 2:
            key = tuple(sorted((f'tide.2.{m_tide}', chi_i, chi_j)))
            JW_mat[i,j] = J_dict[key]*W_dict[(2,m_tide)] * (1+0j)  
        else:
            JW_mat[i,j] = 0 + 0j
del i,j,chi_i,chi_j


for i, chi_i in enumerate(chi_hdr):
    for j, chi_j in enumerate(chi_hdr):
        m1, m2 = int(chi_i.split('.')[-1]), int(chi_j.split('.')[-1])
        m_tide = -(m1 + m2)
        if np.abs(m_tide) <= 2:
            key = tuple(sorted((f'tide.2.{m_tide}', chi_i, chi_j)))
            mJW_mat[i,j] = m_tide*J_dict[key]*W_dict[(2,m_tide)] * (1+0j)  
        else:
            mJW_mat[i,j] = 0 + 0j
del i,j,chi_i,chi_j

for i, chi_i in enumerate(chi_hdr):
    for j, chi_j in enumerate(chi_hdr):
        m1, m2 = int(chi_i.split('.')[-1]), int(chi_j.split('.')[-1])
        m_tide = -(m1 + m2)
        if np.abs(m_tide) <= 2:
            key = tuple(sorted((f'tide.2.{m_tide}', chi_i, chi_j)))
            m5JW_mat[i,j] = m_tide**5*J_dict[key]*W_dict[(2,m_tide)] * (1+0j)  
        else:
            m5JW_mat[i,j] = 0 + 0j
del i,j,chi_i,chi_j

print('Complete')
chi_is_not_g = np.array(['f.0.2' in hdr for hdr in chi_hdr], dtype=np.float64)

@njit 
def dchi_dt(chi_phys_conj, chi_re, chi_im, r, dr_dt, dphi_dt, eps, QW_list, chi_afreq_list, m_chi_list, k3_mat_list, JW_mat, m5JW_mat, M_NS, R_NS, M_tot, M_prime, chi_is_not_g, k4chi, chi_resc,
            g122_amp, g12n2_amp):    
    delta_afreq = chi_afreq_list - m_chi_list*dphi_dt

    _dchi_dt_re = delta_afreq*chi_im  + (L_TIDE+1)*dr_dt/r*chi_re # last term added
    _dchi_dt_im = -delta_afreq*chi_re + chi_afreq_list*eps*QW_list/2 + (L_TIDE+1)*dr_dt/r*chi_im # last term added

    '''
    NL
    '''
    for i_chi in range(len(chi_re)):
        k3_mat = k3_mat_list[i_chi]
        driving = 1j*chi_afreq_list[i_chi]/2*(np.dot(np.dot(chi_phys_conj,k3_mat),chi_phys_conj))
        driving_re = np.real(driving)
        driving_im = np.imag(driving)
        _dchi_dt_re[i_chi] += driving_re
        _dchi_dt_im[i_chi] += driving_im

    nl_tide_driving = 1j*chi_afreq_list/2*(np.dot(JW_mat*eps, chi_phys_conj))
    nl_tide_re = np.real(nl_tide_driving)
    nl_tide_im = np.imag(nl_tide_driving)

    _dchi_dt_re += nl_tide_re
    _dchi_dt_im += nl_tide_im

    k4_driving = 1j*chi_afreq_list/2*np.dot(k4chi, chi_phys_conj)
    k4_driving_re = np.real(k4_driving)
    k4_driving_im = np.imag(k4_driving)    
    _dchi_dt_re += k4_driving_re
    _dchi_dt_im += k4_driving_im

    '''
    g-anharm, f-br
    '''
    g120_amp = np.conjugate(chi_phys_conj[G120_IDX])
    f120_amp = np.conjugate(chi_phys_conj[0])
    f122_amp = chi_phys_conj[2]
    f12n2_amp = chi_phys_conj[1] # opposite beecause it's conjugated

    _canc = K4_CANC * g122_amp*g12n2_amp + \
            ((GGGF_1_K3)*f122_amp*g12n2_amp + GGGF_1_J*eps*g12n2_amp) + \
            ((GGGF_2_K3)*f12n2_amp*g122_amp + GGGF_2_J*eps*g122_amp) + \
            (GGFF_K3*f120_amp**2 + GGFF_J*eps*f120_amp) + \
            (GGFF_2_K3*np.abs(f122_amp)**2) + \
            (EXTRA * f122_amp**2 * g12n2_amp/g122_amp) + \
            (EXTRA_2 * np.abs(f122_amp)**2) + \
            (EXTRA_3 * np.abs(f122_amp)**2*f122_amp/g122_amp)
    cancellation = np.array([_canc, np.conjugate(_canc)]) * 1j*_g1_afreq/2 * np.conjugate(chi_phys_conj[G122_IDX:G12N2_IDX+1])

    # f_mode_br_n2 = ((FGGG_1_K3_BR+ FGGG_2_K3_BR)*np.abs(g122_amp)**2*g12n2_amp + \
    #                  GGFF_2_K3*np.abs(g12n2_amp)**2 * f12n2_amp  + \
    #                  EXTRA * f122_amp * g12n2_amp**2)

    # f_mode_br_2 = np.conjugate(f_mode_br_n2)
    f_mode_br_2 = F1 * np.abs(g122_amp)**2 * g122_amp + \
                  F2 * g122_amp**2 * f12n2_amp + \
                  F3 * np.abs(g122_amp)**2 * f122_amp + \
                  F4 * np.abs(f122_amp)**2 * g122_amp + \
                  F5 * g12n2_amp * f122_amp**2 + \
                  F6 * np.abs(f122_amp)**2 * f122_amp
    f_mode_br_n2 = np.conjugate(f_mode_br_2)

    f_mode_br_0 = 1j*chi_afreq_list[0]/2* GGFF_K3*np.abs(g122_amp)**2 * f120_amp 

                   
    canc_re = np.real(cancellation)
    canc_im = np.imag(cancellation)
    _dchi_dt_re[G122_IDX:G12N2_IDX+1] += canc_re
    _dchi_dt_im[G122_IDX:G12N2_IDX+1] += canc_im

    _dchi_dt_re[0] += np.real(f_mode_br_0)
    _dchi_dt_im[0] += np.imag(f_mode_br_0)

    
    _dchi_dt_re[1] += np.real(1j*chi_afreq_list[0]/2*f_mode_br_2) 
    _dchi_dt_im[1] += np.imag(1j*chi_afreq_list[0]/2*f_mode_br_2) 
    _dchi_dt_re[2] += np.real(1j*chi_afreq_list[0]/2*f_mode_br_n2)
    _dchi_dt_im[2] += np.imag(1j*chi_afreq_list[0]/2*f_mode_br_n2)
    
    
    return _dchi_dt_re*chi_resc/eps, _dchi_dt_im*chi_resc/eps
                          

@njit
def orbital_reaction(chi_re, chi_im, chi_phys_plus, chi_phys, chi_phys_conj, r, m_chi_list, mJW_mat, QW_list, M_NS, E0, eps, M_tot, M_red):
    U = QW_list*eps
    g_r_lin = (2+1)*U*chi_re
    g_phi_lin = m_chi_list*U*(-chi_im)
    
    chi_U_chi = np.dot(np.dot(chi_phys_plus,JW_mat),chi_phys)*eps  # eps for Uablm # factor
    mtide_chi_U_chi = np.dot(np.dot(chi_phys_plus, mJW_mat), chi_phys)*eps # eps for Uablm # factor
    
    g_r_nlin = 1/2*(2+1)*np.real(chi_U_chi)  #NL
    g_phi_nlin = 1/2*np.imag(mtide_chi_U_chi)  #NL

    
    g_r = -2*E0/M_red/r*(np.sum(g_r_lin) + g_r_nlin)
    g_phi = 2*E0/M_red/r*(np.sum(g_phi_lin) + g_phi_nlin)

    return N_NS*g_r, N_NS*g_phi/r


    
@njit
def d2r_dt2(r, dr_dt, dphi_dt, M_NS, g_r, M_tot):
    # Equations to include: (28), (31)
    
    ######
    # tidal acceleration (28) given by g_r
    ######

    ######
    # gw,pp (assuming equal mass binary) (31)
    gw = 16*(M_NS**2)/(5*r**3) * dr_dt * (dr_dt**2 + 6*r**2 * dphi_dt**2 + 4*G*M_tot/(3*r)) * G**2/c**5 # bt - pp
    ######
    
    _d2r_dt2 = r*dphi_dt**2 - G*M_tot/r**2  + gw + g_r
    
    return _d2r_dt2 

@njit
def d2phi_dt2(chi_re, r, dr_dt, dphi_dt, chi_afreq_list, m_chi_list, QW_list, M_NS, R_NS, E0, eps, g_phi_r, M_red):
    # Equations to include (29), (32)+(34)=(35), (33)
    M_red = M_NS/2
    
    ######
    # tidal backreaction (29) given by g_phi_r
    ######

    ######
    # Burke-Thorne acceleration, pp + tidal_backreaction (32)+(34)=(35)
    gw = -32/5*G*M_red* r**3 * dphi_dt**5 / c**5
    ######
    
    ######
    # quadropolar oscillation (33) # ccheck
    # real_chi2_phys = chi2_re*eps**2 # need rescaling factor
    # linear_reaction = Q_list*(chi_re)*(m_chi_list != 0)
    # gw_ns = -128/5*np.sqrt(2*np.pi/15)*M_NS*(R_NS**2)*r*dphi_dt**5*G/c**5 * np.sum(linear_reaction)
    gw_ns = 0
    ######

    _d2phi_dt2 = -2*dr_dt*dphi_dt/r + gw/r + gw_ns/r + g_phi_r
    
    return _d2phi_dt2

@njit
def events(t, v, chi_afreq_list, m_chi_list, Q_list, QW_list, n_chi, k3_mat_list, JW_mat, mJW_mat, m5JW_mat, M_NS, R_NS, E0, M_prime, M_tot, M_red, chi_surf_disp, chi_is_not_g, k4chi_tensor, chi_resc):
    eps = (R_NS/v[0])**3
    chi_re_phys = v[4:4+n_chi]*2 / chi_resc * eps
    surf_disp = np.dot(chi_re_phys,chi_surf_disp)
    
    return v[0] - 2*R_NS - N_NS*np.abs(surf_disp) # MULTIPLY 2 to account for the tidal excitation in both stars
events.terminal=True

def deriv(t, v, chi_afreq_list, m_chi_list, Q_list, QW_list, n_chi, k3_mat_list, JW_mat, mJW_mat, m5JW_mat, M_NS, R_NS, E0, M_prime, M_tot, M_red, chi_surf_disp, chi_is_not_g, k4chi_tensor, chi_resc):
    print(v[0]/R_NS, end='\r')
    '''
    v[0] = r
    v[1] = dr/dt
    v[2] = phi
    v[3] = dphi/dt
    v[4:4+n_chi1] = chi_re
    v[4+n_chi1:4+2*n_chi1] = chi_im
    '''
    r = v[0]
    dr_dt = v[1]
    dphi_dt = v[3]
    eps = (R_NS/r)**3
    chi_re = v[4:4+n_chi]/chi_resc*eps
    chi_im = v[4+n_chi:4+2*n_chi]/chi_resc*eps

    '''
    Obtain physical chi amplitudes
    '''
    chi_phys_plus = chi_re + 1j*chi_im
    chi_phys_minus = np.conjugate(chi_phys_plus)
    for i_chi in range(N_CHI_L2):
        if i_chi%3 == 1:
            # FIX THIS IF USING L=0, L=4 tides too.
            chi_phys_minus[[i_chi, i_chi+1]] = chi_phys_minus[[i_chi+1, i_chi]] # now 0, 2, -2 
    for i_chi_l4 in range(N_CHI_L4):
        if i_chi_l4%5 == 1 or i_chi_l4 == 3:
            chi_phys_minus[[N_CHI_L2 + N_CHI_L0 + i_chi_l4, N_CHI_L2 + N_CHI_L0 + i_chi_l4+1]] = chi_phys_minus[[N_CHI_L2 + N_CHI_L0 + i_chi_l4+1, N_CHI_L2 + N_CHI_L0 + i_chi_l4]] 
        

    chi_phys = chi_phys_plus + chi_phys_minus
    g122_amp = chi_phys[G122_IDX]
    g12n2_amp = chi_phys[G12N2_IDX]
    chi_phys_conj = np.conjugate(chi_phys)

    g_r, g_phi_r = orbital_reaction(chi_re, chi_im, chi_phys_plus, chi_phys, chi_phys_conj, r, m_chi_list, mJW_mat, QW_list, M_NS, E0, eps, M_tot, M_red)

    _d2r_dt2 = d2r_dt2(r, dr_dt, dphi_dt, M_NS, g_r, M_tot)

    _d2phi_dt2 = d2phi_dt2(chi_re, r, dr_dt, dphi_dt, chi_afreq_list, m_chi_list, QW_list, M_NS, R_NS, E0, eps, g_phi_r, M_red)

    k4chi = np.dot(np.dot(k4chi_tensor, chi_phys_conj), chi_phys_conj)

    _dchi1_dt_re, _dchi1_dt_im = dchi_dt(chi_phys_conj,
                                          chi_re, chi_im, 
                                          r, dr_dt, dphi_dt, eps, 
                                          QW_list, chi_afreq_list, m_chi_list,
                                          k3_mat_list, JW_mat, m5JW_mat,
                                          M_NS, R_NS, M_tot, M_prime, 
                                          chi_is_not_g,
                                          k4chi,
                                          chi_resc,
                                          g122_amp, 
                                          g12n2_amp)
    
    return [dr_dt,
            _d2r_dt2,
            dphi_dt,
            _d2phi_dt2,
            *_dchi1_dt_re,
            *_dchi1_dt_im]


def optimize(y0_re_im, chi_afreq_list, m_chi_list, dphi_dt0, r0, dr_dt0, QW_list, eps0, k3_mat_list, k4chi_tensor, JW_mat):
    # y0 optimization
    n_chi = len(chi_afreq_list)
    y0_re = y0_re_im[:n_chi]
    y0_im = y0_re_im[n_chi:]
    y0_plus = y0_re + 1j*y0_im
    y0_minus = np.conjugate(y0_plus)
    for i_chi in range(N_CHI_L2):
        if i_chi % 3 == 1:
            y0_minus[[i_chi, i_chi+1]] = y0_minus[[i_chi + 1, i_chi]]

    for i_chi_l4 in range(N_CHI_L4):
        if i_chi_l4 % 5 == 1 or i_chi_l4 % 5 == 3:
            y0_minus[[N_CHI_L2 + N_CHI_L0 + i_chi_l4, N_CHI_L2 + N_CHI_L0 + i_chi_l4+1]] = y0_minus[[N_CHI_L2 + N_CHI_L0 + i_chi_l4 + 1, N_CHI_L2 + N_CHI_L0 + i_chi_l4]]
    y0_phys = (y0_plus + y0_minus)*eps0
    y0_phys_conj = np.conjugate(y0_phys)
    
    
    pref = 1j*(chi_afreq_list - m_chi_list*dphi_dt0 + 1j*(L_TIDE+1)*dr_dt0/r0)
    RHS = 1j*chi_afreq_list/2*(QW_list*eps0 + \
                               np.dot(JW_mat*eps0, y0_phys_conj) + \
                               np.dot(np.dot(k3_mat_list, y0_phys_conj), y0_phys_conj) + \
                               np.dot(np.dot(np.dot(k4chi_tensor, y0_phys_conj), y0_phys_conj), y0_phys_conj))
    RHS = RHS/(eps0*pref)

    diff = y0_plus - RHS
    diff_re = np.abs(np.real(diff))
    diff_im = np.abs(np.imag(diff))
    
    return np.array([*diff_re, *diff_im])

'''
Main body
'''

# K4_CANC = 192881625.37322968
# GGGF_1_K3 = 1134985.1530595236 # multiply by x_f2 * x_gn2 
# GGGF_1_J = 11109.669144006504 # multiply by eps * x_gn2
# GGGF_2_K3 = 329871.4870493489 # multiply by x_fn2 * x_g2 
# GGGF_2_J = 2974.419345794428 # multiply by eps * x_g2
# GGFF_K3 = 1928.7874017767688 # multiply by x_f0^2
# GGFF_J = -19.610743955962157 # multiply by eps * xf_0
K4_CANC = 192881625.37322968
GGGF_1_K3 = 1134985.1530595236 + 1134985.1530595236 + 2*329871.4870493489
GGGF_1_J = 11109.669144006504*0 #negligible magnitude compared to GGGF_1_K3*f_Q*W22
GGGF_2_K3 = 329871.4870493489 + 1134985.1530595236
GGGF_2_J = 2974.419345794428*0 #negligible magnitude compared to GGGF_2_K3*f_Q*W22
GGFF_K3 = 1928.7874017767688*0
GGFF_J = -19.610743955962157*0#negligible magnitude compared to GGFF_K3*f_Q*W20
GGFF_2_K3 = -2966.3953902803887

EXTRA = 11725.354524657037
EXTRA_2 = 10223.71877604652 + 4981.303938779466
EXTRA_3 = 25.72752106711592

FGGG_1_K3_BR = 1134985.1530595236
FGGG_2_K3_BR = 329871.4870493489

F1 = 329871.4870493489 + 1135003.5943158532
F2 = 10223.718776046524 + 1501.635748610523
F3 = 10223.718776046524 + 2490.651969389732 + -2966.3953902803887
F4 = -35.86425415300735 -35.86425415300735 + 25.72752106711592
F5 = 25.72752106711592 + -35.86425415300735
F6 = (0.6117945786901267 + 0.14812770521023821)
    
if __name__ == '__main__':
    # f_out = f'PRL.{bkg_name}.{N_CHI_L0}.{N_CHI_L2}.{N_CHI_L4}.anharmonic_g.hdf5'
    f_out = 'PRL.SLy4.2.anharm_g.nc_f.fixed_br.2ns.p0check2.hdf5'
    # f_out = 'PRL.SLy4.2.f_only.nc_f.fixed_br.nonlinear.hdf5'
    print(f'Solution will be saved at {f_out}')
    
    R_NS = bkg.R_NS
    M_NS = bkg.M_NS
    E0 = G*M_NS**2/R_NS
    
    # print('Matching Hang\'s computation')
    # R_NS = 12*1e5
    # M_NS = 1.3*M_sun
    # E0 = G*M_NS**2/R_NS

    M_prime = M_NS
    M_tot = M_NS + M_prime
    M_red = M_NS*M_prime/M_tot
    
    r0 = 20*R_NS
    print('Initial separation', r0/R_NS)
    solver_D = Evolver_D(r0, M_NS, M_prime, R_NS) 
    eps0 = (R_NS/r0)**3
    phi0 = 0
    
    dphi_dt0 = solver_D.get_Omega(r0)
    dr_dt0 = solver_D.get_dD_dt(r0)
    d2r_dt20 = 192*G**3/(5*c**5)*(M_NS**2 * (2*M_NS)/r0**4 * dr_dt0) # M_prime
    d2phi_dt20 = solver_D.get_dOmega_dt(r0)

    t0 = 0
    
    n_chi = len(chi_hdr)
    print('chi list:', chi_hdr)
    
    chi_afreq_list = np.array(chi_afreq_list)
    m_chi_list = np.array(m_chi_list)
    QW_list = np.array(QW_list)
    
    y0 = np.array(chi_afreq_list*QW_list, dtype=np.complex128)/2
    y0 /= 1j*(chi_afreq_list - m_chi_list*dphi_dt0 + 1j*(L_TIDE+1)*dr_dt0/r0)
    y0 *= (1j - m_chi_list*d2phi_dt20/(chi_afreq_list - m_chi_list*dphi_dt0)**2)

    y0_minus = np.conjugate(y0)
    print(N_CHI_L2)
    for i_chi in range(N_CHI_L2):
        if i_chi%3 == 1:
            y0_minus[[i_chi+1, i_chi]] = y0_minus[[i_chi, i_chi+1]]
    y0_phys = (y0 + y0_minus)*eps0

    y0_phys_conj = np.conjugate(y0_phys)

    '''
    NL
    '''
    y0_k3 = 1j*chi_afreq_list*np.dot(np.dot(k3_mat_list, y0_phys_conj), y0_phys_conj)/2/(1j*(chi_afreq_list - m_chi_list*dphi_dt0 + 1j*(L_TIDE+1)*dr_dt0/r0))/eps0
    y0_k4 = 1j*chi_afreq_list*np.dot(np.dot(np.dot(k3_mat_list, y0_phys_conj), y0_phys_conj), y0_phys_conj)/2/(1j*(chi_afreq_list - m_chi_list*dphi_dt0 + 1j*(L_TIDE+1)*dr_dt0/r0))/eps0

    y0 = y0 + y0_k3 + y0_k4

    y0_re = np.real(y0)
    y0_im = np.imag(y0)
    y0_re_im = np.array([*y0_re, *y0_im])
    y0_re_im_opt = fsolve(optimize, y0_re_im, args=(chi_afreq_list, m_chi_list, dphi_dt0, r0, dr_dt0, QW_list, eps0, k3_mat_list, k4chi_tensor, JW_mat)) # NL
    # y0_re_im_opt = y0_re_im

    
    print('Initial estimates', y0_re_im)
    print('Optimized estimates', y0_re_im_opt)
    perc_diff = (y0_re_im_opt - y0_re_im)*100/y0_re_im
    print('% difference', perc_diff[:len(m_chi_list)], perc_diff[len(m_chi_list):])
    y0_re_opt = y0_re_im_opt[:len(m_chi_list)]
    y0_im_opt = y0_re_im_opt[len(m_chi_list):]
    
    y0_resc = np.ones_like(y0_re_opt)#1/np.abs(y0_re_opt + 1j*y0_im_opt)
    y0_re_opt *= y0_resc
    y0_im_opt *= y0_resc

    print('Rescaled to')
    print(y0_re_opt, y0_im_opt)

    print('Initial tidal forcing frequency')
    print(2*dphi_dt0, 'rad Hz')
    print('Mode angular frequencies (sorted)')
    print(np.sort(chi_afreq_list))

    
    print('Solving IVP')
    begin_time = time()
    sol = solve_ivp(deriv, t_span=(0, 2*solver_D.t_merge), args=(np.array(chi_afreq_list, dtype=np.float64),
                                                               np.array(m_chi_list, dtype=np.float64),
                                                               np.array(Q_list, dtype=np.float64),
                                                               np.array(QW_list, dtype=np.float64),
                                                               n_chi,
                                                               np.array(k3_mat_list, dtype=np.complex128),
                                                               np.array(JW_mat, dtype=np.complex128),
                                                               np.array(mJW_mat, dtype=np.complex128),
                                                               np.array(m5JW_mat, dtype=np.complex128),
                                                               M_NS,
                                                               R_NS,
                                                               E0,
                                                               M_prime,
                                                               M_tot,
                                                               M_red,
                                                               np.array(chi_surf_disp, dtype=np.float64),
                                                               np.array(chi_is_not_g, dtype=np.float64),
                                                               np.array(k4chi_tensor, dtype=np.complex128),
                                                               np.array(y0_resc, dtype=np.float64)),
                   y0=np.array([r0, dr_dt0, phi0, dphi_dt0,
                                *y0_re_opt, *y0_im_opt], dtype=np.float64), atol=1e-9, rtol=1e-9,  # atol 1e-9 changed for npg5
                   method='RK45',
                   events=events)
    end_time = time()
    print(sol['message'])
    print(f'Contact at {sol.y[0][-1]/R_NS}') 
    print(f'Computation completed after {end_time-begin_time} seconds')

    
    with h5py.File(f_out, 'w') as f_hdf5:
        f_hdf5['t'] = sol.t
        f_hdf5['y'] = sol.y
        f_hdf5['QW'] = QW_list
        f_hdf5['chi_hdr'] = chi_hdr
        f_hdf5['bkg_name'] = BKG_NAME
        f_hdf5['resc'] = y0_resc
        f_hdf5['eps'] = (bkg.R_NS/sol.y[0])**(L_TIDE+1)
    
    print(f'Solution saved at {f_out}')
    