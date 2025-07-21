import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
use_GPU = True
if not use_GPU:
    print('not using GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from MBD_decomposer import Fmbd_Decomposer
from mbd_tf import MBDEvaluator

# Probing dof
dof_prob = 1

# Geo set up of
ang = 1/0.529177249  # 1 A = 1/0.529177249 bohr
h = 10
NatmC1 = 10
NatmC2 = 10
Natm = NatmC1 + NatmC2
beta = 1

# MBD parameters
ratio_list = np.ones(Natm)*0.78
alpha_0 = [12 for i in range(0, Natm)] * ratio_list
C6 = [46.6 for i in range(0, Natm)] * ratio_list ** 2
R_vdw = [3.59 for i in range(0, Natm)] * ratio_list ** (1 / 3)

# Initialize the carbon chain
XC1 = np.linspace(0, (NatmC1 - 1) * 1.2, NatmC1)
YC1 = h * np.ones(NatmC1)
ZC1 = np.zeros(NatmC1)
XC2 = np.linspace((NatmC2 - 1) * 1.2, 0, NatmC2)
YC2 = np.zeros(NatmC2)
ZC2 = np.zeros(NatmC2)

XC1 = XC1 - np.average(XC2)
XC2 = XC2 - np.average(XC2)

XX0 = np.concatenate((XC1, XC2)) * ang
YY0 = np.concatenate((YC1, YC2)) * ang
ZZ0 = np.concatenate((ZC1, ZC2)) * ang
coords = np.vstack((XX0, YY0, ZZ0)).T

# Compute the analytical atomic MBD forces
mbd_ts = MBDEvaluator(hessians=False, method='ts')
EvdW_ts, FvdW_ts = mbd_ts(coords, alpha_0, C6, R_vdw, beta)

# decomposer class
decompose = Fmbd_Decomposer(mode='decompose')
Bmat = Fmbd_Decomposer(mode='Bmat')
dC = Fmbd_Decomposer(mode='dC')

# Output
print('decomposing force...')
F_iajb = decompose(Natm, [dof_prob], coords, alpha_0, C6, R_vdw, beta)
print('eigen decomposition...')
eig, S = Bmat(Natm, [dof_prob], coords, alpha_0, C6, R_vdw, beta)  # Used to construct B or C matrix
print('computing dC...')
dC_mat = dC(Natm, [dof_prob], coords, alpha_0, C6, R_vdw, beta)

# Retain non-zero component of dC
reduce_dC_mat = np.zeros([3 * Natm, 3 * Natm], dtype=np.float64)
for i in range(Natm):
    for a in range(3):
        for j in range(Natm):
            for b in range(3):
                reduce_dC_mat[3 * i + a, 3 * j + b] = dC_mat[3 * i + a, 3 * j + b, i, dof_prob]
