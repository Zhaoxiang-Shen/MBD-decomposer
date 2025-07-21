"""
This code is developed to support pairwise decomposition of many-body dispersion (MBD) forces.
User can choose to output a direct decomposition, or a eigen-decomposition of the dipole interaction matrix,
or its derivatives. These can support 'pairwise' analysis of MBD interaction,
and future development in ML surrogate modeling of MBD.

The MBD calculation is based on a TensorFlow implementation, see details in
Z. Shen et al. 2024, [https://github.com/iansosa/QC-Toolkit]
"""


import math
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
inf = float('inf')
tf_precision = tf.float64
pi = tf.constant(math.pi, tf_precision)
import time

class Fmbd_Decomposer(object):
    def __init__(self, mode='decompose'):
        self._inputs = coords, alpha_0, C6, R_vdw, beta, coords_indicator1, coords_indicator2 = [
            tf.placeholder(tf_precision, shape=shape, name=name)
            for shape, name in [
                ((None, 3), 'coords'),
                ((None, ), 'alpha_0'),
                ((None, ), 'C6'),
                ((None, ), 'R_vdw'),
                ((), 'beta'),
                ((None, None), 'ind1'),
                ((None, None), 'ind2')
            ]
        ]
        self.mode = mode
        self._output = predecomposer(*self._inputs,mode=mode)

    def __call__(self, Natm, dof_probe, coords, alpha_0, C6, R_vdw, beta, zero_diag=True, choose_i=None):
        if self.mode == 'decompose':
            f_iajb_matrix = np.zeros([3 * Natm, 3 * Natm], dtype=np.float64)
            indicator_val = np.zeros((Natm, 3))
            if choose_i is not None:
                list_i = choose_i
            else:
                list_i = range(Natm)

            with tf.Session() as sess:
                for i in list_i:
                    start = time.time()
                    for a in dof_probe:
                        indicator_val[i, a] = 1
                        inputs = dict(zip(self._inputs, [coords, alpha_0, C6, R_vdw, beta,
                                                         np.ones((3*Natm,3*Natm)),indicator_val]))
                        output = sess.run(self._output, inputs)
                        f_iajb_matrix[3 * i + a, :] = output
                        if zero_diag:  # Spurious non-zero diagonal due to double counting using indicator.
                            f_iajb_matrix[3 * i + a, 3 * i:3 * i + 3] = 0
                        indicator_val[i, a] = 0
                    end = time.time()
                    print('indexing atom %d, cost: %.3fs' % (i, end - start))
            return f_iajb_matrix

        elif self.mode == 'Bmat':
            with tf.Session() as sess:
                inputs = dict(zip(self._inputs, [coords, alpha_0, C6, R_vdw, beta,
                                                 np.ones((3*Natm, 3*Natm)), np.ones((3*Natm, 3))]))
                output = sess.run(self._output, inputs)
            return output

        elif self.mode == 'dC':
            dC = np.zeros([3 * Natm, 3 * Natm, Natm, 3], dtype=np.float64)
            indicator_val = np.zeros((3 * Natm, 3 * Natm))
            with tf.Session() as sess:
                for ia in range(3 * Natm):
                    start = time.time()
                    for jb in range(3 * Natm):
                        indicator_val[ia, jb] = 1
                        inputs = dict(zip(self._inputs,
                                          [coords, alpha_0, C6, R_vdw, beta, np.ones((3 * Natm, 3 * Natm)),
                                           indicator_val]))
                        output = sess.run(self._output, inputs)
                        dC[ia, jb, :, :] = output

                        indicator_val[ia, jb] = 0
                    end = time.time()
                    print('indexing atom dof %d, cost: %.3fs' % (ia, end - start))
            return dC

def predecomposer(coords, alpha_0, C6, R_vdw, beta, coords_indicator1, coords_indicator2, mode='decompose'):
    # build C matrix
    omega = 4/3*C6/alpha_0**2
    sigma = (tf.sqrt(2/pi)*(alpha_0)/3)**(1/3)
    dipmat = dipole_matrix(coords, beta, R_vdw, sigma)
    pre = _repeat(omega*tf.sqrt(alpha_0), 3)
    C = tf.diag(_repeat(omega**2, 3))+pre[:, None]*pre[None, :]*dipmat
    eigs, S = tf.linalg.eigh(C)
    Lambda_val = tf.stop_gradient(tf.abs(eigs))
    S_val = tf.stop_gradient(S)
    S_ind = coords_indicator1 * S_val  # Using indicator to decompose forces

    #####
    Lambda_inv = tf.linalg.inv(tf.diag(tf.sqrt(Lambda_val)))
    F = tf.matmul(Lambda_inv,S_ind,transpose_b=True)
    F = tf.matmul(F, C)
    F = tf.matmul(F, S_ind)
    F = -1/4*tf.trace(F)
    F = tf.gradients(F, coords, unconnected_gradients='zero')[0]

    if mode == 'decompose':
        # add a second indicators for each force component
        F_ind = F * coords_indicator2
        F_full = tf.gradients(F_ind, coords_indicator1, unconnected_gradients='zero')[0]
        Fj_full = tf.reduce_sum(F_full, 1)
        return Fj_full
    elif mode == 'Bmat':
        return eigs, S
    elif mode == 'dC':
        return tf.gradients(C*coords_indicator2, coords, unconnected_gradients='zero')[0]
    else:
        raise ValueError('Unsupported decomposer method')

def damping_fermi(R, S_vdw, d):
    return 1/(1+tf.exp(-d*(R/S_vdw-1)))

def T_bare(R):
    inf = tf.constant(math.inf, tf_precision)
    R_2 = tf.reduce_sum(R**2, -1)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_5 = _set_diag(R_1**5, inf)
    return (
        -3*R[:, :, :, None]*R[:, :, None, :]
        + R_2[:, :, None, None]*np.eye(3)[None, None, :, :]
    )/R_5[:,:,None,None]

def _set_diag(A, val):
    return tf.matrix_set_diag(A, tf.fill(tf.shape(A)[0:1], tf.cast(val, tf_precision)))

def _repeat(a, n):
    return tf.reshape(tf.tile(a[:, None], (1, n)), (-1,))

def dipole_matrix(coords, beta, R_vdw=None, sigma=None):
    Rs = coords[:, None, :]-coords[None, :, :]
    sigmaij = tf.sqrt(sigma[:, None]**2+sigma[None, :]**2)
    dipmat = T_erf_coulomb(Rs, sigmaij, beta)
    n_atoms = tf.shape(coords)[0]
    return tf.reshape(tf.transpose(dipmat, (0, 2, 1, 3)), (3*n_atoms, 3*n_atoms))

def T_erf_coulomb(R, sigma, beta):
    pi = tf.constant(math.pi, tf_precision)
    inf = tf.constant(math.inf, tf_precision)
    bare = T_bare(R)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_5 = _set_diag(R_1**5, inf)
    RR_R5 = R[:, :, :, None]*R[:, :, None, :]/R_5[:, :, None, None]
    zeta = R_1/(sigma*beta)
    theta = 2*zeta/tf.sqrt(pi)*tf.exp(-zeta**2)
    erf_theta = tf.erf(zeta) - theta
    return erf_theta[:, :, None, None]*bare + \
        (2*(zeta**2)*theta)[:, :, None, None]*RR_R5


