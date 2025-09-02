import argparse
import autograd.numpy as anp
import numpy as np
from scipy.special import logsumexp
from math import isnan
from autograd import grad
import os
from scipy.stats import poisson
import argparse
import time
from numba import njit

@njit
def update_params_rowwise_numba(V, D, K, i, Theta_IC, w_KC, Theta_IK, phi_DK, phi_VDK):
    Theta_IK[i, :] = w_KC @ Theta_IC[i, :]
    phi_VK = Theta_IK[i, :].copy()
    for d in range(D):
        if d == 0:
            phi_DK[d, :] += phi_VK
            phi_VDK[i, d, :] = phi_VK
        else:
            new_phi = (phi_VK * Theta_IK[i, :]) / (d + 1)
            phi_DK[d, :] += new_phi
            phi_VDK[i, d, :] = new_phi
            phi_VK = new_phi - phi_VK * Theta_IK[i, :]

    for d in range(D):
        for k in range(K):
            if phi_DK[d, k] < 1e-10:
                phi_DK[d, k] = 1e-10
            phi_DK[d, k] = np.log(phi_DK[d, k])
            if phi_VDK[i, d, k] < 1e-10:
                phi_VDK[i, d, k] = 1e-10
            phi_VDK[i, d, k] = np.log(phi_VDK[i, d, k])
    return Theta_IK, phi_DK, phi_VDK

    def update_theta_iC_full(self, i, Y_C):
        C = self.C
        K = self.K
        phi_vDK = np.exp(self.phi_VDK[i, :, :])
        phi_vDC = phi_vDK[:, :C]
        d_vals = self.gammas_DK[self.min_order:self.D, :]
        phi_slice = phi_vDK[self.min_order-1:self.D-1, :]
        d_val_sum = np.sum(d_vals * phi_slice, axis=0)
        denominators_C = (d_val_sum @ self.w_KC).T
        d_range = np.arange(self.min_order, self.D)[:, None, None]
        w_KC_power = self.w_KC[C:K, :][None, :, :] ** (d_range - 1)
        gammas_slice = self.gammas_DK[d_range[:,0,0]-1, C:K][:, :, None]
        phi_DC_slice = phi_vDC[d_range[:,0,0]-2, :][:, None, :]
        wp_KC = w_KC_power * phi_DC_slice
        term_KC = gammas_slice * wp_KC
        denominators_C -= np.sum(term_KC, axis=(0,1))
        return Y_C / denominators_C

@njit
def update_theta_all_full_numba(V, C, K, D, min_order, Y_VC, phi_VDK, gammas_DK, w_KC, Theta_IC, Theta_IK, phi_DK):
    for i in range(V):
        Y_C = Y_VC[i, :]
        phi_vDK = np.exp(phi_VDK[i, :, :])
        phi_vDC = phi_vDK[:, :C]
        d_vals = gammas_DK[min_order:D, :]
        phi_slice = phi_vDK[min_order-1:D-1, :]
        d_val_sum = np.sum(d_vals * phi_slice, axis=0)
        denominators_C = (d_val_sum @ w_KC).T
        d_range = np.arange(min_order, D)[:, None, None]
        w_KC_power = w_KC[C:K, :][None, :, :] ** (d_range - 1)
        gammas_slice = gammas_DK[d_range[:,0,0]-1, C:K][:, :, None]
        phi_DC_slice = phi_vDC[d_range[:,0,0]-2, :][:, None, :]
        wp_KC = w_KC_power * phi_DC_slice
        term_KC = gammas_slice * wp_KC
        term_sum = np.sum(term_KC, axis=0)  
        denominators_C -= np.sum(term_sum, axis=0) 
        new_theta_iC = Y_C /denominators_C
        new_theta_iC = np.maximum(new_theta_iC, 0)
        new_theta_iK = w_KC @ new_theta_iC
        Theta_IC[i, :] = new_theta_iC
        Theta_IK[i, :] = new_theta_iK
        Theta_IK, phi_DK, phi_VDK = update_params_rowwise_numba(V, D, Theta_IC.shape[1], i,Theta_IC, w_KC, Theta_IK, phi_DK, phi_VDK)
    return Theta_IC, Theta_IK, phi_DK, phi_VDK

@njit
def update_theta_all_numba_fast(V, C, D, min_order, Y_VC, phi_VDK, gammas_DK, w_KC, Theta_IC, Theta_IK, phi_DK):
    for i in range(V):
        start = min_order
        phi_slice = np.exp(phi_VDK[i, start-2:D-1, :])
        d_val_sum = np.sum(gammas_DK[start-1:D, :] * phi_slice, axis=0)
        denominators = d_val_sum @ w_KC
        new_theta_iC = Y_VC[i, :] / (denominators + 1e-10)
        new_theta_iK = w_KC @ new_theta_iC
        Theta_IC[i, :] = new_theta_iC
        Theta_IK[i, :] = new_theta_iK
        Theta_IK, phi_DK, phi_VDK = update_params_rowwise_numba(V, D, Theta_IC.shape[1], i,
                                                                Theta_IC, w_KC, Theta_IK, phi_DK, phi_VDK)
    return Theta_IC, Theta_IK, phi_DK, phi_VDK


def holdout(Y_indices_D, Y_counts_D, start, V):
    Y_indices_test_D = []
    Y_counts_test_D = []
    Y_indices_train_D = []
    Y_counts_train_D = []

    D = len(Y_indices_D)
    
    # Pad the lower-order entries
    for d in range(start-1):
        Y_indices_train_D.append([None])
        Y_counts_train_D.append([None])
        Y_indices_test_D.append([None])
        Y_counts_test_D.append([None])

    for d in range(start-1, D):
        Y_indices = Y_indices_D[d]
        Y_counts = Y_counts_D[d]
        n = Y_indices.shape[0]

        n_test = min(int(np.floor(n * 0.1)), 1000)

        if n_test > 0:
            test_indices = np.random.choice(n, n_test, replace=False)
            train_indices = np.setdiff1d(np.arange(n), test_indices)
        else:
            test_indices = np.array([], dtype=int)
            train_indices = np.arange(n)

        # Generate new random test combinations
        test_indices_zero = np.zeros((n_test, d), dtype=int)
        existing_tuples = {tuple(row) for row in Y_indices}

        for i in range(n_test):
            new = np.random.choice(np.arange(V), d, replace=False)
            while tuple(new) in existing_tuples:
                new = np.random.choice(np.arange(V), d, replace=False)
            existing_tuples.add(tuple(new))
            test_indices_zero[i, :] = new

        if len(test_indices) > 0:
            Y_indices_test = np.vstack([Y_indices[test_indices, :], test_indices_zero])
            Y_counts_test = np.concatenate([Y_counts[test_indices], np.zeros(n_test)])
        else:
            Y_indices_test = test_indices_zero
            Y_counts_test = np.zeros(n_test)

        Y_indices_test_D.append(Y_indices_test)
        Y_counts_test_D.append(Y_counts_test)
        Y_indices_train_D.append(Y_indices[train_indices, :])
        Y_counts_train_D.append(Y_counts[train_indices])

    return Y_indices_test_D, Y_counts_test_D, Y_indices_train_D, Y_counts_train_D


def compute_phi_DK_min(input_mat, D):
    phi_DK = np.sum(input_mat, axis=0, keepdims=True).T
    phi_VK = phi_DK.T - input_mat
    for d in range(2, D+1):
        new_phi_DK_row = (np.sum(phi_VK * input_mat, axis=0, keepdims=True).T) / d
        if d == 2:
            phi_DK = np.vstack([phi_DK.T, new_phi_DK_row.T])
        else:
            phi_DK = np.vstack([phi_DK[:d-1, :], new_phi_DK_row.T])
        phi_VK = new_phi_DK_row.T - phi_VK * input_mat
        phi_VK = np.maximum(phi_VK, 1e-30)
    phi_DK = np.maximum(phi_DK, 1e-30)
    return phi_DK

def load_data(directory, MAX_ORDER=25, MIN_ORDER=2, test=False):
    data = np.load(f"data/{directory}/edges.npz", allow_pickle=True)
    Y_indices_raw = data["edges"].astype(int)
    Y_counts_raw = np.round(data["counts"]).astype(int)

    D = min(Y_indices_raw.shape[1], MAX_ORDER)
    D_uncapped = Y_indices_raw.shape[1]

    if D_uncapped > MAX_ORDER:
        Y_indices_raw = Y_indices_raw[:, :D]

    # Arrays indexed by interaction order 'd'
    Y_indices_D = [None] * (D)
    Y_counts_D = [None] * (D)
    inds_VD = [None] * (D)

    # Fill up to MIN_ORDER - 1 with dummy arrays
    for i in range(1, MIN_ORDER):
        Y_indices_D[i] = np.zeros((1, 1), dtype=int)
        Y_counts_D[i] = np.zeros((1,), dtype=int)

    V = int(np.max(Y_indices_raw))

    # Split data into orders
    for d in range(MIN_ORDER-1, D):
        if D_uncapped > MAX_ORDER:
            indices = [i for i, row in enumerate(Y_indices_raw)
                       if np.sum(row == 0) == D - d]
        else:
            indices = [i for i, row in enumerate(Y_indices_raw)
                       if np.sum(row == 0) == D - d]

        Y_indices_D[d] = Y_indices_raw[indices, :d] - 1
        Y_counts_D[d] = Y_counts_raw[indices] - 1

    # Optionally hold out test set
    if test:
        (Y_indices_test_D, Y_counts_test_D,
         Y_indices_train_D, Y_counts_train_D) = holdout(Y_indices_D, Y_counts_D, MIN_ORDER, V)
        Y_indices_D = Y_indices_train_D
        Y_counts_D = Y_counts_train_D

    # Build inds_VD
    for d in range(MIN_ORDER-1, D):
        Y_indices = Y_indices_D[d]
        inds_V = []
        for i in range(V):
            inds_V.append(find_matching_rows(Y_indices, i))
        inds_VD[d] = inds_V

    if test:
        return V, D, Y_indices_D, Y_counts_D, Y_indices_test_D, Y_counts_test_D, inds_VD
    else:
        return V, D, Y_indices_D, Y_counts_D, inds_VD


def get_parsed_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="supreme-court",
                        help="Path to the input data directory")
    parser.add_argument("--test", action="store_true",
                        help="If set, runs the script in test mode (a boolean flag)", default=False)
    parser.add_argument("--C", type=int, default=3,
                        help="Parameter C for the model")
    parser.add_argument("--K", type=int, default=6,
                        help="Parameter K for the model")
    parser.add_argument("--model", type=str, default="semi",
                        help="Specify the model type (e.g., 'semi', 'omni')")
    parser.add_argument("--MIN_ORDER", type=int, default=2,
                        help="Minimum order for a model component")
    parser.add_argument("--MAX_ORDER", type=int, default=9,
                        help="Maximum order for a model component")
    parser.add_argument("--CONV_TOL", type=float, default=1.0,
                        help="Convergence tolerance for the optimization")
    parser.add_argument("--MAX_ITER", type=int, default=500,
                        help="Maximum number of iterations")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-6,
                        help="Learning rate for the optimizer")
    parser.add_argument("--NUM_RESTARTS", type=int, default=10,
                        help="Number of random restarts for the optimization")
    parser.add_argument("--NUM_STEPS", type=int, default=1,
                        help="Number of steps to run")
    parser.add_argument("--CHECK_EVERY", type=int, default=10,
                        help="Frequency of checking for convergence (in iterations)")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for reproducibility")
    return parser.parse_args(args)

def logsubexp(a, b):
    assert (a > b).all(), a - b
    return a + np.log1p(-np.exp(b - a))

def find_matching_rows(Y_IK, i_prime):
    matching_rows = np.any(Y_IK == i_prime, axis=1)
    true_vals = np.where(matching_rows)[0]
    return true_vals
            
class Model():
    def __init__(self, args):
        if args.test == True:
            self.Y_indices_test_D = args.Y_indices_test_D
            self.Y_counts_test_D = args.Y_counts_test_D
        self.seed = args.seed
        self.directory = args.directory
        self.lr = args.LEARNING_RATE
        self.D = args.D
        self.K = args.K
        self.C = args.C
        self.V = args.V
        self.init()
        self.min_order = args.MIN_ORDER
        self.Y_indices_D = args.Y_indices_D
        self.Y_counts_D = args.Y_counts_D
        self.inds_VD = args.inds_VD
        self.model_type = args.model
        self.old_elbo = -1e10
        self.change_elbo=1e5
        self.ELBOs = []
        self.times = []
        self.heldout_llk_D = np.zeros((0, self.D - self.min_order + 1))
        self.times = []
        self.alpha = 0
        self.beta = 0
    
    def init(self):
        self.init_W_KC, self.log_w_KC = self.init_W()
        self.w_KC = self.init_W_KC.copy()  
        self.gammas_DK = np.ones((self.D, self.K))
        self.Theta_IC = np.random.gamma(1000,1/1000, size=(self.V, self.C))
        #self.Theta_IC /= np.sum(self.Theta_IC) * self.V * self.C
        self.update_params()
    
    def init_W(self):
        C = self.C 
        K = self.K
        w_KC = np.zeros((self.K, self.C))
        for c in range(self.C):
            w_KC[c, c] = 1.0
        for k in range(C, K):
            alpha = np.ones(C) / (k+1 - C) * C
            w_KC[k, :] = np.random.dirichlet(alpha)
        return w_KC, np.log(w_KC)

    def update(self):
        start_time = time.time()
        if self.model_type == "omni":
            self.allocate_all_full()
            self.update_gamma_DK_full()
            #self.Theta_IC, self.Theta_IK, self.phi_DK, self.phi_VDK = update_theta_all_full_numba(self.V, self.C, self.K, self.D, self.min_order, self.Y_VC, self.phi_VDK, self.gammas_DK, self.w_KC, self.Theta_IC, self.Theta_IK, self.phi_DK)
            self.update_theta_all_full()
            #if self.K != self.C: #take gradient step on non-assortative communities
                #self.optimize_Welbo_full()
            self.update_params()
            self.compute_llk_full()
        else:  # semi: faster but less expressive in more disassortative settings
            self.allocate_all()
            self.update_gamma_DK()
            #self.Theta_IC, self.Theta_IK, self.phi_DK, self.phi_VDK = update_theta_all_numba_fast(V=self.V,C=self.C,D=self.D,min_order=self.min_order,Y_VC=self.Y_VC,phi_VDK=self.phi_VDK,gammas_DK=self.gammas_DK,w_KC=self.w_KC,Theta_IC=self.Theta_IC,Theta_IK=self.Theta_IK,phi_DK=self.phi_DK)
            self.update_theta_all()
            #if self.K != self.C:
                #self.optimize_Welbo()
            #self.update_params() # update cached phi_DK, phi_VDK, Theta_IK
            self.compute_llk()
        self.times.append(time.time() - start_time)
        print(self.old_elbo)

    def allocate_all_full(self):
        self.Y_VC = np.zeros((self.V, self.C))
        self.Y_KC = np.zeros((self.K, self.C))
        self.Y_DK = np.zeros((self.D, self.K))

        for d in range(self.min_order-1, self.D):
            Y_indices = self.Y_indices_D[d] 
            Y_counts = self.Y_counts_D[d]   
            gammas_K = self.gammas_DK[d, :] 
            if len(Y_counts) > 0:
                Y_IK, Y_IKC = self.allocate_K_full(Y_indices, Y_counts, gammas_K)    
                self.Y_VC += np.sum(Y_IKC, axis = 1)
                self.Y_KC += np.sum(Y_IKC, axis = 0)
                self.Y_DK[d, :] = np.sum(Y_IK, axis = 0)


    def allocate_K_full(self, Y_indices, Y_counts, gammas_K):
        C = self.C
        K = self.K
        Y_IK = np.zeros((Y_indices.shape[0], self.K))
        Theta_IK = np.maximum(self.Theta_IK, 1e-300)
        for d in range(Y_indices.shape[1]):
            Y_IK += np.log(Theta_IK[Y_indices[:, d], :])
        Y_IK = np.exp(Y_IK) 

        w_KC_d = self.w_KC**self.D
        Y_IC = Y_IK[:, :self.C]
        sub_IK = Y_IC @ w_KC_d.T
        Y_IK[:,C:K] -= sub_IK[:, C:K]
        np.maximum(Y_IK, 1e-300, out=Y_IK)
        Y_IK *= gammas_K.T
        Y_IK /= np.sum(Y_IK, axis=1, keepdims=True)
        Y_IK *= Y_counts[:,None]
        Y_IKC = np.zeros((self.V, K, C))
        w_KC_d = self.w_KC ** (self.D - 1)
        l_w_kc_d = np.log(w_KC_d)
        l_w_kc = np.log(self.w_KC)
        for i in range(len(Y_counts)):
            counts_KC = self.allocate_iCK(Theta_IK, Y_IK[i,:], Y_indices[i,:], w_KC_d, l_w_kc_d, l_w_kc)
            Y_IKC[Y_indices[i,:],:,:] += counts_KC
        return Y_IK, Y_IKC       
    
    def update_gamma_DK_full(self):
        e_phi_DK = np.exp(self.phi_DK)
        self.gammas_DK = np.ones((self.D, self.K))#np.random.gamma(1, (self.D, self.K))
        for d in range(self.min_order-1,self.D):
            for k in range(self.K):
                if k <= self.C-1:
                    self.gammas_DK[d, k] = (self.Y_DK[d,k] + self.alpha) / (e_phi_DK[d,k] + self.beta)
                else:
                    self.gammas_DK[d, k] = (self.Y_DK[d,k] + self.alpha) / np.maximum((e_phi_DK[d,k]+ self.beta - np.sum(e_phi_DK[d, :self.C] * self.w_KC[k, :]**d)), self.beta)
    
    def update_theta_all_full(self):
        for i in range(self.V):
            Y_C = self.Y_VC[i,:]
            new_theta_iC = self.update_theta_iC_full(i, Y_C)
        new_theta_iC = np.maximum(new_theta_iC, 0)
        new_Theta_iK = self.w_KC @ new_theta_iC
        self.Theta_IK[i, :] = new_Theta_iK
        self.Theta_IC[i, :] = new_theta_iC
        self.update_params()
    
    def update_theta_iC_full(self, i, Y_C):
        C = self.C
        K = self.K
        phi_vDK = np.exp(self.phi_VDK[i, :, :])
        phi_vDC = phi_vDK[:, :C]
        d_vals = self.gammas_DK[self.min_order:self.D, :]
        phi_slice = phi_vDK[self.min_order-1:self.D-1, :]
        d_val_sum = np.sum(d_vals * phi_slice, axis=0)
        denominators_C = (d_val_sum @ self.w_KC).T
        d_range = np.arange(self.min_order, self.D)[:, None, None]
        w_KC_power = self.w_KC[C:K, :][None, :, :] ** (d_range - 1)
        gammas_slice = self.gammas_DK[d_range[:,0,0]-1, C:K][:, :, None]
        phi_DC_slice = phi_vDC[d_range[:,0,0]-2, :][:, None, :]
        wp_KC = w_KC_power * phi_DC_slice
        term_KC = gammas_slice * wp_KC
        denominators_C -= np.sum(term_KC, axis=(0,1))
        return Y_C / denominators_C


    def update_phi_DK(self, old, new, i):
        D, K = self.phi_DK.shape
        self.phi_DK[0, :] = logsumexp([self.phi_DK[0, :], np.log(new)], axis=0, keepdims=True)
        self.phi_DK[0, :] = logsubexp(self.phi_DK[0, :], np.log(old))
        for d in range(1, D): 
            self.phi_DK[d, :] = logsumexp([self.phi_DK[d, :], self.phi_VDK[i, d-1, :] + np.log(new)],axis=0, keepdims=True)
            self.phi_DK[d, :] = logsubexp(self.phi_DK[d, :], self.phi_VDK[i, d-1, :] + np.log(old))


    def update_phi_VDK(self, j):
        D, K = self.phi_DK.shape
        theta_k = self.Theta_IK[j, :]
        self.phi_VDK[j, 0, :] = logsubexp(self.phi_DK[0, :], np.log(theta_k))

        for d in range(1, D):  
            self.phi_VDK[j, d, :] = logsubexp(self.phi_DK[d, :],np.log(theta_k) + self.phi_VDK[j, d-1, :])

    
    def allocate_all(self):
        D = self.D
        self.Y_VC = np.zeros((self.V, self.C))
        self.Y_KC = np.zeros((self.K, self.C))
        self.Y_DK = np.zeros((self.D, self.K))
        for d in range(self.min_order-1,D):
            Y_indices = self.Y_indices_D[d]
            Y_counts = self.Y_counts_D[d]
            if len(Y_counts) > 0:
                gammas_K = self.gammas_DK[d, :]
                inds_V = self.inds_VD[d]
                Y_IK = self.allocate(Y_indices, Y_counts, gammas_K)
                Y_VC_raw, Y_KC_raw = self.reallocate(Y_IK, inds_V)
                self.Y_VC += Y_VC_raw
                self.Y_KC += Y_KC_raw
                self.Y_DK[d, :] = np.sum(Y_IK, axis = 0)

    def allocate(self, Y_indices, Y_counts, gammas_K):
            I = self.V
            K = self.K
            log_theta = np.log(self.Theta_IK[Y_indices, :].clip(1e-30))  
            Y_IK = np.sum(log_theta, axis=1)                  
            Y_IK += np.log(gammas_K)[None, :]             
            Y_IK = np.exp(Y_IK - logsumexp(Y_IK, axis=1, keepdims=True)) * Y_counts[:, None]
            return Y_IK
    
    def reallocate(self, Y_IK, inds_V):
        V = self.V
        C = self.C
        K = self.K
        Y_VC = np.zeros((V, C))
        Y_KC = np.zeros((K, C))
        w_IKC = self.w_KC[None,:,:] * self.Theta_IC[:,None,:]
        w_IKC /= np.sum(w_IKC, axis=2, keepdims=True).clip(1e-100)
        for i in range(V):
            y_K = np.sum(Y_IK[inds_V[i],:], axis=0)
            w_KC = w_IKC[i,:,:]
            Y_VC[i, :] = (y_K @ w_KC)
            Y_KC += y_K[:,None] * w_KC
        return Y_VC, Y_KC

    def update_gamma_DK(self):
        self.gammas_DK = (self.Y_DK + self.alpha) / (np.exp(self.phi_DK) + self.beta)
    
    def update_theta_all(self):
        for i in range(self.V):
            self.update_theta_iC_D(i)
        
    def update_theta_iC_D(self, i):
        Y_C = self.Y_VC[i, :]
        old_theta = self.Theta_IK[i,:]
        start, D = self.min_order, self.D
        phi_VK = np.exp(self.phi_DK[0,:]) - self.Theta_IK[i,:]
        phi_vDK = np.zeros((self.D-1, self.K))
        phi_vDK[0,:]=phi_VK
        for d in range(1, self.D-1):
            phi_vDK[d,:] = np.exp(self.phi_DK[d,:]) - phi_vDK[d-1,:]*old_theta
        phi_slice = phi_vDK              
        d_val_sum = np.sum(self.gammas_DK[start-1:D, :] * phi_slice, axis=0)  
        denominators = d_val_sum @ self.w_KC       
        new_theta_iC = (Y_C/denominators)
        new_theta_iK = self.w_KC @ new_theta_iC    
        self.Theta_IK[i, :] = new_theta_iK
        self.Theta_IC[i, :] = new_theta_iC
        self.phi_DK = np.log(compute_phi_DK_min(self.Theta_IC @ self.w_KC.T, self.D))
        #self.update_params()



    def optimize_Welbo(self, steps=1):
        lr = self.lr
        grad_w_KC = np.zeros(self.w_KC.shape)
        for step in range(steps):
            grad_w_KC = self.compute_gradient()
            grad_w_KC = grad_w_KC.clip(min=-1e5, max=1e5)
            grad_w_KC = np.nan_to_num(grad_w_KC, nan=0.0)
            self.log_w_KC += lr*grad_w_KC
        self.w_KC = np.exp(self.log_w_KC)
    
    def optimize_Welbo_full(self, steps=1):
        lr = self.lr
        grad_w_KC = np.zeros(self.w_KC.shape)
        log_w_KC = np.log(self.w_KC)
        for step in range(steps):
            grad_w_KC = self.compute_gradient_full()
            grad_w_KC = grad_w_KC.clip(min=-1e5, max=1e5)
            grad_w_KC = np.nan_to_num(grad_w_KC, nan=0.0)
            log_w_KC += lr*grad_w_KC
            self.log_w_KC = log_w_KC
        self.w_KC = np.exp(self.log_w_KC)

    def compute_gradient_full(self):
        grad_fn = grad(self.compute_Welbo_full)
        return grad_fn(self.log_w_KC)
    
    def compute_gradient(self):
        grad_fn = grad(self.compute_Welbo)
        return grad_fn(self.log_w_KC)

    def compute_Welbo(self, log_w_KC):
        D = self.D
        start = self.min_order - 1
        gammas_DK = self.gammas_DK[start:D,:]
        w_KC = np.exp(log_w_KC)
        phi_DK = compute_phi_DK_min(self.Theta_IC @ w_KC.T, D)
        phi_DK_sub = phi_DK[start:D, :]
        llk = np.sum(self.Y_KC * log_w_KC)
        llk -= np.sum(phi_DK_sub * gammas_DK)
        return llk
    
    def compute_Welbo_full(self, log_w_KC):
        D = self.D
        C = self.C
        start = self.min_order - 1
        gammas_DK = self.gammas_DK[start:D,:]
        w_KC = np.exp(log_w_KC)
        phi_DK = compute_phi_DK_min(self.Theta_IC @ w_KC.T, D)
        phi_DK_sub = phi_DK[start:D, :]
        llk = np.sum(self.Y_KC * log_w_KC)
        llk -= np.sum(phi_DK_sub * gammas_DK)
        for d in range(D-start):
            llk += np.sum(gammas_DK[d,C:] * np.sum(w_KC[C:, :]**(d+start+1) * phi_DK_sub[d,:C].T, axis=1))
        return llk


    def update_params_log(self):
        self.Theta_IK = self.Theta_IC @ self.w_KC.T #Theta_IK is positive
        Theta_IK = self.Theta_IK.astype(np.float64)
        self.phi_DK = np.sum(self.Theta_IK, axis=0, keepdims=True) #sum over nodes
        log_phi_VK = logsubexp(np.log(self.phi_DK), np.log(self.Theta_IK)) #phi_DK
        log_phi_VK = log_phi_VK.astype(np.float64)
        log_phi_DK = np.log(self.phi_DK).T
        log_phi_DK = log_phi_DK.astype(np.float64)
        log_phi_VDK = np.full((self.V, self.D, self.K), -np.inf)
        log_phi_VDK[:, 0, :] = log_phi_VK
        for d in range(1, self.D):
            log_term = log_phi_VK + np.log(Theta_IK)
            log_new_phi = logsumexp(log_term, axis=0, keepdims=True).T - np.log(float(d+1))
            if d == 1:
                log_phi_DK = np.vstack([log_phi_DK.T, log_new_phi.T])
            else:
                log_phi_DK = np.vstack([log_phi_DK, log_new_phi.T])
            log_phi_VK = logsubexp(log_new_phi.T, log_term) #all combinations of d minus 
            log_phi_VDK[:, d, :] = log_phi_VK
        self.phi_DK = log_phi_DK
        self.phi_VDK = log_phi_VDK
    def update_params(self):
        self.Theta_IK = self.Theta_IC @ self.w_KC.T 
        self.phi_DK = np.sum(self.Theta_IK, axis=0, keepdims=True).T 
        phi_VK = self.phi_DK.T - self.Theta_IK 
        self.phi_VDK = np.zeros((self.V, self.D, self.K)) 
        self.phi_VDK[:, 0, :] = phi_VK 
        for d in range(1, self.D): 
            new_phi_DK_row = (np.sum(phi_VK * self.Theta_IK, axis=0,keepdims=True).T) / (d+1) 
            if d == 1: 
                self.phi_DK = np.vstack([self.phi_DK.T, new_phi_DK_row.T]) 
            else: 
                self.phi_DK = np.vstack([self.phi_DK[:d, :], new_phi_DK_row.T]) 
            phi_VK = np.maximum(new_phi_DK_row.T - phi_VK * self.Theta_IK, 0) 
            self.phi_VDK[:, d, :] = phi_VK
            self.phi_DK = np.log(np.maximum(self.phi_DK, 1e-10))
            self.phi_VDK = np.log(np.maximum(self.phi_VDK, 1e-10))

    
    def allocate_iCK(self, Theta_IK, count_K, index, w_KC_d, l_w_kc_d, l_w_kc):
        D = len(index)
        y_dKC = np.zeros((D, self.K, self.C))
        y_dKC[:, np.arange(self.C), np.arange(self.C)] = count_K[:self.C]
        if np.sum(count_K[self.C:self.K]) > 1e-10:
            log_theta_IK = np.log(Theta_IK[index, self.C:self.K])
            log_theta_IC = np.log(self.Theta_IC[index, :])
            total_const   = np.sum(log_theta_IK, axis=0)
            total_const_C = np.sum(log_theta_IC, axis=0) 
            const_d   = total_const[None, :] - log_theta_IK  
            const_C_d = total_const_C[None, :] - log_theta_IC
            potentials_C = (l_w_kc[self.C:self.K, None, :]  + log_theta_IC[None, :, :] + logsubexp(const_d.T[:, :, None], l_w_kc_d[self.C:self.K, None, :] + const_C_d[None, :, :])) 
            probs = np.exp(potentials_C - logsumexp(potentials_C, axis=2, keepdims=True))  
            probs = np.maximum(probs, 1e-10)
            probs /= np.sum(probs, axis=2, keepdims=True)
            y_dKC[:, self.C:self.K,:] = (probs.transpose(1, 0, 2) * count_K[None, self.C:self.K, None])
        return y_dKC
    
    def test_stuff(self):
        directory = self.directory
        model = self.model_type
        if model == "omni":
            log_pmf, auc = self.make_predictions_d()
        elif model == "semi":
            log_pmf, auc = self.make_predictions()
        else:
            print("Invalid model")
            return

        self.heldout_llk_D = np.vstack([self.heldout_llk_D, log_pmf.T])
        times_np = np.array(self.times, dtype=float)
        heldout_llk_D_np = np.array(self.heldout_llk_D, dtype=float)
        os.makedirs(f"results/{directory}", exist_ok=True)

        if model == "omni":
            np.savez(f"results/{directory}/C{self.C}K{self.K}seed{self.seed}RR_full.npz", times=times_np, llks=heldout_llk_D_np)
        elif model == "semi":
            np.savez(f"results/{directory}/C{self.C}K{self.K}seed{self.seed}RR.npz", times=times_np, llks=heldout_llk_D_np)
        else:
            print("Invalid model")

    
    def make_predictions(self):
        Y_indices_test_D = self.Y_indices_test_D
        Y_counts_test_D = self.Y_counts_test_D
        D = len(self.Y_indices_test_D)
        start = self.min_order
        predictions = np.array([])
        log_pmf = np.array([])
        prob_greater_0 = np.array([])
        first_half = np.array([])
        second_half = np.array([])
        log_pmfs = np.zeros(D - start + 1)
        aucs = np.zeros(D - start + 1)

        for d in range(self.min_order - 1, D):
            Y_indices = Y_indices_test_D[d]
            Y_counts = Y_counts_test_D[d]
            K = self.K
            rates_IK = np.log(self.gammas_DK[d, :] + 1e-300)[None, :] * np.ones((Y_indices.shape[0], K))

            for m in range(d):
                rates_IK += np.log(self.Theta_IK[Y_indices[:, m], :] + 1e-300)

            rates_IK[np.isnan(rates_IK)] = -np.inf

            predictions_d = np.sum(np.exp(rates_IK), axis=1)
            prob_greater_0_d = 1.0 - np.exp(-predictions_d)

            mid = int(np.round(len(prob_greater_0_d) / 2))
            first_half_d = predictions_d[:mid]
            second_half_d = predictions_d[mid:]

            log_poisson = poisson.logpmf(Y_counts, predictions_d + 1e-300)

            predictions = np.concatenate([predictions, predictions_d])
            log_pmfs[d - start + 1] = np.mean(log_poisson)
            aucs[d - start + 1] = (np.sum(first_half_d > second_half_d) + 0.5 * np.sum(first_half_d == second_half_d)) / len(first_half_d)
            log_pmf = np.concatenate([log_pmf, log_poisson])
            prob_greater_0 = np.concatenate([prob_greater_0, prob_greater_0_d])
            first_half = np.concatenate([first_half, first_half_d])
            second_half = np.concatenate([second_half, second_half_d])
        return log_pmfs, aucs

    def make_predictions_d(self):
        Y_indices_test_D = self.Y_indices_test_D
        Y_counts_test_D = self.Y_counts_test_D
        K = self.K
        C = self.C
        start = self.min_order
        D = len(Y_indices_test_D)

        predictions = np.array([])
        log_pmf = np.array([])
        prob_greater_0 = np.array([])
        first_half = np.array([])
        second_half = np.array([])

        log_pmfs = np.zeros(D - start + 1)
        aucs = np.zeros(D - start + 1)

        for d in range(start - 1, D):
            Y_indices = Y_indices_test_D[d]
            Y_counts = Y_counts_test_D[d]
            rates_IK = np.zeros((Y_indices.shape[0], K))
            if len(Y_counts) > 0:
                for m in range(d):
                    rates_IK += np.log(self.Theta_IK[Y_indices[:, m], :] + 1e-300)

                rates_IK[np.isnan(rates_IK)] = -np.inf
                rates_IK = np.exp(rates_IK)
                w_KC_d = self.w_KC ** d

                for k in range(C, K):
                    diff = np.sum((w_KC_d[k, :C])[None, :] * rates_IK[:, :C], axis=1)
                    rates_IK[:, k] -= diff

                rates_IK *= self.gammas_DK[d, :][None, :]
                predictions_d = np.sum(rates_IK, axis=1)
                predictions_d = np.maximum(predictions_d, 0)
                prob_greater_0_d = 1.0 - np.exp(-predictions_d)

                mid = int(np.round(len(prob_greater_0_d) / 2))
                first_half_d = predictions_d[:mid]
                second_half_d = predictions_d[mid:]

                log_poisson = poisson.logpmf(Y_counts, predictions_d + 1e-300)

                predictions = np.concatenate([predictions, predictions_d])
                log_pmfs[d - start + 1] = np.mean(log_poisson)
                log_pmf = np.concatenate([log_pmf, log_poisson])
                prob_greater_0 = np.concatenate([prob_greater_0, prob_greater_0_d])
                first_half = np.concatenate([first_half, first_half_d])
                second_half = np.concatenate([second_half, second_half_d])
                aucs[d - start + 1] = np.sum(first_half_d > second_half_d) / len(first_half_d)

        return log_pmfs, aucs
    
    def compute_llk_full(self):
        llk = 0
        start = self.min_order
        D = self.D
        K = self.K
        C = self.C
        llk -= np.sum(np.exp(self.phi_DK[start-1:self.D, :]) * self.gammas_DK[start-1:self.D, :])
        for d in range(start-1, D):
            term = self.gammas_DK[d, self.C:self.K] * np.sum((self.w_KC[self.C:self.K, :] ** (d+1)) * np.exp(self.phi_DK[d, :self.C]), axis=1)
            llk += np.sum(term)

        for d in range(start-1, self.D):
            Y_counts = self.Y_counts_D[d]
            Y_indices = self.Y_indices_D[d]
            rates_IK = np.zeros((Y_indices.shape[0], K))
            for m in range(Y_indices.shape[1]):
                rates_IK += np.log(self.Theta_IK[Y_indices[:, m], :] + 1e-300)
            rates_IK = np.exp(rates_IK)

            for k in range(C, K):
                adjustment = np.sum(rates_IK[:, :C] * (self.w_KC[k, :C] ** (d+1)), axis=1)
                rates_IK[:, k] -= adjustment
            rates_IK = np.maximum(rates_IK, 0)

            rates_IK = np.log(rates_IK + 1e-80) + np.log(self.gammas_DK[d, :] + 1e-300)
            llk += np.sum(logsumexp(rates_IK, axis=1) * Y_counts)
        self.change_elbo = np.abs(llk - self.old_elbo)
        self.old_elbo = llk
        self.ELBOs.append(llk)

    def compute_llk(self):
        llk = 0
        D = self.D
        K = self.K
        start = self.min_order
        llk -= np.sum(np.exp(self.phi_DK[start-1:D, :]) * self.gammas_DK[start-1:self.D, :])
        for d in range(start-1, self.D):
            Y_counts = self.Y_counts_D[d]
            Y_indices = self.Y_indices_D[d]
            rates_IK = np.zeros((Y_indices.shape[0], K)) + np.log(self.gammas_DK[d, :] + 1e-300)
            for m in range(Y_indices.shape[1]):
                rates_IK += np.log(self.Theta_IK[Y_indices[:, m], :] + 1e-300)
            llk += np.sum(logsumexp(rates_IK, axis=1) * Y_counts)
        self.change_elbo = np.abs(llk - self.old_elbo)
        self.old_elbo = llk
        self.ELBOs.append(llk)
