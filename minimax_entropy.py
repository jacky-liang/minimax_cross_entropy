import numpy as np
import scipy.io as sio

import torch
from torch.autograd import Variable

class MinimaxEntropyEstimator:
    
    def __init__(self, poly_coeff_path):
        poly_entro = sio.loadmat(poly_coeff_path)['poly_entro']
        self._poly_entro = {
            d : poly_entro[d][0].flatten()
            for d in range(poly_entro.shape[0])
        }        

    @staticmethod
    def _f(p):
        if (p.data == 0).all():
            return 0
        return - p * torch.log(p) / np.log(2)

    @staticmethod
    def _f2(p):
        return - 1. / (p * np.log(2))

    def entro(self, dist):
        H = Variable(torch.zeros(1)).double()
        for p in dist:
            H += self._f(p)
        return H
    
    @staticmethod
    def _g(p, q):
        if (p.data == 0).all():
            return 0
        return - q * torch.log(p) / np.log(2)
    
    def cross_entro(self, dist_p, dist_q):
        H = Variable(torch.zeros(1)).double()
        for i in range(len(dist_p)):
            H += self._g(dist_p[i], dist_q[i])
        return H
    
    def minimax_cross_entro(self, dist_p, dist_q, n):
        th = np.log(n)/n
        order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
        
        H = Variable(torch.zeros(1)).double()        
        for i in range(len(dist_p)):
            p = dist_p[i]
            if (p.data == 0).all():
                continue
            if (p.data < th).all():
                H_i = self._non_smooth(p, order)
            else:
                H_i = self._smooth(p, n)
            H += dist_q[i] /1./ p * H_i
        return H

    def _non_smooth(self, p, order):
        d = order - 1
        H = Variable(torch.zeros(1)).double()
        for m, c in enumerate(self._poly_entro[d]):
            H += (c * torch.pow(p, m)).flatten()[0]
        H /= np.log(2)
        return H        

    def _smooth(self, p, n):
        return self._f(p) - self._f2(p) * p * (1 - p) / (2 * n)          
        
    def minimax_entro(self, dist, n):
        th = np.log(n)/n
        order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
            
        H = Variable(torch.zeros(1)).double()
        for p in dist:
            if (p.data < th).all():
                H += self._non_smooth(p, order)
            else:
                H += self._smooth(p, n)
        return H  