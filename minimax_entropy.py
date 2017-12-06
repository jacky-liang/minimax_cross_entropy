import numpy as np
import scipy.io as sio

import torch
from torch.autograd import Variable

class MinimaxEntropyEstimator:
    
    def __init__(self, poly_coeff_path, gpu=False, base=np.e):
        poly_entro = sio.loadmat(poly_coeff_path)['poly_entro']
        self._poly_entro = {
            d : poly_entro[d][0].flatten()
            for d in range(poly_entro.shape[0])
        }
        self._gpu = gpu
        self._denom = np.log(base)

    def _f(self, p):
        if (p.data == 0).all():
            return 0
        return - p * torch.log(p) / self._denom

    def _f2(self, p):
        return - 1. / (p * self._denom)

    def entro(self, dist):
        H = Variable(torch.zeros(1)).double()
        for p in dist:
            H += self._f(p)
        return H
    
    def _g(self, p, q):
        if (p.data == 0).all():
            return 0
        if (q.data == 1).all():
            return - torch.log(p) / self._denom
        return - torch.log(1 - p) / self._denom
    
    def cross_entro_loss(self, dist_p, dist_q):
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for i in range(len(dist_p)):
            H += self._g(dist_p[i], dist_q[i])
        return H
    
    def minimax_cross_entro(self, dist_p, dist_q, n):
        th = np.log(n)/n
        order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
        
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for i in range(len(dist_p)):
            p = dist_p[i]
            if (p.data == 0).all():
                continue
            if (p.data < th).all():
                H_i = self._non_smooth(p, order)
            else:
                H_i = self._smooth(p, n)
            H += dist_q[i] / p * H_i
        return H
    
    def minimax_cross_entro_loss(self, dist_p, dist_q, n=10):
        th = np.log(n)/n
        order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
        
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for i in range(len(dist_p)):
            p = dist_p[i]
            if (p.data == 0).all():
                continue
            if (p.data < th).all():
                H_i = self._smooth(p, order)
            else:
                H_i = self._smooth(p, n)
            H += dist_q[i] / p * H_i
        return H

    def _non_smooth(self, p, order):
        d = order - 1
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for m, c in enumerate(self._poly_entro[d]):
            H += (c * torch.pow(p, m)).flatten()[0]
        return H / self._denom

    def _smooth(self, p, n):
        H = self._f(p) - self._f2(p) * p * (1 - p) / (2 * n)          
        if self._gpu:
            H = H.cuda()
        return H
        
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