import numpy as np
import scipy.io as sio

import torch
from torch.autograd import Variable
eps = 1e-12

class MinimaxEntropyEstimator:
    
    def __init__(self, poly_coeff_path, n, gpu=False, base=np.e):
        poly_entro = sio.loadmat(poly_coeff_path)['poly_entro']
        self._poly_entro = {
            d : poly_entro[d][0].flatten()
            for d in range(poly_entro.shape[0])
        }
        self._gpu = gpu
        self._denom = np.log(base)
        self._n = n
        self._th = np.log(n)/n
        self._order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)

    def _f(self, p):
        return - p * torch.log(p + eps) / self._denom

    def _f2(self, p):
        return - 1. / ((p + eps) * self._denom)

    def entro(self, dist):
        H = Variable(torch.zeros(1)).double()
        for p in dist:
            H += self._f(p)
        return H
    
    def _g(self, p, q):
        if (q.data == 1).all():
            return - torch.log(p + eps) / self._denom
        return - torch.log(1 - p + eps) / self._denom
    
    def cross_entro_loss(self, dist_p, dist_q):
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for i in range(len(dist_p)):
            H += self._g(dist_p[i], dist_q[i])
        return H
    
    def minimax_cross_entro(self, dist_p, dist_q):
        n = self._n
        th = self._th
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
                H_i = self._smooth(p)
            H += dist_q[i] / p * H_i
        return H
    
    def minimax_cross_entro_loss(self, dist_p, dist_q):
        n = self._n
        th = self._th
        order = self._order
        
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for i in range(len(dist_p)):
            p = dist_p[i]
            q = dist_q[i]
            
            if (q.data == 1).all():
                op = p
            else:
                op = 1 - p
            
            if (op.data < th).all():
                H_i = self._non_smooth(op, order)
            else:
                H_i = self._smooth(op)

            H += H_i / (op + eps)
        return H

    def _non_smooth(self, p, order):
        d = order - 1
        H = Variable(torch.zeros(1)).double()
        if self._gpu:
            H = H.cuda()
        for m, c in enumerate(self._poly_entro[d]):
            H += (c * torch.pow(p, m)).flatten()[0]
        return H / self._denom

    def _smooth(self, p):
        n = self._n
        H = self._f(p) - self._f2(p) * p * (1 - p) / (2 * n)          
        if self._gpu:
            H = H.cuda()
        return H
        
    def minimax_entro(self, dist):
        n = self._n
        th = np.log(n)/n
        order = min(4 + int(np.ceil(1.2 * np.log(n))), 22)
            
        H = Variable(torch.zeros(1)).double()
        for p in dist:
            if (p.data < th).all():
                H += self._non_smooth(p, order)
            else:
                H += self._smooth(p)
        return H  