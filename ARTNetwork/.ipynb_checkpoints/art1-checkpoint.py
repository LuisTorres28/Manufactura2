import numpy as np
class ART(object):
    def __init__(self, n, m, rho=0.7, L=5):
        self.rho = rho
        self.L = L
        # initialize weights
        self.z_ij = np.ones((n, m)) * (L / (L - 1 + n))
        self.z_ji = np.ones((m, n))
        # counters
        self.epochs = 0
    
    def read_input(self, I):
        # choice functions
        T_j = self.z_ij.T @ I # (m, n) x (n,1)
        # winner take all
        while True:
            J = np.argmax(T_j)
            y_j = np.zeros_like(T_j) # (m, 1)
            y_j[J] = 1
            # F1 inputs from F2
            V_i = self.z_ji.T @ y_j # (n, m) x (m, 1) = (n, 1)
            # z_J = self.z_ji[J]
            # vigilance subsystem
            M_J = (V_i.T @ I)[0][0] / np.sum(I)
            if self.rho == M_J or self.rho < M_J:
                # update weights
                self.z_ji[J] = np.squeeze(self.z_ji[J][:, None] * I) # (n, 1) * (n, 1) - > (n,)
                self.z_ij[:, J] = (self.z_ji[J] * self.L) / (self.L - 1 + np.sum(self.z_ji[J]))
                self.epochs += 1
                break
            else:
                T_j[J] = 0 # deactivate category
                if np.all(T_j == 0):
                    # no more categories
                    break
        return np.squeeze(y_j)
