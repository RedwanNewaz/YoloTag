from typing import Any
import numpy as np 

class ButterWorth:
    def __init__(self, a_coeff, b_coeff, order) -> None:
        self.x = [0] * (order + 1)
        self.y = [0] * (order + 1)
        self.a_coeff = a_coeff
        self.b_coeff = b_coeff
        self.order = order
        self.initialized = False

    def __call__(self, z) -> Any:
        self.x[0] = z

        if not self.initialized:
            for i in range(self.order + 1):
                self.x[i] = self.y[i] = z

            self.initialized = True

        ay = 0
        for i, a in enumerate(self.a_coeff):
            ay += a * self.y[i+1]
        
        bx = 0 
        for j, b in enumerate(self.b_coeff):
            bx +=  b * self.x[j]
        
        self.y[0] = ay + bx 
        for i in range(self.order - 1, -1, -1):
            self.x[i+1] = self.x[i] # store xi
            self.y[i+1] = self.y[i] # store yi

        return self.y[0]



class LowpassFilter:
    def __init__(self, a_coeff, b_coeff, order) -> None:
        self.a_coeff = a_coeff
        self.b_coeff = b_coeff
        self.x_dim = None
        self.filters = [] 
        self.order = order
    
    def __call__(self, z) -> Any:
        if self.x_dim is None:
            self.x_dim = len(z)
            for i in range(self.x_dim):
                f = ButterWorth(self.a_coeff, self.b_coeff, self.order)
                self.filters.append(f)
        
        if len(self.filters) < 1:
            return 

        x_hat = [self.filters[i](z_raw) for i, z_raw in enumerate(z)]        

        return np.array(x_hat)


# class LowpassFilter:
#     def __init__(self, alpha, outlier_dist) -> None:
#         self.alpha = alpha 
#         self.x = None 
#         self.count = 10
#         self.outlier_dist = outlier_dist
    
#     def __call__(self, z) -> Any:
#         if self.x is None:
#             self.buffer = np.zeros((0, len(z)))
        
#         if self.x is None or self.count > 0:
#             z = np.reshape(z, (1, len(z)))
#             self.buffer = np.vstack((self.buffer, z))
#             self.x = self.buffer[-1]

#             self.count -= 1
#         else:
#             dist = np.linalg.norm(self.x - z)
            
#             if dist < self.outlier_dist:
#                 self.x = self.alpha * self.x + (1-self.alpha) * z
#             else:
#                 print(dist)

#         return self.x