import numpy as np
import scipy.sparse as sp
import time

from mod_PARDISO import mod_pardiso

class PARDISO_wrapper:
    #A class that wraps a fortran module for using MKL_PARDISO
    def __init__(self,H_fl):
        self.H_fl = H_fl

        print('')
        print('Setting up PARDISO and factorizing')
        t_1 = time.perf_counter()
        mod_pardiso.setup(self.H_fl.data,self.H_fl.indptr,self.H_fl.indices)
        t_2 = time.perf_counter()
        print('Done!')
        print(f'Time for PARDISO setup and factorization: {t_2-t_1} s')
        print('')

        return

    def solve(self,b):
        x = np.zeros(self.H_fl.shape[0],dtype = np.complex128)

        mod_pardiso.solve(self.H_fl.data,self.H_fl.indptr,self.H_fl.indices,x,b)

        return x

    def cleanup(self):
        mod_pardiso.cleanup()