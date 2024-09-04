import numpy as np
import time
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from block_LU import block_lu


class Floquet_system:
    def __init__(self,H,Z,omega,E_0,N_blocks_abs,N_blocks_em,shift = 0.0, fortran = False):
        self.H = H.copy()
        self.V = 0.5*E_0*Z.copy()
        self.omega = omega
        self.E_0 = E_0
        self.N_blocks_abs = N_blocks_abs
        self.N_blocks_em = N_blocks_em
        self.shift = shift

        self.N_elements = self.H.shape[0]

        self.N_floquet_blocks = self.N_blocks_abs + self.N_blocks_em +1
        self.N_floquet = self.N_elements*self.N_floquet_blocks

        self.m = list(range(-self.N_blocks_abs,self.N_blocks_em+1))
        self.m_omega = np.array([m*self.omega for m in self.m])

        self.diag = np.zeros(self.N_floquet,dtype = np.complex128)
        self.diag_inv = np.zeros(self.N_floquet,dtype = np.complex128)
        for i in range(self.N_floquet_blocks):
            self.diag[i*self.N_elements:(i+1)*self.N_elements] = np.diag(self.H) + self.m_omega[i]

        for i in range(self.N_floquet):
            if np.abs(self.diag[i]) >= 1e-13:
                self.diag_inv[i] = 1.0/self.diag[i]

        self.M_inv = sp.diags(self.diag_inv)
        self.M = sp.diags(self.diag)

        self.H = sp.csc_matrix(self.H)
        self.V = sp.csc_matrix(self.V)

        self.H_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.matvec,rmatvec=self.rmatvec,dtype = np.complex128)

        if fortran:
            self.V_dense = self.V.toarray()
            self.block_solve_setup_fortran()
            self.H_invop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.block_solve_fortran,dtype = np.complex128)

        else:
            self.block_solve_setup()
            self.H_invop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.block_solve,dtype = np.complex128)

        #eigs = spl.eigs(self.H_linop,k=6,return_eigenvectors=False)
        #self.weight = 1.0/eigs[np.argmax(np.abs(eigs))]
        #self.H_0_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.H_0_matvec,rmatvec=self.H_0_rmatvec)

        return

    def matvec(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] += self.H@v[i*self.N_elements:(i+1)*self.N_elements] + (self.m_omega[i])*v[i*self.N_elements:(i+1)*self.N_elements]

            if i != self.N_floquet_blocks-1:
                result[i*self.N_elements:(i+1)*self.N_elements] += self.V@v[(i+1)*self.N_elements:(i+2)*self.N_elements]
                result[(i+1)*self.N_elements:(i+2)*self.N_elements] += self.V@v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def matvec_weight(self,v):
        return self.weight*self.matvec(v)

    def matvec_non_diag(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            if i != self.N_floquet_blocks-1:
                result[i*self.N_elements:(i+1)*self.N_elements] += self.V@v[(i+1)*self.N_elements:(i+2)*self.N_elements]
                result[(i+1)*self.N_elements:(i+2)*self.N_elements] += self.V@v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def matvec_D_E(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] += self.H@v[i*self.N_elements:(i+1)*self.N_elements] + self.m_omega[i]*v[i*self.N_elements:(i+1)*self.N_elements]
            if i != self.N_floquet_blocks-1:
                result[(i+1)*self.N_elements:(i+2)*self.N_elements] += self.V@v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def matvec_F(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            if i != self.N_floquet_blocks-1:
                result[i*self.N_elements:(i+1)*self.N_elements] += self.V@v[(i+1)*self.N_elements:(i+2)*self.N_elements]
        return result

    def H_0_matvec(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] += self.H@v[i*self.N_elements:(i+1)*self.N_elements] + (self.m_omega[i]+1e2)*v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def rmatvec(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] += np.conjugate(self.H)@v[i*self.N_elements:(i+1)*self.N_elements] + self.m_omega[i]*v[i*self.N_elements:(i+1)*self.N_elements]

            if i != self.N_floquet_blocks-1:
                result[i*self.N_elements:(i+1)*self.N_elements] += np.conjugate(self.V)@v[(i+1)*self.N_elements:(i+2)*self.N_elements]
                result[(i+1)*self.N_elements:(i+2)*self.N_elements] += np.conjugate(self.V)@v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def H_0_rmatvec(self,v):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        for i in range(self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] += np.conjugate(self.H)@v[i*self.N_elements:(i+1)*self.N_elements] + self.m_omega[i]*v[i*self.N_elements:(i+1)*self.N_elements]

        return result

    def H_matvec_Neu(self,v,weight = 1.0):
        b = weight*self.matvec(v)

        return v - b

    def neumann(self,v,num_iter = 10):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        eigs = spl.eigs(self.H_linop,k=6,return_eigenvectors=False)
        print(eigs)
        weight = 1.0/eigs[np.argmax(np.abs(eigs))]

        temp = v.copy()
        for i in range(num_iter):
            temp = self.H_matvec_Neu(temp,weight)
            result += temp

        diff = v-self.H_linop@(v+result)

        if sl.norm(v)>0:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(v)}, num_iter: {i+1}')

        return v + result

    def jacobi(self,b,max_iter = 200,rtol = 1e-5):

        result = np.zeros(self.N_floquet,dtype = np.complex128)

        eigs = spl.eigs(self.H_linop,k=6,return_eigenvectors=False)
        #print(eigs)
        weight = 1.0/eigs[np.argmax(np.abs(eigs))]

        RHS_nonzero = True
        if sl.norm(b) == 0.0:
            RHS_nonzero = False

        for i in range(max_iter):
            #result = self.M_inv@(b-self.H_linop@result+self.M@result)
            result = result + weight*self.M_inv@(b-self.H_linop@result)
            diff = self.H_linop@result-b

            if RHS_nonzero:
                if sl.norm(diff)/sl.norm(b) < rtol:
                    break
            elif sl.norm(diff) < rtol:
                break 

        if RHS_nonzero:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(b)}, num_iter: {i+1}')

        return result

    def jacobi_block(self,b,max_iter=5,rtol=1e-5):
        result = np.zeros(self.N_floquet,dtype = np.complex128)

        RHS_nonzero = True
        if sl.norm(b) == 0.0:
            RHS_nonzero = False

        for i in range(max_iter):
            result = self.jacobi(b) - self.jacobi(self.matvec_non_diag(result))
            diff = self.H_linop@result-b

            if RHS_nonzero:
                if sl.norm(diff)/sl.norm(b) < rtol:
                    break
            elif sl.norm(diff) < rtol:
                break 

        if RHS_nonzero:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(b)}, num_iter: {i+1}')

        return result

    def gauss_seidel(self,b,max_iter=5,rtol=1e-5):
        x_k = np.zeros(self.N_floquet,dtype=np.complex128)
        x_k_1 = np.zeros(self.N_floquet,dtype=np.complex128)

        RHS_nonzero = True
        if sl.norm(b) == 0.0:
            RHS_nonzero = False

        for i in range(max_iter):
            #x_k = self.H_linop@x_k

            x_k = self.matvec_F(x_k)
            x_k_1  = self.block_gs_solve(b-x_k)


            #x_k_1[0] = self.diag_inv[0]*(b[0]-np.sum(x_k[1:]))
            #for j in range(1,self.N_floquet):
            #    A_x_k_1 = self.H_linop@x_k_1
            #    x_k_1[j] = self.diag_inv[j]*(b[j]-np.sum(x_k[j+1:])-np.sum(A_x_k_1[0:j]))

            x_k = x_k_1.copy()
            diff = self.H_linop@x_k_1-b

            if RHS_nonzero:
                if sl.norm(diff)/sl.norm(b) < rtol:
                    break
            elif sl.norm(diff) < rtol:
                break 

        if RHS_nonzero:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(b)}, num_iter: {i+1}')

        return x_k_1

    def SOR(self,b,max_iter=2,rtol=1e-5,weight = 1.5e0):
        x_k = np.zeros(self.N_floquet,dtype=np.complex128)
        x_k_1 = np.zeros(self.N_floquet,dtype=np.complex128)

        RHS_nonzero = True
        if sl.norm(b) == 0.0:
            RHS_nonzero = False

        for i in range(max_iter):
            #x_k = self.H_linop@x_k

            x_k = -self.weight*self.matvec_F(x_k) + (1-self.weight)*self.H_0_matvec(x_k) 
            x_k_1  = self.block_SOR_solve(self.weight*b+x_k,self.weight)


            #x_k_1[0] = self.diag_inv[0]*(b[0]-np.sum(x_k[1:]))
            #for j in range(1,self.N_floquet):
            #    A_x_k_1 = self.H_linop@x_k_1
            #    x_k_1[j] = self.diag_inv[j]*(b[j]-np.sum(x_k[j+1:])-np.sum(A_x_k_1[0:j]))

            x_k = x_k_1.copy()
            diff = self.H_linop@x_k_1-b

            if RHS_nonzero:
                if sl.norm(diff)/sl.norm(b) < rtol:
                    break
            elif sl.norm(diff) < rtol:
                break 

        if RHS_nonzero:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(b)}, num_iter: {i+1}')

        return x_k_1

    def block_solve_setup(self):
        #Perform tridiagonal Block LU factorization with Python routines

        print('')
        print('Doing LU factorization of block matrix...')
        #self.Blocks = self.N_floquet_blocks*[sp.lil_matrix((self.N_elements,self.N_elements),dtype = np.complex128)]
        self.factors = []
        #Block = np.zeros((self.N_elements,self.N_elements),dtype=np.complex128)
        identity = sp.identity(self.N_elements,dtype =np.complex128, format='csc')

        t_1 = time.perf_counter()
        Block = self.H + (self.m_omega[0]-self.shift)*identity
        self.factors.append(sl.lu_factor(Block.toarray()))
        for i in range(self.N_floquet_blocks-1):
            print(f'Doing block {i+1}/{self.N_floquet_blocks}')
            Block = self.H + (self.m_omega[i+1]-self.shift)*identity - self.V@sl.lu_solve(self.factors[i],self.V.toarray())
            self.factors.append(sl.lu_factor(Block))
        #self.Blocks = [sp.csr_array(Block) for Block in self.Blocks]
        t_2 = time.perf_counter()
        #for block in self.Blocks:
        #    print(f'# of nonzero in Blocks: {block.nnz}, out of {self.N_elements**2}, sparisty: {(self.N_elements**2-block.nnz)/self.N_elements**2}')
        print(f'Done with LU factorization! Wall time: {t_2-t_1} s')
        print('')

        return

    def block_solve_setup_fortran(self):
        #Perform tridiagonal Block LU factorization with fortran routines

        print('')
        print('Doing LU factorization of block matrix...')
        t_1 = time.perf_counter()
        block_lu.block_lu_setup(self.H.toarray(),self.V_dense,self.shift,self.m_omega,self.N_floquet_blocks,self.N_elements)
        t_2 = time.perf_counter()
        print(f'Done with LU factorization! Wall time: {t_2-t_1} s')
        print('')

    def block_solve(self,b):
        #Solve LU facgored Block system using python routines

        #print('')
        #print('Solving with factorized block matrix...')

        y = np.zeros((self.N_elements,self.N_floquet_blocks),dtype = np.complex128)
        y[:,0] = sl.lu_solve(self.factors[0],b[:self.N_elements])

        for i in range(1,self.N_floquet_blocks):
            y[:,i] = sl.lu_solve(self.factors[i],b[i*self.N_elements:(i+1)*self.N_elements]-self.V@y[:,i-1])

        result = np.zeros(self.N_floquet,dtype=np.complex128)

        result[(self.N_floquet_blocks-1)*self.N_elements:] = y[:,self.N_floquet_blocks-1]

        for i in reversed(range(self.N_floquet_blocks-1)):
            w = sl.lu_solve(self.factors[i],self.V@result[(i+1)*self.N_elements:(i+2)*self.N_elements])
            result[i*self.N_elements:(i+1)*self.N_elements] = y[:,i]-w 

        #print('Done!')
        #print('')

        return result

    def block_solve_fortran(self,b):
        #Solve LU factored Block system using fortran routines

        result = np.zeros(self.N_floquet,dtype=np.complex128)
        #Better to keep a dense copy of V than to do toarray on every call to the solve routine.
        block_lu.block_lu_solve(b,result,self.V_dense)

        return result

    def block_gs_solve(self,b):
        result = np.zeros(b.shape,dtype = np.complex128)
        id = sp.identity(self.N_elements)

        result[:self.N_elements] = spl.spsolve(self.H+self.m_omega[0]*id,b[:self.N_elements])
        for i in range(1,self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] = self.lgmres(self.H+self.m_omega[i]*id,b[i*self.N_elements:(i+1)*self.N_elements]-self.V@result[(i-1)*self.N_elements:i*self.N_elements])

        return result

    def block_SOR_solve(self,b,weight):
        result = np.zeros(b.shape,dtype = np.complex128)
        id = sp.identity(self.N_elements)

        result[:self.N_elements] = spl.spsolve(self.H+(self.m_omega[0])*id,b[:self.N_elements])
        for i in range(1,self.N_floquet_blocks):
            result[i*self.N_elements:(i+1)*self.N_elements] = self.lgmres(self.H+(self.m_omega[i])*id,b[i*self.N_elements:(i+1)*self.N_elements]-weight*self.V@result[(i-1)*self.N_elements:i*self.N_elements])

        return result