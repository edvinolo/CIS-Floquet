import numpy as np

#Need to set environment varialbe to use propack
import os
#os.environ["SCIPY_USE_PROPACK"] = "true"
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spl


class Floquet_system:
    def __init__(self,H,Z,omega,E_0,N_blocks_abs,N_blocks_em,shift = 0.0):
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

        self.m_omega = np.array([m*self.omega for m in range(-self.N_blocks_abs,self.N_blocks_em+1)])

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


        self.H_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.matvec,rmatvec=self.rmatvec)
        #eigs = spl.eigs(self.H_linop,k=6,return_eigenvectors=False)
        #self.weight = 1.0/eigs[np.argmax(np.abs(eigs))]
        #self.H_0_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.H_0_matvec,rmatvec=self.H_0_rmatvec)
        self.block_solve_setup()
        self.H_invop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.block_solve)

        



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


    def svd(self):
        print('')
        print('Doing svd ...')
        self.u,self.s,self.vh = spl.svds(self.H_linop,which = 'SM',solver = 'propack',maxiter=8)
        self.n_sing_vals = self.s.shape[0]
        print('svd done!')
        print('')

        return
    
    def svd_precond(self,b):
        result = np.zeros(self.N_floquet,dtype=np.complex128)
        for i in range(self.n_sing_vals):
            result += 1.0/self.s[i]*np.dot(self.u[:,i],b)*self.vh[i,:] 

        return result
    
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
    
    def gmres_precond(self,b,max_iter=5,rtol=1e-5):
        print(sl.norm(b))
        result,info = spl.gmres(self.H_linop,b,maxiter=max_iter,tol=rtol)

        return result

    
    def bicg(self,b,max_iter=10,rtol=1e-5):
        RHS_nonzero = True
        if sl.norm(b) == 0.0:
            RHS_nonzero = False

        result,info = spl.bicg(self.H_0_linop,b,tol=rtol,maxiter=max_iter)
        diff = self.H_0_linop@result-b
        
        if RHS_nonzero:
            print(f'Relative forward error: {sl.norm(diff)/sl.norm(b)}, num_iter: {info}')

        return result
    
    def lgmres(self,A,b):
        result,info =spl.lgmres(A,b,tol=1e-13,maxiter=300)
        if info != 0:
                    RuntimeError(f'LGMRES exited with: {info}')

        return result
    
    def solve(self,b,M=None):
        sigma = 0.0
        print('solving')
        if M is None:
            #jacobi_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.jacobi)
            #gs_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.gauss_seidel)
            SOR_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.SOR)
            #bicg_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.bicg)
            #neumann_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.neumann)
            #gmres_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.gmres_precond)
            #weight_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.matvec_weight)
            #self.svd()
            #svd_linop = spl.LinearOperator((self.N_floquet,self.N_floquet),matvec = self.svd_precond)
            result,info = spl.gmres(self.H_linop,b,M=SOR_linop,maxiter=100,tol=1e-10,restart = 50)
            #result,info = spl.lgmres(self.H_linop,b,maxiter=40,tol=1e-10)
            #result,info = spl.bicgstab(self.H_linop,b,tol=1e-10,maxiter=100)

            #result,info = spl.gmres(self.H_linop,b,maxiter=300,tol=1e-10,restart=50)
            #result /=self.weight  

            #result = jacobi_linop@b
        else:
            result,info = spl.gmres(self.H_linop,b,M=M,maxiter=40)
        print(f'GMRES info: {info}')
        print(f'{sl.norm(self.H_linop@result-b)}')
        

        return result
    
    def block_solve_setup(self):
        print('')
        print('Doing LU factorization of block matrix...')
        self.Blocks = self.N_floquet_blocks*[sp.lil_matrix((self.N_elements,self.N_elements),dtype = np.complex128)]
        self.factors = []
        print(len(self.Blocks))

        identity = sp.identity(self.N_elements,dtype =np.complex128, format='csc')

        self.Blocks[0] = self.H + (self.m_omega[0]-self.shift)*identity
        self.Blocks[0].tocsc()
        self.factors.append(spl.splu(self.Blocks[0]))
        for i in range(self.N_floquet_blocks-1):
            print(f'Doing block {i+1}/{self.N_floquet_blocks}')
            for j in range(self.N_elements):
                #w = self.lgmres(self.Blocks[i],self.Z[:,j])
                w = self.factors[i].solve(self.V[:,[j]].toarray())
                self.Blocks[i+1][:,[j]] = self.H.getcol(j)-self.V@w
            self.Blocks[i+1] += (self.m_omega[i+1]-self.shift)*identity
            self.Blocks[i+1] = sp.csc_matrix(self.Blocks[i+1])
            self.factors.append(spl.splu(self.Blocks[i+1]))

        #self.Blocks = [sp.csr_array(Block) for Block in self.Blocks]

        #print(f'# of nonzero in Blocks: {self.Blocks.nnz}, out of {self.N_elements**2*self.N_blocks}')
        print('Done with LU factorization!')
        print('')

        return
    
    #def check_matrix(self):

    
    def block_solve(self,b):
        
        #print('')
        #print('Solving with factorized block matrix...')
        
        #y = sp.lil_array((self.N_elements,self.N_blocks),dtype = np.complex128)
        y = np.zeros((self.N_elements,self.N_floquet_blocks),dtype = np.complex128)
        #y[:,0] = self.lgmres(self.Blocks[0],b[:self.N_elements])
        y[:,0] = self.factors[0].solve(b[:self.N_elements])

        for i in range(1,self.N_floquet_blocks):
            #y[:,i] = self.lgmres(self.Blocks[i],b[i*self.N_elements:(i+1)*self.N_elements]-self.Z@b[(i-1)*self.N_elements:i*self.N_elements])
            y[:,i] = self.factors[i].solve(b[i*self.N_elements:(i+1)*self.N_elements]-self.V@y[:,i-1])

        result = np.zeros(self.N_floquet,dtype=np.complex128)

        result[(self.N_floquet_blocks-1)*self.N_elements:] = y[:,self.N_floquet_blocks-1]

        for i in reversed(range(self.N_floquet_blocks-1)):
            #w = self.lgmres(self.Blocks[i],self.Z@result[(i+1)*self.N_elements:(i+2)*self.N_elements])
            w = self.factors[i].solve(self.V@result[(i+1)*self.N_elements:(i+2)*self.N_elements])
            result[i*self.N_elements:(i+1)*self.N_elements] = y[:,i]-w 

        #print('Done!')
        #print('')

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

    