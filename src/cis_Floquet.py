import os 
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from class_Floquet import Floquet_system

sys.path.append('./Effective-Ham/src/')
sys.path.append('./format-plot/')
from plot_utils import format_plot
from effective_Hamiltonian import complex_Rabi


def Floquet(omega,E_0,H,z,N_blocks_up,N_blocks_down,**kwargs):
    energy = kwargs.get('energy',0.0)
    plot = kwargs.get('plot',False)
    tol = kwargs.get('tol',2e-1)
    I_p = kwargs.get('I_p',None)
    N_eigenvalues = kwargs.get('N_eig',6)
    sort_type = kwargs.get('sort_type','real')
    prev_vec = kwargs.get('prev_vec',None)
    fortran = kwargs.get('fortran',False)

    N_elements = H.shape[0]
    I_omega = omega*np.eye(N_elements,dtype = np.complex128)

    m = range(-N_blocks_up,N_blocks_down+1)
    N_blocks = len(m)    

    N_floquet = N_elements*N_blocks

    print('')
    print('Diagonalizing floquet Hamiltonian')
    print(f'shift: {energy}')
    print(f'omega: {omega}')
    print(f'E_0: {E_0}')
    U_p = E_0**2/(4*omega**2)
    alpha_0 = E_0/omega**2
    print(f'U_p: {U_p}')
    print(f'U_p/omega: {U_p/omega}')
    if I_p:
        N_photon = (I_p+U_p)/omega
        print(f'# of photons to ionize: {N_photon}') 


    print(f'Quiver radius: {alpha_0}')
    print(f'N_blocks: {N_blocks}, Abs: {N_blocks_up}, Em: {N_blocks_down}')
    print('')

    H_test = sp.lil_matrix(H)
    print(f'# of nonzero in H: {H_test.nnz}, out of {N_elements**2}')

    #This is for setting up the full Floquet Matrix explicitly, should not be needed
    #H_fl = sp.lil_matrix((N_floquet,N_floquet),dtype = np.complex128)
    """ for i in range(N_blocks):
        H_fl[i*N_elements:(i+1)*N_elements,i*N_elements:(i+1)*N_elements] = H + m[i]*I_omega


        if i !=N_blocks-1:
            H_fl[i*N_elements:(i+1)*N_elements,(i+1)*N_elements:(i+2)*N_elements] = z*E_0/2 
            H_fl[(i+1)*N_elements:(i+2)*N_elements,i*N_elements:(i+1)*N_elements] = z*E_0/2 

    H_fl = sp.csr_matrix(H_fl)#,blocksize = (N_elements,N_elements))
    print(f'# of nonzero in H_fl: {H_fl.nnz}, out of {N_floquet**2}') """

    H_fl_sys = Floquet_system(H,z,omega,E_0,N_blocks_up,N_blocks_down,shift = energy,fortran = fortran)

    t1 = time.perf_counter()
    eigs,vecs = spl.eigs(H_fl_sys.H_linop,k=N_eigenvalues,maxiter=500,sigma=energy,OPinv = H_fl_sys.H_invop)
    t2 = time.perf_counter()

    print('')
    print('----------------------------------------')    
    print('Eigenvalues found by Arnoldi iteration: ')
    print(eigs)
    print('----------------------------------------')
    print(f'Time for Arnoldi: {t2-t1} s')
    print('')

    if sort_type == 'real':
        indices = np.argsort(np.real(eigs))
    elif sort_type == 'abs':
        indices = np.argsort(np.abs(eigs-energy))
    elif sort_type == 'vec':
        proj = np.zeros(N_eigenvalues)
        for i in range(N_eigenvalues):
            proj[i] = np.abs(np.dot(prev_vec,vecs[:,i]))
        print(f'Projections: {proj}')
        indices = np.argsort(proj)
        indices = np.flip(indices) #Need to reverse the indices so that largest projection comes first


    eigs = eigs[indices]
    vecs = vecs[:,indices]

    print('')
    print('----------------------------------------')    
    print(f'Eigenvalues sorted by {sort_type}: ')
    print(eigs)
    print('----------------------------------------')
    print('')

    max_block = np.zeros(N_eigenvalues,dtype = np.int64)
    block = 0
    for i in range(N_eigenvalues):
        block_proj_max = -1
        for j in range(H_fl_sys.N_floquet_blocks):
            block_proj = sl.norm(vecs[H_fl_sys.N_elements*j:H_fl_sys.N_elements*(j+1),i])
            if block_proj > block_proj_max:
                block_proj_max = block_proj
                block = j
        max_block[i] = H_fl_sys.m[block]

    #print(vecs[N_blocks*N_elements:N_blocks*N_elements+2,0])
    #print(vecs[N_blocks*N_elements:N_blocks*N_elements+2,1])

    """ eigs_H, vecs_H = spl.eigs(H,sigma = -0.1)
    for k in range(vecs_H.shape[1]):
        index = np.argmax(np.abs(vecs_H[:,k]))
        print(vecs_H[index,k])
        vecs_H[:,k] *=  np.exp(-1j*np.angle(vecs_H[index,k]))

    a = vecs_H[:,0]
    b = vecs_H[:,2]

    dipole = np.dot(b,np.matmul(z,a))
    b *= np.exp(-1j*np.angle(dipole))

    print(f'Dipole: {np.dot(b,np.matmul(z,a))}')

    a_N = np.zeros(N_floquet,dtype = np.complex128)
    a_N[N_blocks_up*N_elements:(N_blocks_up+1)*N_elements] = a.copy()

    b_N_1 = np.zeros(N_floquet,dtype = np.complex128)
    b_N_1[(N_blocks_up-1)*N_elements:N_blocks_up*N_elements] = b.copy()

    indices = []
    for k in range(eigs.shape[0]):
        #if  np.imag(eigs[k])<0  :
        #    indices.append(k)
        print(f'Projection on |a,N>: {np.abs(np.dot(a_N,vecs[:,k]))}')
        print(f'Projection on |b,N-1>: {np.abs(np.dot(b_N_1,vecs[:,k]))}')
        if (np.abs(np.dot(a_N,vecs[:,k])) >tol or np.abs(np.dot(b_N_1,vecs[:,k])) >tol) and np.imag(eigs[k])<0  :
            indices.append(k)

    eigs = eigs[indices]
    print(f'Kept eigenvalues: {eigs}')
    vecs = vecs[:,indices]

    indices = np.argsort(np.abs(eigs))
    #eigs = eigs[indices]
    #vecs = vecs[:,indices]


    c_a_A = np.dot(a_N,vecs[:,0])
    c_a_B = np.dot(a_N,vecs[:,1])
    c_b_A = np.dot(b_N_1,vecs[:,0])
    c_b_B = np.dot(b_N_1,vecs[:,1])

    A = np.array([c_a_A,c_b_A])
    B = np.array([c_a_B,c_b_B])

    S = np.array([[c_a_A,c_a_B],[c_b_A,c_b_B]])
    S_1 = sl.inv(S)
    Lambda = np.diag(eigs)
    #h = np.matmul(S,np.matmul(Lambda,S_1))

    #h = eigs[0]*np.outer(A,A)/np.dot(A,A) + eigs[1]*np.outer(B,B)/np.dot(B,B)
    #print(h)
    #print(sl.eigvals(h))

    for k in range(vecs.shape[1]):
        vecs[:,k] *= 1./np.sqrt(np.dot(vecs[:,k],vecs[:,k]))
        index = np.argmax(np.abs(vecs[:,k]))
        vecs[:,k] *=  np.exp(-1j*np.angle(vecs[index,k]))

    #print(np.dot(a_N,vecs[:,0]))
    #print(np.dot(a_N,vecs[:,1]))
    #print(np.dot(b_N_1,vecs[:,0]))
    #print(np.dot(b_N_1,vecs[:,1]))


    h_12 = (eigs[0]-eigs[1])*np.dot(a_N,vecs[:,0])*np.dot(b_N_1,vecs[:,0])
    print(f'h_12: {h_12}')
    h_12 = (eigs[1]-eigs[0])*np.dot(a_N,vecs[:,1])*np.dot(b_N_1,vecs[:,1])
    print(f'h_12: {h_12}')

    h_11 = 0.5*(eigs[0]+ eigs[1]) -0.5*np.sqrt((eigs[0]-eigs[1])**2-4*h_12**2)
    h_22 = 0.5*(eigs[0]+ eigs[1]) +0.5*np.sqrt((eigs[0]-eigs[1])**2-4*h_12**2)
    print(f'h_11: {h_11}')
    print(f'h_22: {h_22}')

    h = np.array([[h_11,h_12],[h_12,h_22]])

    vals = sl.eigvals(h)
    print(f'lambda_A: {vals[0]}')
    print(f'lambda_B: {vals[1]}') """


    if plot:
        fig,ax = plt.subplots()

        #ax.plot(np.abs(a_N))
        #ax.plot(np.abs(b_N_1))
        for k in range(vecs.shape[1]):
            #if np.abs(np.dot(a_N,vecs[:,k])) >tol:
            #    ax.plot(np.abs(vecs[:,k]))
            ax.semilogy(np.abs(vecs[:,k]))
            #if k==1:
            #    break

        for i in range(N_blocks):
            ax.vlines(i*N_elements,1e-8,1,colors='k',linestyles='dashed')

        ax.set_ylim(ymin=1e-8)

        format_plot(fig,ax,'State index','Amplitude')
        plt.show()

    return eigs,vecs,max_block

def E_au_I_wcm2(I):
    return np.sqrt(I/(3.51*10**(16)))
