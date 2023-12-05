import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from class_Floquet import Floquet_system

sys.path.append('./Effective-Ham/src/')
sys.path.append('./plot_utils/')
from plot_utils import format_plot
from effective_Hamiltonian import complex_Rabi




z_re = np.loadtxt('dipoles_re.dat')
z_im = np.loadtxt('dipoles_im.dat')
H_re = np.loadtxt('mat_re.dat')
H_im = np.loadtxt('mat_im.dat')

z = z_re + 1j*z_im
#z = np.zeros(z_re.shape,dtype = np.complex128)
H = H_re + 1j*H_im



def Floquet(omega,E_0,H,z,N_blocks_up,N_blocks_down,**kwargs):
    
    energy = kwargs.get('energy',0.0)
    plot = kwargs.get('plot',False)
    tol = kwargs.get('tol',2e-1)
    I_p = kwargs.get('I_p',None)


    N_elements = H.shape[0]
    I_omega = omega*np.eye(N_elements,dtype = np.complex128)

    m = range(-N_blocks_up,N_blocks_down+1)
    N_blocks = len(m)    

    N_floquet = N_elements*N_blocks
    
    print('')
    print('Diagonalizing floquet Hamiltonian')
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
    
    
    

    H_fl = sp.lil_matrix((N_floquet,N_floquet),dtype = np.complex128)

    H_test = sp.lil_matrix(H)
    print(f'# of nonzero in H: {H_test.nnz}, out of {N_elements**2}')

    for i in range(N_blocks):
        H_fl[i*N_elements:(i+1)*N_elements,i*N_elements:(i+1)*N_elements] = H + m[i]*I_omega


        if i !=N_blocks-1:
            H_fl[i*N_elements:(i+1)*N_elements,(i+1)*N_elements:(i+2)*N_elements] = z*E_0/2 
            H_fl[(i+1)*N_elements:(i+2)*N_elements,i*N_elements:(i+1)*N_elements] = z*E_0/2 

    H_fl = sp.csr_matrix(H_fl)#,blocksize = (N_elements,N_elements))
    print(f'# of nonzero in H_fl: {H_fl.nnz}, out of {N_floquet**2}')
    
    #H_fl_sys = Floquet_system(H,z,omega,E_0,N_blocks_up,N_blocks_down,shift = -1e-7)

    #ones = np.ones(N_floquet,dtype = np.complex128)
    #bs = np.zeros(N_floquet,dtype = np.complex128)
    #bs[N_blocks_up*N_elements+1] = 1.0
    #diff = H_fl@ones - H_fl_sys.H_linop@ones
    #print(f'Relative L_2 norm difference: {sl.norm(diff)/sl.norm(ones)}')

    #spilu = spl.spilu(H_fl)
    #M = spl.LinearOperator((N_floquet,N_floquet),spilu.solve)
    #res,info = spl.gmres(H_fl,ones,M = M,maxiter = 40)
    #print(f'GMRES info: {info}')

    #H_fl_sys.block_solve_setup()
    #res = H_fl_sys.block_solve(ones)
    #res_2 = spl.spsolve(H_fl,bs)
    #res = H_fl_sys.solve(ones)
    #diff = ones-H_fl_sys.H_linop@res
    #print(f'Relative L_2 norm difference: {sl.norm(diff)/sl.norm(ones)}')

    
    
    
    eigs,vecs = spl.eigs(H_fl,sigma = energy,k=6) 
    #eigs,vecs = spl.eigs(H_fl_sys.H_invop,k=6,maxiter=10,tol = 1e-1)#,sigma=0.0,OPinv = H_fl_sys.H_invop)

    print('')
    print('----------------------------------------')    
    print('Eigenvalues found by Arnoldi iteration: ')
    print(eigs)
    print('----------------------------------------')
    print('')

    #print(vecs[N_blocks*N_elements:N_blocks*N_elements+2,0])
    #print(vecs[N_blocks*N_elements:N_blocks*N_elements+2,1])

    eigs_H, vecs_H = spl.eigs(H,sigma = -0.1)
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
    print(f'lambda_B: {vals[1]}')


    if plot:
        fig,ax = plt.subplots()

        ax.plot(np.abs(a_N))
        #ax.plot(np.abs(b_N_1))
        for k in range(vecs.shape[1]):
            #if np.abs(np.dot(a_N,vecs[:,k])) >tol:
            #    ax.plot(np.abs(vecs[:,k]))
            ax.semilogy(np.abs(vecs[:,k]))
            if k==1:
                break

        for i in range(N_blocks):
            ax.vlines(i*N_elements,1e-8,1,colors='k',linestyles='dashed')

        ax.set_ylim(ymin=1e-8)

    
        format_plot(fig,ax,'State index','Amplitude')

    return eigs,vecs

def E_au_I_wcm2(I):
    return np.sqrt(I/(3.51*10**(16)))

omega = 0.375
E_0=0.024
#E_0 = 0.09245003270420488
#E_0 = 0
#E_0= 1.0e-5
N_blocks_abs = 5
N_blocks_em = 5

#eigs,vecs = Floquet(omega,E_0,H,z,N_blocks)

#eigs,vecs = Floquet(omega,E_0,H,z,N_blocks_abs,N_blocks_em,plot = False,I_p = 0.5)

#omega = 0.79720797798132026
omega_vec = np.linspace(0.78,0.8,9)
#omega_vec = np.linspace(0.372,0.378,9)

fig_re,ax_re = plt.subplots()
fig_im,ax_im = plt.subplots()
for i,omega in np.ndenumerate(omega_vec):
    eigs,vecs = Floquet(omega,E_0,H,z,N_blocks_abs,N_blocks_em)

    re_max = np.argmax(np.real(eigs[:2]))
    re_min = np.argmin(np.real(eigs[:2]))

    ax_re.plot(omega,np.real(eigs[re_max]), 'k+')
    ax_re.plot(omega,np.real(eigs[re_min]),'r+')
    #ax_re.plot(omega,np.real(eigs[2]),'gs')
    

    ax_im.semilogy(omega,-2*np.imag(eigs[re_max]), 'k+')
    ax_im.semilogy(omega,-2*np.imag(eigs[re_min]),'r+')
    #ax_im.semilogy(omega,-2*np.imag(eigs[2]),'gs')

    AT_split = np.abs(np.real(eigs[0]-eigs[1]))
    print(eigs)
    print(f'Estimated AT-splitting: {AT_split}')

format_plot(fig_re,ax_re,'Frequency [a.u.]','Re $E$ [a.u.]')
format_plot(fig_im,ax_im,'Frequency [a.u.]','$\gamma$ [a.u.]')


omega = 0.79720797798132026
#omega = 0.79720797798132026/4.5
#omega = 0.375
#omega = 0.5/(1.4*np.pi)

I = np.logspace(np.log10(1.5e12),np.log10(1.5e14),10)
E_vec = E_au_I_wcm2(I)

fig_re,ax_re = plt.subplots()
fig_im,ax_im = plt.subplots()
cmap = plt.get_cmap("tab10")

for i,E_0 in np.ndenumerate(E_vec):
    eigs,vecs = Floquet(omega,E_0,H,z,N_blocks_abs,N_blocks_em,plot = True,I_p = 0.5,energy = 0.0)

    

    re_max = np.argmax(np.real(eigs[:2]))
    re_min = np.argmin(np.real(eigs[:2]))

    ax_re.plot(I[i],np.real(eigs[re_max]), 'k+')
    ax_re.plot(I[i],np.real(eigs[re_min]),'r+')

    for k in range(eigs.shape[0]):
        ax_re.plot(I[i],np.real(eigs[k]), '+',color = cmap(k+1))
        ax_im.loglog(I[i],-2*np.imag(eigs[k]), '+',color = cmap(k+1))

    

    ax_im.loglog(I[i],-2*np.imag(eigs[re_max]), 'k+')
    ax_im.loglog(I[i],-2*np.imag(eigs[re_min]),'r+')

    AT_split = np.abs(np.real(eigs[0]-eigs[1]))
    print(f'Estimated AT-splitting: {AT_split}')

    atom  = 'He'
    if atom == 'H':
        c_R = complex_Rabi(0,0.375,E_0,0.74493553902759979,0.375)
        c_R.shifts_to_order_4(-4.2994235515557779,11.696868492111513 -1j*0.52234350391660433,
                                1451.8840693208547 -1j*233.08047840057807,
                                2887.8029818581272-1j*95.572064969963833,
                                54.256253682354838+1j*9.3216026855676510,disp=False)
    elif atom == 'He':
        c_R = complex_Rabi(0,0.7972081390130366,E_0,0.40401408872998712,
                             0.7972081390130366)
    
        c_R.shifts_to_order_4(-2.8531912039522886,2.0102907828882017 -1j*7.87087926573738425E-003,
                                155.55015484148478 -1j*10.320460115321691,
                                120.01707430398346-1j*0.21535244713361390,
                                8.5330261414000717+1j*0.23255995550376793,disp = True)
    
    c_R.compute_eigenvalues(disp=False)
    print(f'Model h_11: {c_R.h_11}')
    print(f'Model h_22: {c_R.h_22}')
    print(f'Model h_12: {c_R.h_12}')
    print(f'lambda_+: {c_R.lambda_B}')
    print(f'lambda_-: {c_R.lambda_A}')

    ax_re.plot(I[i],np.real(c_R.lambda_B), 'kx')
    ax_re.plot(I[i],np.real(c_R.lambda_A),'rx')    
    
    ax_im.loglog(I[i],-2*np.imag(c_R.lambda_B), 'kx')
    ax_im.loglog(I[i],-2*np.imag(c_R.lambda_A),'rx')

format_plot(fig_re,ax_re,'Intensity [W/cm$^2$]','Re $E$ [a.u.]')
format_plot(fig_im,ax_im,'Intensity [W/cm$^2$]','$\gamma$ [a.u.]')

plt.show()

