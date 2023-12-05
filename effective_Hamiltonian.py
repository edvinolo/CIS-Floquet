import numpy as np #For working with arrays
import matplotlib #Used for changing some plot parameters
import matplotlib.pyplot as plt #Used for plotting
import sys #Used for getting command line arguments
import scipy.optimize as so


sys.path.append('/home/edvinolofsson/bin/')
from constants import *
from plot_utils import format_plot

class complex_Rabi:
    #Class that has parameters and functions related to the Rabi model that inlcudes ionization losses through
    #through complex energy shifts. The shifts are supplied by the user, and can be estimated using time-dependent 
    #perturbation theory.
    #The theory is based on  C. R. Holt, M. G. Raymer, and W. P. Reinhardt, Phys. Rev. A 27 2971 (1983)
    
    def __init__(self,E_1,E_2,E_0,dipole,omega):
        #Initialize the Rabi model with parameters that are used in the lowest order 
        #E_1,E_2: energies of two level system
        #E_0: electric field strength
        #dipole: the dipole matrix element that couples the levels of the two level system
        #omega: freuquncy of the laser that drives the Rabi oscillations
        self.E_1 = E_1
        self.E_2 = E_2
        self.E_0 = E_0
        self.dipole = dipole
        self.omega = omega
        
        #Calculate some useful quantities for Rabi oscillations
        self.omega_21 = self.E_2-self.E_1
        self.detuning = self.omega-self.omega_21
        self.Rabi_freq = self.E_0*self.dipole
        self.gen_Rabi_freq = np.sqrt(self.detuning**2+self.Rabi_freq**2)
        
        #print('Rabi_freq: ' + str(self.Rabi_freq))
        
    def shifts_to_order_4(self,alpha_2_1,alpha_2_2,alpha_4_1,alpha_4_2,alpha_3_21,disp = True):
        #Compute the shifts to fourth order in perturbation theory using polarizabilities that have been precomputed
        #alpha_2_1: second order coefficient for state 1
        #alpha_2_2: second order coefficient for state 2
        #alpha_4_1: fourth order coefficient for state 1
        #alpha_4_2: fourth order coefficient for state 2
        #alpha_3_21: third order coefficient for transition between states 1 and 2
        
        if disp:
            print('')
            print('---------------------------')
            print('Computing the energy shifts to 4th order...')
            print('---------------------------')
        
        self.alpha_2_1 = alpha_2_1
        self.alpha_2_2 = alpha_2_2
        self.alpha_4_1 = alpha_4_1
        self.alpha_4_2 = alpha_4_2
        self.alpha_3_21 = alpha_3_21
        
        #Compute the real AC-Stark shifts to fourth order
        self.delta_1 = np.real(self.alpha_2_1)*self.E_0**2/2**2 + np.real(self.alpha_4_1)*self.E_0**4/2**4
        self.delta_2 = np.real(self.alpha_2_2)*self.E_0**2/2**2 + np.real(self.alpha_4_2)*self.E_0**4/2**4
        
        #Compute ionization rates to fourth order
        self.gamma_1 = -2*np.imag(self.alpha_4_1)*self.E_0**4/2**4
        self.gamma_2 = -2*np.imag(self.alpha_2_2)*self.E_0**2/2**2 -2*np.imag(self.alpha_4_2)*self.E_0**4/2**4
        
        #Off-diagonal third order contribution
        self.beta = 2*self.alpha_3_21*self.E_0**3/2**3 #Check if the sign is correct
        
        #self.c = np.array([0,self.dipole,0,self.alpha_3_21])
        #self.p,self.q = robust_pade(self.c,2,m=1)
        #self.p,self.q = pade(self.c,2)
        #print(self.p.coeffs)
        #print(self.q.coeffs)
        #print(self.p(self.E_0/2)/self.q(self.E_0/2))
        #print(self.Rabi_freq)


        
        if disp:
            print(f'delta_1: {self.delta_1}')
            print(f'delta_2: {self.delta_2}')
            print(f'gamma_1: {self.gamma_1}')
            print(f'gamma_2: {self.gamma_2}')
            print(f'beta: {self.beta}')
            print(f'Shift of resonance: {self.delta_2-self.delta_1}')
        
    def set_parameters(self,delta_1,delta_2,gamma_1,gamma_2,sign_beta):
        #Set some of the parameters in two-level system by hand 
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
        if sign_beta == 'p':
            self.beta = np.sqrt(gamma_1*gamma_2)
        elif sign_beta == 'm':
            self.beta = -np.sqrt(gamma_1*gamma_2)
        else:
            print('Error: sign_beta should be m or p') 
        
        
    def compute_eigenvalues(self,disp = True):
        #Compute the eigenvalues of the complex Hamiltonian, which are needed for the time dependent amplitudes of 
        #the two level system
        
        if disp:
            print('')
            print('---------------------------')
            print('Computing the eigenvalues of 2x2 Hamiltonian...')
            print('---------------------------')
        
        #Matrix elements of complex symmetric Hamiltonian
        self.h_11 = self.E_1 + self.delta_1 - 1j*self.gamma_1/2
        self.h_22 = self.E_2 + self.delta_2-self.omega - 1j*self.gamma_2/2
        self.h_12 = +self.Rabi_freq/2+self.beta/2
        #print(self.h_12)
        #self.h_12 = self.p(self.E_0/2)/self.q(self.E_0/2)
        #print(self.h_12)
        #self.h_12 = 0.5*((self.Rabi_freq+self.beta)- (self.beta)**2/((self.beta)-self.Rabi_freq))
        #print(self.h_12)

        self.gen_Rabi_freq = np.sqrt(self.Rabi_freq**2+np.real(self.h_22-self.h_11)**2)
        
        #Eigenvalues of complex symmetric Hamiltonian
        self.lambda_A = (self.h_11+self.h_22)/2 - 0.5*np.sqrt((self.h_11-self.h_22)**2+4*self.h_12**2)
        self.lambda_B = (self.h_11+self.h_22)/2 + 0.5*np.sqrt((self.h_11-self.h_22)**2+4*self.h_12**2)
        
        #Generalized Rabi parameters
        self.complex_detuning = self.h_11-self.h_22
        self.complex_Rabi = -self.lambda_A+self.lambda_B
        if disp:
            print(self.lambda_A)
            print(self.lambda_B)
        
    def compute_gamma_diff(self,E_0):
        #Calulcates the difference in ionization rates for states 1 and 2
        self.gamma_1 = -2*np.imag(self.alpha_4_1)*E_0**4/2**4
        self.gamma_2 = -2*np.imag(self.alpha_2_2)*E_0**2/2**2 -2*np.imag(self.alpha_4_2)*E_0**4/2**4
        return self.gamma_1-self.gamma_2
        
        
    def balance_rates(self,set_E_field=False,limits = (1e-9,1)):
        #Find the electric field strength at which the ionization rates from states 1 and 2 are equal.
        #Also sets the electric field to this value if specified, and recomputes the relevant parameters
        #By default the search is within E_0 \in [1e-9,1] a.u.
        
        print('')
        print('---------------------------')
        print('Balancing the ionization rates...')
        print('---------------------------')
        
        root,r = so.brentq(self.compute_gamma_diff, limits[0],limits[1],full_output = True)
        print(f"Root finding converged? {r.converged}")
        print(f'Rates are the same when E_0 = {root} a.u.')
        
        if set_E_field:
            print(f'Setting the electric field to {root} a.u.')
            self.E_0 = root
            self.shifts_to_order_4(self.alpha_2_1,self.alpha_2_2,self.alpha_4_1,self.alpha_4_2,self.alpha_3_21)
            self.compute_eigenvalues()
    
    def lambda_A_fun(self,E_0):
        self.E_0 = E_0
        self.shifts_to_order_4(self.alpha_2_1,self.alpha_2_2,self.alpha_4_1,self.alpha_4_2,self.alpha_3_21,disp = False)
        #if np.sign(self.beta)>0:
        #    self.set_parameters(self.delta_1,self.delta_2,self.gamma_1,self.gamma_2,'p')
        #else:
        #    self.set_parameters(self.delta_1,self.delta_2,self.gamma_1,self.gamma_2,'m')
        self.compute_eigenvalues(disp = False)
        
        return np.abs(np.imag(self.lambda_A))
    
    def lambda_B_fun(self,E_0):
        self.E_0 = E_0
        self.shifts_to_order_4(self.alpha_2_1,self.alpha_2_2,self.alpha_4_1,self.alpha_4_2,self.alpha_3_21,disp = False)
        #if np.sign(self.beta)>0:
        #    self.set_parameters(self.delta_1,self.delta_2,self.gamma_1,self.gamma_2,'p')
        #else:
        #    self.set_parameters(self.delta_1,self.delta_2,self.gamma_1,self.gamma_2,'m')
        self.compute_eigenvalues(disp = False)
        
        return np.abs(np.imag(self.lambda_B))
        
    def find_blockade(self,set_E_field = False, limits = (1e-5,1)):
        #minimize the imaginary part of the eigenvalues
        #Warning! This method changes the value of E_0 when the optimization runs
        #Be careful to check the value after you're done!
        res_A = so.minimize_scalar(self.lambda_A_fun, bounds=limits, args=(), method='bounded', tol=None, options=None)
        res_B = so.minimize_scalar(self.lambda_B_fun, bounds=limits, args=(), method='bounded', tol=None, options=None)
        
        print(res_A.x,self.lambda_A_fun(res_A.x))
        print(res_B.x,self.lambda_B_fun(res_B.x))
        
        if set_E_field:
            self.E_0 = np.minimum(res_A.x,res_B.x)
        
        return
    
    def evaluate_amplitudes(self,t):
        #Evaluate the time-dependent amplitudes of the two-level system
        a_1 = ((self.lambda_A-self.h_22)/(self.lambda_A-self.lambda_B)*np.exp(-1j*self.lambda_A*t)
                -(self.lambda_B-self.h_22)/(self.lambda_A-self.lambda_B)*np.exp(-1j*self.lambda_B*t))
        a_2 = self.h_12/(self.lambda_A-self.lambda_B)*(np.exp(-1j*self.lambda_A*t)-np.exp(-1j*self.lambda_B*t))
        return [a_1,a_2]
    
    def evaluate_derivative(self,t):
        #Evaluate the derivative of the population in the two-level system
        values = self.evaluate_amplitudes(t)
        population_derivative_1 = (-self.gamma_1*np.abs(values[0])**2 -self.gamma_2*np.abs(values[1])**2
                                 +4*np.imag(self.h_12)*np.real(np.conj(values[0])*values[1]))
        
        population_derivative = (2*np.imag(self.lambda_A)
                                 *np.abs(self.lambda_A-self.h_22)**2*np.exp(2*np.imag(self.lambda_A)*t)+
                                2*np.imag(self.lambda_B)*np.abs(self.lambda_B-self.h_22)**2
                                *np.exp(2*np.imag(self.lambda_B)*t))
        
        population_derivative += 2*np.imag((np.conj(self.lambda_B)-self.lambda_A)*np.conj(self.lambda_B-self.h_22)
                                  *(self.lambda_A-self.h_22)*np.exp(1j*(np.conj(self.lambda_B)-self.lambda_A)*t))
        
        population_derivative += np.abs(self.h_12)**2*(2*np.imag(self.lambda_A)*np.exp(2*np.imag(self.lambda_A)*t)
                                  + 2*np.imag(self.lambda_B)*np.exp(2*np.imag(self.lambda_B)*t))
        
        population_derivative += np.abs(self.h_12)**2*2*np.imag((np.conj(self.lambda_B)-self.lambda_A)
                                            *np.exp(1j*(np.conj(self.lambda_B)-self.lambda_A)*t))
                                
        population_derivative /= np.abs(self.complex_Rabi)**2
        
        return population_derivative
    
    def dressed_populations(self,t):
        a,b = self.evaluate_amplitudes(t)
        prefactor_A = 1/(1+np.abs(self.h_12/(self.lambda_A-self.h_22))**2)
        prefactor_B = 1/(1+np.abs(self.h_12/(self.lambda_B-self.h_22))**2)
        p_A = prefactor_A*np.abs(a+self.h_12/(self.lambda_A-self.h_22)*b)**2
        p_B = prefactor_B*np.abs(a+self.h_12/(self.lambda_B-self.h_22)*b)**2
        
        return [p_A,p_B]
    
    def plot_populations(self,T,**kwargs):
        #Plot the populations of the Rabi-oscillating states over the interval [0,T].
        #The default number of timesteps is 500
        
        time_steps = kwargs.get('time_steps',500)
        scale_t = kwargs.get('scale_t',False) 
        femto = kwargs.get('femto',False)
        vlines = kwargs.get('vlines',None)
        dressed_states = kwargs.get('dressed_states',False)
        ionization = kwargs.get('ionization',False)
        
        t_vec = np.linspace(0,T,time_steps)
        
        amplitudes = self.evaluate_amplitudes(t_vec)
        pop_1 = np.abs(amplitudes[0])**2
        pop_2 = np.abs(amplitudes[1])**2
        pop_tot = pop_1+pop_2
        rate_estimate = self.gamma_2*0.5*self.Rabi_freq**2/self.gen_Rabi_freq**2 + (1.0-0.5*self.Rabi_freq**2/self.gen_Rabi_freq**2)*self.gamma_1
        pop_rate_estimate = np.exp(-t_vec*rate_estimate)
        
        pop_dressed = self.dressed_populations(t_vec)
        pop_A = pop_dressed[0]
        pop_B = pop_dressed[1]
        
        coeff = (pop_tot[-1]-pop_tot[0])/(t_vec[-1]-t_vec[0])
        pop_line = 1+coeff*t_vec
        
        derivative_num = (pop_tot[1:]-pop_tot[:time_steps-1])/(t_vec[1]-t_vec[0])
        derivative = self.evaluate_derivative(t_vec)
        
        if scale_t:
            t_vec *= np.abs(self.Rabi_freq)/(2*np.pi)
            
        if femto:
            t_vec *= 1.0/fs_to_au  
        
        fig_pop,ax_pop = plt.subplots()
        
        ax_pop.plot(t_vec,pop_1,label = '$P_a$')
        ax_pop.plot(t_vec,pop_2,'--',label = '$P_b$')
        ax_pop.plot(t_vec,pop_tot,':',label = '$P_{a+b}$')
        #ax_pop.plot(t_vec,pop_tot,label = '$P_{a+b}$')
        lines, labels = ax_pop.get_legend_handles_labels()
            
        if ionization:
            ax_ion = ax_pop.twinx()
            ax_ion.plot(t_vec,1.0-pop_tot,'C3-.',label = '$P_{ion}$')
            ax_ion.set_ylabel('Ionization probability', fontsize = 20)
            ax_ion.ticklabel_format(axis = 'y',style = 'scientific',scilimits = (1,4))
            ax_ion.yaxis.offsetText.set_fontsize(20)
            ax_ion.tick_params(axis = 'both',labelsize = 20)
            lines2, labels2 = ax_ion.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        
        if dressed_states:
            ax_pop.plot(t_vec,pop_A,'-.',label = '$|c_-|^2e^{-\gamma_-t}$')
            ax_pop.plot(t_vec,pop_B,'-.',label = '$|c_+|^2e^{-\gamma_+t}$')
            #ax_pop.plot(t_vec,pop_A + pop_B,'-.',label = 'Sum A+B')
            #ax_pop.plot(t_vec,pop_B[0]*np.exp(2*np.imag(self.lambda_B)*t_vec*fs_to_au))
            lines2, labels2 = ax_pop.get_legend_handles_labels()
            lines = lines2
            labels = labels2
        
        ax_pop.set_xlabel('time/a.u.', fontsize = 20)
        if scale_t:
            ax_pop.set_xlabel('Time/Rabi periods', fontsize = 20)
        
        if femto:
            ax_pop.set_xlabel('Time [fs]',fontsize = 20)
            
        if vlines is not None:
            for vline in vlines:
                lower_limit = min([pop_B[-1],pop_A[-1]])
                lower_limit *= 0.98 
                ax_pop.vlines(vline,lower_limit,1,'k',linewidth = 4.0)
        
        ax_pop.set_ylabel('Population', fontsize = 20)
        ax_pop.tick_params(axis = 'both', labelsize = 20)
        
        #leg_pop = ax_pop.legend(lines,labels,fontsize = 18,loc = 'best',framealpha  = 0.0,bbox_to_anchor=(0.26,0.72,0.1,0.1))#Fig.4
        #leg_pop = ax_pop.legend(lines,labels,fontsize = 13,loc = 'upper left',framealpha  = 0.0,bbox_to_anchor=(-0.02,0.72))#Fig. 1
        leg_pop = ax_pop.legend(lines,labels,fontsize = 13,loc = 'best',framealpha  = 0.0)

        #ax_pop.grid()
        
        fig_pop.tight_layout()
        
        #fig_sum,ax_sum = plt.subplots()
        fig_sum,ax_sum = plt.subplots(1,2)

        vert_size = 4.5
        fig_sum.set_size_inches(vert_size*2.2,vert_size)
        
        ax_sum[0].semilogy(t_vec,pop_tot,label = '$P_{a+b}$')
        ax_sum[0].semilogy(t_vec,pop_rate_estimate,'-.',label = 'Rate estimate')
        #ax_sum[0].plot(t_vec,pop_line,'--',label = 'Linear')
        
        ax_sum[1].plot(t_vec,derivative,label = 'analytic')
        ax_sum[1].plot(t_vec[:time_steps-1],derivative_num[:time_steps],'--',label = 'numeric')
        ax_sum[1].axhline(y = -self.gamma_1,color='k',label='$\gamma_1$')
        ax_sum[1].axhline(y = -self.gamma_2,color = 'r',label='$\gamma_2$')
        
        
        #ax_sum[0].ticklabel_format(axis = 'y', style='sci', scilimits=(0,0),useOffset=False)
        ax_sum[0].xaxis.offsetText.set_fontsize(20)
        ax_sum[1].ticklabel_format(axis = 'y', style='sci', scilimits=(0,0),useOffset=False)
        ax_sum[1].xaxis.offsetText.set_fontsize(20)
        
        ax_sum[0].set_xlabel('time/a.u.', fontsize = 20)
        ax_sum[1].set_xlabel('time/a.u.', fontsize = 20)
        if scale_t:
            ax_sum[0].set_xlabel('Time/Rabi periods', fontsize = 20)
            ax_sum[1].set_xlabel('Time/Rabi periods', fontsize = 20)
        ax_sum[0].set_ylabel('Population', fontsize = 20)
        ax_sum[1].set_ylabel('Derivative', fontsize = 20)
        ax_sum[0].tick_params(axis = 'both', labelsize = 20)
        ax_sum[1].tick_params(axis = 'both', labelsize = 20)
        
        
        
        ax_sum[0].legend(loc = 'lower left')
        ax_sum[1].legend()
        ax_sum[0].grid()
        ax_sum[1].grid()
        
        fig_sum.tight_layout()
        
    def sweep_E_field(self,T,E_0_min,E_0_max):
        #Plot the population remaining in the two-level system as a function of the electric field strength
        self.E_0  = np.linspace(E_0_min,E_0_max,30000)
        self.shifts_to_order_4(self.alpha_2_1,self.alpha_2_2,self.alpha_4_1,self.alpha_4_2,self.alpha_3_21)
        #self.set_parameters(self.delta_1,self.delta_2,self.gamma_1,self.gamma_2,'p')
        self.compute_eigenvalues()
        a,b = self.evaluate_amplitudes(T)
        pop = np.abs(a)**2 + np.abs(b)**2
        
        fig_sweep,ax_sweep = plt.subplots()
        ax_sweep.plot(self.E_0,1-pop)
        ax_sweep.ticklabel_format(axis = 'y', style='sci', scilimits=(0,0),useOffset=False)
        ax_sweep.xaxis.offsetText.set_fontsize(20)
        ax_sweep.set_xlabel('$E_0$/ a.u.', fontsize = 20)
        ax_sweep.set_ylabel('Ionization probability', fontsize = 20)
        ax_sweep.tick_params(axis = 'both', labelsize = 20)
        fig_sweep.tight_layout()
        
        fig_lambda,ax_lambda = plt.subplots()
        ax_lambda.plot(self.E_0,np.imag(self.lambda_A))
        ax_lambda.plot(self.E_0,np.imag(self.lambda_B))
        
    def imagninary_part(self):
        det = np.real(self.complex_detuning)
        gamma_0 = self.gamma_1 + self.gamma_2
        term = det**2 + self.Rabi_freq**2-gamma_0**2/4-self.beta**2
        first_sqrt = np.sqrt(term**2 + 0.25*(det*gamma_0-2*self.beta*self.Rabi_freq)**2)
        second_sqrt = np.sqrt(first_sqrt + term )
        B = np.sqrt(2)*(det*gamma_0-2*self.beta*self.Rabi_freq)/(4*second_sqrt)
        return [-gamma_0/4+B,-gamma_0/4-B]
    
    def compute_E_breakdown(self,delta_E_bc):
        E_limit = delta_E_bc/np.abs(self.dipole)
        print(f'Expected E_0 where effective Hamiltonian breaks down: {E_limit} [a.u.]')
        return E_limit