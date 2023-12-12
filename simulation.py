import os 
import numpy as np
from cis_Floquet import Floquet, E_au_I_wcm2

class Simulation:
    def __init__(self,input_file):
        self.input_file = input_file
        with open(input_file,'r') as file:
            self.cis_loc = self.read_line(file)
            print(f'cis_loc: {self.cis_loc}')
            self.output = self.read_line(file)
            print(f'output: {self.output}')
            self.gauge = self.read_line(file) #Not needed at the moment, only length gauge is currently implemented
            print(f'gauge: {self.gauge}')
            self.N_eigenvalues = int(self.read_line(file))
            print(f'N_eigenvalues: {self.N_eigenvalues}')
            self.shift = float(self.read_line(file))
            print(f'shift: {self.shift}')
            N_blocks = self.read_line(file).split()
            self.N_blocks_up = int(N_blocks[0])
            self.N_blocks_down = int(N_blocks[1])
            print(f'N_blocks: {self.N_blocks_up} {self.N_blocks_down}')
            self.scan_type = self.read_line(file)
            print(f'scan_type: {self.scan_type}')
            scan_range = self.read_line(file).split()
            self.start_range = float(scan_range[0])
            self.end_range = float(scan_range[1])
            self.N_scan = int(scan_range[2])
            print(f'scan_range: {scan_range}')
            self.other_parameter = float(self.read_line(file))
            print(f'other_parameter: {self.other_parameter}')
            plot = self.read_line(file)
            print(f'plot: {plot}')

        if plot == 't':
            self.plot = True
        elif plot == 'f':
            self.plot = False
        else:
            print(f'Unknwown plot value {plot}, please use \'t\' or \'f\'' )
            exit()


        if self.scan_type == 'omega':
            print('----------------')
            print(f'Doing omega scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a. u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [a. u.], End: {self.end_range} [a. u.], N_points: {self.N_scan}')
            print(f'Intensity: {self.other_parameter} [W/cm2]')
            print('----------------')
            print('')
            self.run_simulation = self.omega_scan


        elif self.scan_type == 'intensity':
            print('----------------')
            print(f'Doing intensity scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a. u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [a. u.], End: {self.end_range} [a. u.], N_points: {self.N_scan}')
            print(f'Intensity: {self.other_parameter} [W/cm2]')
            print('----------------')
            print('')
            self.run_simulation = self.intensity_scan

        else:
            print(f'Unknown scan type: {self.scan_type}, the currently implemented ones are \'omega\' and \'intensity\'')
            exit()

        return

    def read_line(self,file):
        return file.readline().strip('\n')

    def read_matrices(self):
        #Read dipole matrix and hamiltonian used for Floquet and convert to sparse format
        z_re = np.loadtxt(f'{self.cis_loc}/dipoles_re.dat')
        z_im = np.loadtxt(f'{self.cis_loc}/dipoles_im.dat')
        H_re = np.loadtxt(f'{self.cis_loc}/mat_re.dat')
        H_im = np.loadtxt(f'{self.cis_loc}/mat_im.dat')
        self.z = z_re + 1j*z_im
        self.H = H_re + 1j*H_im
        return


    def omega_scan(self):
        #Calculate Floquet energies and vectors for different omega and fixed intensity
        omega_vec = np.linspace(self.start_range,self.end_range,self.N_scan)
        intensity = self.other_parameter
        E_0 = E_au_I_wcm2(intensity)

        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)

        for index,omega in np.ndenumerate(omega_vec):
            self.eigs[index[0],:],self.vecs[:,index[0]:index[0] + self.N_eigenvalues] = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues)       

        print('')
        print('The calculations have finished!')
        print('')

        self.save_output(omega_vec)
        return


    def intensity_scan(self):
        #Calculate Floquet energies and vectors for different intensities and fixed omega
        intensity_vec = np.logspace(np.log10(self.start_range),np.log10(self.end_range),self.N_scan)
        E_0_vec = E_au_I_wcm2(intensity_vec)
        omega = self.other_parameter
        
        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)

        for index,E_0 in np.ndenumerate(E_0_vec):
            self.eigs[index[0],:],self.vecs[:,index[0]:index[0] + self.N_eigenvalues] = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues)

        print('')
        print('The calculations have finished!')
        print('')
        self.save_output(intensity_vec)
        return

    def make_outputfolder(self):
        #Make a numbered directory XXXX in output directory to store calculation results
        for i in range(1,10000):
            this_run = '{:d}'.format(i).zfill(4)
            output_folder = f'{self.output}/{this_run}'
            if not(os.path.isdir((output_folder))):
                os.makedirs(output_folder)
                self.output_folder = output_folder
                break
        return

    def save_output(self,scan_vec):
        #Save the output of calculations, and copy input file to directory where output is stored
        self.make_outputfolder()
        print('')
        print(f'Saving output in {self.output_folder}')
        print('')

        os.system(f'cp {self.input_file} {self.output_folder}')
        if self.scan_type == 'omega':
            np.savetxt(f'{self.output_folder}/omega.out',scan_vec,header = 'omega [a.u.]')
        elif self.scan_type == 'intensity':
            np.savetxt(f'{self.output_folder}/intensity.out',scan_vec,header = 'Intensity [a.u.]')

        np.savetxt(f'{self.output_folder}/energies.out',self.eigs,header = 'Energies in [a.u], each row is one calculation')
        np.savetxt(f'{self.output_folder}/vecs.out',self.vecs,header = 'Amplitudes of eigenvectors stored in columns, each column block of size N_eigenvalues is one calculation')

        return