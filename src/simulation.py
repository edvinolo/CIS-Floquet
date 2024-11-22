import os
import numpy as np
import scipy.sparse as sp
from cis_Floquet import Floquet, E_au_I_wcm2

class Simulation:
    def __init__(self,input_file):
        self.input_file = input_file
        with open(input_file,'r') as file:
            self.cis_loc = self.read_line(file)
            print(f'cis_loc: {self.cis_loc}')

            self.output = self.read_line(file)
            print(f'output: {self.output}')

            self.gauge = self.read_line(file)
            print(f'gauge: {self.gauge}')

            self.N_eigenvalues = int(self.read_line(file))
            print(f'N_eigenvalues: {self.N_eigenvalues}')

            shift = self.read_line(file).split()
            self.shift = float(shift[0]) + 1j*float(shift[1])
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

            plot_store_vecs = self.read_line(file).split()
            plot = plot_store_vecs[0]
            store_vecs = plot_store_vecs[1]
            print(f'plot: {plot}')
            print(f'store_vecs: {store_vecs}')

            factor_options = self.read_line(file).split()
            form_H_fl = factor_options[0]
            use_fortran = factor_options[1]
            print(f'form_H_fl: {form_H_fl}')
            print(f'use_fortran: {use_fortran}')

        if not (self.gauge == 'l' or self.gauge == 'v'):
            print(f'Wrong gauge: {self.gauge}. Please use length (l) or velocity (v)')
            exit()

        if plot == 't':
            self.plot = True
        elif plot == 'f':
            self.plot = False
        else:
            print(f'Unknwown plot value {plot}, please use \'t\' or \'f\'')
            exit()

        if store_vecs == 't':
            self.store_vecs = True
        elif store_vecs == 'f':
            self.store_vecs = False
        else:
            print(f'Unknown store_vecs value {store_vecs}, please use \'t\' or \'f\'')

        if form_H_fl == 't':
            self.form_H_fl = True
        elif form_H_fl == 'f':
            self.form_H_fl = False
        else:
            print(f'Unknown form_H_fl value {form_H_fl}, please use \'t\' or \'f\'')

        if use_fortran == 't':
            self.use_fortran = True
        elif use_fortran == 'f':
            self.use_fortran = False
        else:
            print(f'Unknown use_fortran value {use_fortran}, please use \'t\' or \'f\'')


        if self.scan_type == 'omega':
            print('----------------')
            print(f'Doing omega scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [a.u.], End: {self.end_range} [a.u.], N_points: {self.N_scan}')
            print(f'Intensity: {self.other_parameter} [W/cm2]')
            print('----------------')
            print('')
            self.run_simulation = self.omega_scan


        elif self.scan_type == 'intensity':
            print('----------------')
            print(f'Doing intensity scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [W/cm2], End: {self.end_range} [W/cm2], N_points: {self.N_scan}')
            print(f'Omega: {self.other_parameter} [a.u.]')
            print('----------------')
            print('')
            self.run_simulation = self.intensity_scan

        elif self.scan_type == 'omega_follow':
            print('----------------')
            print(f'Doing omega follow scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [a.u.], End: {self.end_range} [a.u.], N_points: {self.N_scan}')
            print(f'Intensity: {self.other_parameter} [W/cm2]')
            print('----------------')
            print('')
            self.run_simulation = self.omega_follow

        elif self.scan_type == 'intensity_follow':
            print('----------------')
            print(f'Doing intensity follow scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Start: {self.start_range} [W/cm2], End: {self.end_range} [W/cm2], N_points: {self.N_scan}')
            print(f'Omega: {self.other_parameter} [a.u.]')
            print('----------------')
            print('')
            self.run_simulation = self.intensity_follow

        elif self.scan_type == 'circle':
            print('----------------')
            print(f'Doing circle scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Center: {self.start_range} [a.u.], {self.end_range} [W/cm2], N_points: {self.N_scan}')
            print(f'Circle radius: {self.other_parameter} [a.u.]')
            print('----------------')
            print('')
            self.run_simulation = self.circle_scan

        elif self.scan_type == 'circle_follow':
            print('----------------')
            print(f'Doing circle follow scan of {self.N_eigenvalues} Floquet energies close to {self.shift} [a.u.]')
            print(f'# Blocks_up: {self.N_blocks_up}, # Blocks_down: {self.N_blocks_down}')
            print(f'Center: {self.start_range} [a.u.], {self.end_range} [W/cm2], N_points: {self.N_scan}')
            print(f'Circle radius: {self.other_parameter} [a.u.]')
            print('----------------')
            print('')
            self.run_simulation = self.circle_follow

        else:
            print(f'Unknown scan type: {self.scan_type}, the currently implemented ones are \'omega\', \'intensity\',\'circle\', \'omega_follow\', \'intensity_follow\', and \'circle_follow\'')
            exit()

        return

    def read_line(self,file):
        return file.readline().strip('\n')

    def read_matrices(self):
        #Read dipole matrix and hamiltonian used for Floquet in sparse format
        #z_re = np.loadtxt(f'{self.cis_loc}/dipoles_re.dat')
        #z_im = np.loadtxt(f'{self.cis_loc}/dipoles_im.dat')
        #H_re = np.loadtxt(f'{self.cis_loc}/mat_re.dat')
        #H_im = np.loadtxt(f'{self.cis_loc}/mat_im.dat')

        #self.z = z_re + 1j*z_im
        #self.H = H_re + 1j*H_im

        self.H = self.read_COO(f'{self.cis_loc}/H_COO.dat')
        self.z = self.read_COO(f'{self.cis_loc}/Z_COO.dat')

        Ediag = self.read_Ediag(f'{self.cis_loc}/Ediag.dat')
        Ediag = sp.diags(Ediag)
        self.H += Ediag

        return

    def read_COO(self,path):
        #Read a complex COO matrix from file
        with open(path,'r') as file:
            shape = self.read_line(file).split()
            shape = (int(shape[0]),int(shape[1]))
            nnz = int(self.read_line(file))

            if nnz == 0:
                mat = sp.coo_matrix(shape,dtype=np.complex128)
            else:
                re = np.fromstring(self.read_line(file),sep = ' ')
                im = np.fromstring(self.read_line(file),sep = ' ')
                elements = re + 1j*im

                assert elements.shape[0]==nnz, "Elements array has wrong lengt"

                row = np.fromstring(self.read_line(file), dtype = np.int64, sep = ' ')
                col = np.fromstring(self.read_line(file), dtype = np.int64, sep = ' ')
                mat = sp.coo_matrix((elements,(row,col)),shape=shape)

        return mat.tocsr()

    def read_Ediag(self,path):
        #Read diagonal elements of H
        E = np.loadtxt(path)
        diag = E[0,:] + 1j*E[1,:]

        return diag

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
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        for index,omega in np.ndenumerate(omega_vec):
            self.eigs[index[0],:],self.vecs[:,index[0]:index[0] + self.N_eigenvalues],self.max_block[index[0],:] = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        self.gauge, energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

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
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        for index,E_0 in np.ndenumerate(E_0_vec):
            self.eigs[index[0],:],self.vecs[:,index[0]:index[0] + self.N_eigenvalues],self.max_block[index[0],:] = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        self.gauge, energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

        print('')
        print('The calculations have finished!')
        print('')
        self.save_output(intensity_vec)
        return

    def circle_scan(self):
        #Follow N_eigenvalues as the intensity and omega is varied around a circle
        angle_vec = np.linspace(0,2*np.pi,self.N_scan)
        radius = self.other_parameter
        omega_vec = self.start_range*(1.0 + radius*np.sin(angle_vec))
        intensity_vec = self.end_range*(1.0 + radius*np.cos(angle_vec))
        E_0_vec = E_au_I_wcm2(intensity_vec)

        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        for index,E_0 in np.ndenumerate(E_0_vec):
            omega = omega_vec[index[0]]
            self.eigs[index[0],:],self.vecs[:,index[0]:index[0] + self.N_eigenvalues],self.max_block[index[0],:] = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        self.gauge, energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,sort_type = 'abs',
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

        print('')
        print('The calculations have finished!')
        print('')
        self.save_output_circle(omega_vec,intensity_vec)
        return

    def omega_follow(self):
        #Follow N_eigenvalues as the omega is scanned for fixed intensity
        omega_vec = np.linspace(self.start_range,self.end_range,self.N_scan)
        intensity = self.other_parameter
        E_0 = E_au_I_wcm2(intensity)

        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        self.eigs[0,:],self.vecs[:,:self.N_eigenvalues],self.max_block[0,:] = Floquet(omega_vec[0],E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down,
                                                                                        self.gauge, energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

        for index,omega in np.ndenumerate(omega_vec[1:]):
            for eig_index,eig in np.ndenumerate(self.eigs[index[0],:]):
                #Here I set n_eig = 2 since I am only interested in one eigenvalue (could maybe use less?)
                eigs,vecs,blocks = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down, self.gauge,
                                          energy = eig,plot = self.plot,N_eig = 2,sort_type = 'vec',prev_vec = self.vecs[:,(index[0])*self.N_eigenvalues + eig_index[0]],
                                          fortran = self.use_fortran, form_H_fl = self.form_H_fl)

                # Need to add 1 to index since ndenumerate starts at 0
                self.eigs[index[0]+1,eig_index[0]] = eigs[0]
                self.vecs[:,(index[0] + 1)*self.N_eigenvalues +eig_index[0]] = vecs[:,0]
                self.max_block[index[0]+1,eig_index[0]] = blocks[0]

        print('')
        print('The calculations have finished!')
        print('')

        self.save_output(omega_vec)
        return

    def intensity_follow(self):
        #Follow N_eigenvalues as the intensity is scanned for fixed omega
        intensity_vec = np.logspace(np.log10(self.start_range),np.log10(self.end_range),self.N_scan)
        E_0_vec = E_au_I_wcm2(intensity_vec)
        omega = self.other_parameter

        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        self.eigs[0,:],self.vecs[:,:self.N_eigenvalues],self.max_block[0,:] = Floquet(omega,E_0_vec[0],self.H,self.z,self.N_blocks_up,self.N_blocks_down, self.gauge,
                                                                                        energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

        for index,E_0 in np.ndenumerate(E_0_vec[1:]):
            for eig_index,eig in np.ndenumerate(self.eigs[index[0],:]):
                #Here I set n_eig = 2 since I am only interested in one eigenvalue (could maybe use less?)
                eigs,vecs,blocks = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down, self.gauge,
                        energy = eig,plot = self.plot,N_eig = 2,sort_type = 'vec',prev_vec = self.vecs[:,(index[0])*self.N_eigenvalues + eig_index[0]],
                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

                # Need to add 1 to index since ndenumerate starts at 0
                self.eigs[index[0]+1,eig_index[0]] = eigs[0]
                self.vecs[:,(index[0] + 1)*self.N_eigenvalues +eig_index[0]] = vecs[:,0]
                self.max_block[index[0]+1,eig_index[0]] = blocks[0]

        print('')
        print('The calculations have finished!')
        print('')
        self.save_output(intensity_vec)
        return

    def circle_follow(self):
        #Follow N_eigenvalues as the intensity and omega is varied around a circle
        angle_vec = np.linspace(0,2*np.pi,self.N_scan)
        radius = self.other_parameter
        omega_vec = self.start_range*(1.0 + 0.02*radius*np.sin(angle_vec))
        intensity_vec = self.end_range*(1.0 + radius*np.cos(angle_vec))
        E_0_vec = E_au_I_wcm2(intensity_vec)

        print(omega_vec)
        print(E_0_vec)

        self.read_matrices()

        self.eigs = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.complex128)
        N_floquet_states = self.H.shape[0]*(self.N_blocks_up+self.N_blocks_down + 1)
        vecs_shape = (N_floquet_states,self.N_eigenvalues*self.N_scan)
        self.vecs = np.zeros(vecs_shape,dtype = np.complex128)
        self.max_block = np.zeros((self.N_scan,self.N_eigenvalues),dtype = np.int64)

        self.eigs[0,:],self.vecs[:,:self.N_eigenvalues],self.max_block[0,:] = Floquet(omega_vec[0],E_0_vec[0],self.H,self.z,self.N_blocks_up,self.N_blocks_down, self.gauge,
                                                                                        energy = self.shift,plot = self.plot,N_eig = self.N_eigenvalues,
                                                                                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

        for index,E_0 in np.ndenumerate(E_0_vec[1:]):
            omega = omega_vec[index[0]+1]
            for eig_index,eig in np.ndenumerate(self.eigs[index[0],:]):
                #Here I set n_eig = 2 since I am only interested in one eigenvalue (could maybe use less?)
                eigs,vecs,blocks = Floquet(omega,E_0,self.H,self.z,self.N_blocks_up,self.N_blocks_down, self.gauge,
                        energy = eig,plot = self.plot,N_eig = 2,sort_type = 'vec',prev_vec = self.vecs[:,(index[0])*self.N_eigenvalues + eig_index[0]],
                        fortran = self.use_fortran, form_H_fl = self.form_H_fl)

                # Need to add 1 to index since ndenumerate starts at 0
                self.eigs[index[0]+1,eig_index[0]] = eigs[0]
                self.vecs[:,(index[0] + 1)*self.N_eigenvalues +eig_index[0]] = vecs[:,0]
                self.max_block[index[0]+1,eig_index[0]] = blocks[0]

        print('')
        print('The calculations have finished!')
        print('')
        self.save_output_circle(omega_vec,intensity_vec)
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
        if self.scan_type == 'omega' or self.scan_type == 'omega_follow':
            np.savetxt(f'{self.output_folder}/omega.out',scan_vec,header = 'omega [a.u.]')
        elif self.scan_type == 'intensity' or self.scan_type == 'intensity_follow':
            np.savetxt(f'{self.output_folder}/intensity.out',scan_vec,header = 'Intensity [a.u.]')

        np.savetxt(f'{self.output_folder}/energies.out',self.eigs,header = 'Energies in [a.u], each row is one calculation')
        np.savetxt(f'{self.output_folder}/max_block.out',self.max_block,header = 'The Floquet block with maximum norm, each row is one calculation')
        if self.store_vecs and (self.vecs is not None):
            np.savetxt(f'{self.output_folder}/vecs.out',self.vecs,header = 'Amplitudes of eigenvectors stored in columns, each column block of size N_eigenvalues is one calculation')

        return

    def save_output_circle(self,omega_vec,intensity_vec):
        #Save the output of calculations, and copy input file to directory where output is stored
        self.make_outputfolder()
        print('')
        print(f'Saving output in {self.output_folder}')
        print('')

        os.system(f'cp {self.input_file} {self.output_folder}')

        np.savetxt(f'{self.output_folder}/omega.out',omega_vec,header = 'omega [a.u.]')
        np.savetxt(f'{self.output_folder}/intensity.out',intensity_vec,header = 'Intensity [a.u.]')

        np.savetxt(f'{self.output_folder}/energies.out',self.eigs,header = 'Energies in [a.u], each row is one calculation')
        np.savetxt(f'{self.output_folder}/max_block.out',self.max_block,header = 'The Floquet block with maximum norm, each row is one calculation')
        if self.store_vecs and (self.vecs is not None):
            np.savetxt(f'{self.output_folder}/vecs.out',self.vecs,header = 'Amplitudes of eigenvectors stored in columns, each column block of size N_eigenvalues is one calculation')

        return
