import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import Normalize


sys.path.append('./format-plot/')
from plot_utils import format_plot


folder = sys.argv[1]

energies = np.loadtxt(f'{folder}/energies.out',dtype = np.complex128)
energies_real = np.real(energies)
rates = -2*np.imag(energies)

cmap = matplotlib.cm.get_cmap('viridis')
max_block = np.loadtxt(f'{folder}/max_block.out')
colors = cmap(max_block)

sim_type = None
if os.path.isfile(f'{folder}/omega.out'):
    sim_type = 'omega'
    x_variable = np.loadtxt(f'{folder}/omega.out')

elif os.path.isfile(f'{folder}/intensity.out'):
    sim_type = 'intensity'
    x_variable = np.loadtxt(f'{folder}/intensity.out')

else:
    raise RuntimeError('Could not find appropriate data file for x values')


fig_re,ax_re = plt.subplots()
fig_im,ax_im = plt.subplots()
ax_im.set_yscale('log')

for i in range(energies.shape[1]):
    for j in range(x_variable.shape[0]):
        ax_re.plot(x_variable[j],energies_real[j,i],'s',color= colors[j,i])
        ax_im.plot(x_variable[j],rates[j,i],'s',color = colors[j,i])

if sim_type == 'omega':
    format_plot(fig_re,ax_re,'omega [a.u.]','Re [a.u.]')
    format_plot(fig_im,ax_im,'omega [a.u.]','Rate [a.u.]')

elif sim_type == 'intensity':
    ax_re.set_xscale('log')
    ax_im.set_xscale('log')
    format_plot(fig_re,ax_re,'Intensity [W/cm$^2$]','Re [a.u.]')
    format_plot(fig_im,ax_im,'Intensity [W/cm$^2$]','Rate [a.u.]')

plt.show()