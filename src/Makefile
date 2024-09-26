SHELL = /bin/sh
FC = gfortran
LD = gfortran
FFLAGS = -Wall -O3 -march=native -m64 -I"${MKLROOT}/include"
NPY_DISTUTILS_APPEND_FLAGS = 1

factor:
	$(FC) $(FFLAGS) -c class_factor.f90

block_LU: class_factor.o
	python3 -m numpy.f2py --f90flags="-Wall -O3 -march=native -m64  -I"${MKLROOT}/include"" -c -m block_LU class_factor.o block_LU.f90

class_PARDISO:
	$(FC) $(FFLAGS) -c class_PARDISO.f90

mod_PARDISO: class_PARDISO.o
	python3 -m numpy.f2py --f90flags="-Wall -O3 -march=native -m64  -I"${MKLROOT}/include"" -c -m mod_PARDISO class_PARDISO.o mod_PARDISO.f90
