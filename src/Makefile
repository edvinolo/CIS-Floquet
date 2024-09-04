SHELL = /bin/sh
FC = gfortran
LD = gfortran
FFLAGS = -Wall -O3 -march=native -m64 -I"${MKLROOT}/include"
LDFLAGS = -m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
NPY_DISTUTILS_APPEND_FLAGS = 1

factor:
	$(FC) $(FFLAGS) -c class_factor.f90

block_LU:
	f2py --f90flags="-Wall -O3 -march=native -m64  -I"${MKLROOT}/include"" -c -m block_LU class_factor.o block_LU.f90
