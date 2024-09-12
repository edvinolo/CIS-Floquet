# CIS-Floquet
For finding eigenvalues of Floquet Hamiltonian with the Hamiltonian from a CIS basis

## How to run:
  ```python3 main.py path/to/input_file```


## How to use the fortran+MKL routines for block-LU factorization:
  
  -Set the oneapi environment variables: 
    ```source /path/to/intel/oneapi/setvars.sh```
  
  -Make sure that the compiler and linker flags are properly set in the Makefile,
    use this link to get the correct flags: 
    https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html \
    Make sure to select dynamic linking.
  
  -Compile factor class and interface the block_LU module with python:
    
    cd src
    
    make factor
    
    make block_LU
    
    cd ..
  
  
  -Set use_fortran = t in input file
