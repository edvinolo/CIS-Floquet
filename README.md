# CIS-Floquet
For finding eigenvalues of Floquet Hamiltonian with the Hamiltonian from a CIS basis

## How to run:
  ```python3 main.py path/to/input_file```


## How to use the fortran+MKL routines for block-LU factorization:
  
  -Set the oneapi environment variable: 
    ```source /path/to/intel/oneapi/setvars.sh```
  
  -Compile factor class and interface block_LU module with python:
    
    cd src
    
    make factor
    
    make block_LU
    
    cd ..
  
  
  -Set use_fortran = f in input file
