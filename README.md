# CIS-Floquet
For finding eigenvalues of Floquet Hamiltonian with the Hamiltonian from a CIS basis

## How to run:
  ```python3 main.py path/to/input_file```


## How to use the fortran+MKL routines for block-LU factorization:
  
- Set the oneapi environment variables: 
    ```source /path/to/intel/oneapi/setvars.sh```
- Make sure that the compiler and linker flags are properly set in the Makefile and environment variables,
    use this link to get the correct flags: 
    https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html \
    - Select the appropriate MKL version
    - Select OpenMP threading (have not tried if TBB works)
    - Using dynamic linking is the easiest.
    - Make sure to select the appropriate interface layer, (check the size of integer variables in Fortran, e.g. sizeof(1)==4 means 32-bit integers)
  
- Add the suggested link line to the LDFLAGS environment variable eg:
  
    ```export LDFLAGS="-m64  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"```
    
- Add the compiler options to the Makefile (if they differ from what's already in there)
  
- Compile factor class and interface the block_LU module with Python:
   ``` 
    cd src
    
    make block_LU
    
    cd ..
  ```
  
- Set use_fortran = t in the input file

  ### Special instructions for running on kestrel
  The MKL version installed on kestrel is outdated, and the above steps need to be modified slightly for things to work
  - Select the 2021 MKL version and ```Single Dynamic Library``` in the link line advisor 
  - Add ```-shared``` to LDFLAGS (this is not related to mkl I think)
  - LDFLAGS should now look like this:
    ```
    -shared -m64  -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl
    ```
  - Set the LD_PRELOAD environment variable:
    ```
    export LD_PRELOAD=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so
    ```
  - Set the interface layer that the single dynamic library should use:
    ```
    export MKL_INTERFACE_LAYER=GNU,LP64
    ```
    See this link for more info: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-2/dynamic-select-the-interface-and-threading-layer.html
  - Optional: change the threading layer. Default uses intel threading, but you can change to GNU with
    ```
    export MKL_THREADING_LAYER=GNU
    ```
