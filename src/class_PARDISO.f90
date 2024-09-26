include 'mkl_pardiso.f90'
module class_PARDISO
    use mkl_pardiso
    implicit none

    type,public :: PARDISO_solver
        type(MKL_PARDISO_HANDLE), dimension(:), allocatable  :: pt !Pointers used by PARDISO
        integer, dimension(:), allocatable :: iparm !PARDISO parameter array
        integer :: maxfct, mnum, mtype, phase, n, nrhs, error, msglvl, nnz !PARDISO integer parameters
        integer, dimension(1) :: idum !dummy integer array
        double complex, dimension(1) :: ddum !dummy complex array

        contains
        procedure :: setup => PARDISO_setup
        procedure :: solve => PARDISO_solve
        procedure :: cleanup => PARDISO_cleanup
    end type PARDISO_solver

contains
    subroutine PARDISO_setup(this,n,nnz,a,ia,ja)
        class(PARDISO_solver), intent(inout) :: this
        integer, intent(in) :: n,nnz
        double complex, dimension(0:nnz-1), intent(in) :: A
        integer, dimension(0:n), intent(in) :: ia
        integer, dimension(0:nnz-1), intent(in) :: ja

        integer :: i !loop index


        this%n = n
        this%nnz = nnz
        this%nrhs = 1 !Hard code this for now, could change later

        this%error  = 0 !initialize error flag
        this%msglvl = 0 !print no statistical information
        this%mtype  = 13 !complex nonsymmetric
        this%maxfct = 1 !Maximum number of factors
        this%mnum = 1 !Number of matrix to solve


        !.. Initialize the internal solver memory pointer. This is only
        ! necessary for the FIRST call of the PARDISO solver.
        allocate (this%pt(64))
        do i = 1, 64
            this%pt(i)%dummy =  0
        end do

        !..
        !.. Set up PARDISO control parameters
        !..
        allocate(this%iparm(64))
        do i = 1, 64
            this%iparm(i) = 0
        end do

        !Please see the intel MKL online documentation for PARDISO for the complete meaning of all params
        this%iparm(1) = 1 !Do not use defaults for all iparm
        this%iparm(2) = 3 !Parallel fill in reordering
        this%iparm(10) = 13 !Perturbation of small pivots by 1e-13
        this%iparm(11) = 1 !Enable scaling vectors
        this%iparm(13) = 1 !Enable weighted matching
        this%iparm(24) = 0 !Change to 10 for two-level factorization (must disable param 11 and 13 if used)
        this%iparm(27) = 0 !Enable matrix checker, could be useful for debugging, but probably disable later
        this%iparm(35) = 1 !Use zero-based indexing, since arrays are comming from Python

        this%phase = 12 !Analysis + numerical factorization
        call pardiso (this%pt, this%maxfct, this%mnum, this%mtype, this%phase, this%n, a, ia, ja, &
              this%idum, this%nrhs, this%iparm, this%msglvl, this%ddum, this%ddum, this%error)

        if (this%error.ne.0) then
            write(6,*) 'PARDISO setup error: ', this%error
        end if

    end subroutine PARDISO_setup

    subroutine PARDISO_solve(this,a,ia,ja,x,b)
        class(PARDISO_solver), intent(inout) :: this
        double complex, dimension(0:this%nnz-1), intent(in) :: a
        integer, dimension(0:this%n), intent(in) :: ia
        integer, dimension(0:this%nnz-1), intent(in) :: ja
        double complex, dimension(0:this%n-1), intent(inout) :: x,b

        this%phase = 33 !Compute solution
        call pardiso (this%pt, this%maxfct, this%mnum, this%mtype, this%phase, this%n, a, ia, ja, &
              this%idum, this%nrhs, this%iparm, this%msglvl, b, x, this%error)

        if (this%error.ne.0) then
            write(6,*) 'PARDISO solve error: ', this%error
        end if

    end subroutine PARDISO_solve

    subroutine PARDISO_cleanup(this)
        class(PARDISO_solver), intent(inout) :: this

        this%phase = -1 !Release internal memory for solver
        call pardiso (this%pt, this%maxfct, this%mnum, this%mtype, this%phase, this%n, this%ddum, this%idum, this%idum, &
              this%idum, this%nrhs, this%iparm, this%msglvl, this%ddum, this%ddum, this%error)

        deallocate(this%pt,this%iparm)

    end subroutine PARDISO_cleanup
end module class_PARDISO