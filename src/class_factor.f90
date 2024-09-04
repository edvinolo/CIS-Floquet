module class_factor
    implicit none
    type,public :: factor
    double complex, dimension(:,:), allocatable :: mat
    integer, dimension(:), allocatable :: ipiv
    integer :: N_elements
    integer :: factor_info,solve_info
    contains
    procedure :: setup => factor_setup
    procedure :: factorize => factor_factorize
    procedure :: solve => factor_solve
    procedure :: solve_mat => factor_solve_mat
    end type factor
contains

    subroutine factor_setup(this,N_elements,H)
        class(factor), intent(inout) :: this
        integer, intent(in) :: N_elements
        double complex, dimension(:,:),intent(in) :: H

        if (.not.allocated(this%mat)) then
            allocate(this%mat(N_elements,N_elements))
        end if
        
        this%mat = H
        this%N_elements = N_elements

    end subroutine factor_setup

    subroutine factor_factorize(this)
        class(factor), intent(inout) :: this

        if (.not.allocated(this%ipiv)) then
            allocate(this%ipiv(this%N_elements))
        end if

        call zgetrf(this%N_elements,this%N_elements,this%mat,this%N_elements,this%ipiv,this%factor_info)
        
        if (this%factor_info.ne.0) then
            write(6,*) 'Factorization returned with info: ', this%factor_info
            stop
        end if
    end subroutine factor_factorize

    subroutine factor_solve(this,b)
        class(factor), intent(in) :: this
        double complex, dimension(:),intent(inout) :: b

        call zgetrs('N',this%N_elements,1,this%mat,this%N_elements,this%ipiv,b,this%N_elements,this%solve_info)
        if (this%solve_info.ne.0) then
            write(6,*) 'Solve returned with info: ', this%solve_info
            stop
        end if

    end subroutine factor_solve

    subroutine factor_solve_mat(this,b_mat)
        class(factor), intent(in) :: this
        double complex, dimension(:,:),intent(inout) :: b_mat

        call zgetrs('N',this%N_elements,this%N_elements,this%mat,this%N_elements,this%ipiv,b_mat,this%N_elements,this%solve_info)
        if (this%solve_info.ne.0) then
            write(6,*) 'Solve returned with info: ', this%solve_info
            stop
        end if

    end subroutine factor_solve_mat
    
end module class_factor