module mod_PARDISO
    use class_PARDISO
    implicit none
    private
    type(PARDISO_solver) :: solver

    public :: setup, solve, cleanup

    contains
        subroutine setup(n,nnz,a,ia,ja)
            integer, intent(in) :: n,nnz
            double complex, dimension(0:nnz-1), intent(in) :: a
            integer, dimension(0:n), intent(in) :: ia
            integer, dimension(0:nnz-1), intent(in) :: ja

            !Think about adding support for low-rank updates, since sparsity pattern never really changes.
            call solver%setup(n,nnz,a,ia,ja)
        end subroutine setup

        subroutine solve(n,nnz,a,ia,ja,x,b)
            integer, intent(in) :: n,nnz
            double complex, dimension(0:nnz-1), intent(in) :: a
            integer, dimension(0:n), intent(in) :: ia
            integer, dimension(0:nnz-1), intent(in) :: ja
            double complex, dimension(0:n-1), intent(inout) :: x,b

            call solver%solve(a,ia,ja,x,b)
        end subroutine solve

        subroutine cleanup()
            call solver%cleanup()
        end subroutine cleanup
end module mod_PARDISO