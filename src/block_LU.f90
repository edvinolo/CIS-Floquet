module block_LU
    use class_factor
    implicit none
    private 
    type(factor),dimension(:),allocatable :: factors
    integer :: N_blocks
    !double complex, dimension(:,:),allocatable :: V_mod

    public :: block_LU_setup,block_LU_solve

contains
    subroutine block_LU_setup(H,V,shift,m_omega,N_floquet_blocks,N_elements)
        double complex, dimension(:,:), intent(in) :: H,V
        double complex, intent(in) :: shift
        double precision,dimension(:),intent(in) :: m_omega
        integer, intent(in) :: N_floquet_blocks,N_elements

        double complex, dimension(:), allocatable :: w
        double complex, dimension(:,:), allocatable :: w_mat
        integer :: i,j

        N_blocks = N_floquet_blocks
        
        if (.not.allocated(factors)) then
            allocate(factors(N_blocks))
        end if

        !if (.not.allocated(V_mod)) then
        !    allocate(V_mod(N_elements,N_elements))
        !    V_mod = V
        !end if
        

        allocate(w(N_elements),w_mat(N_elements,N_elements))
        
        call factors(1)%setup(N_elements,H)
        do i = 1,N_elements
            factors(1)%mat(i,i) = factors(1)%mat(i,i) + (m_omega(1)-shift)
        end do
        call factors(1)%factorize()

        do i = 1,N_blocks-1
            write(6,*) 'Doing block ', i+1,'/',N_floquet_blocks
            call factors(i+1)%setup(N_elements,H)
            w_mat = V
            call factors(i)%solve_mat(w_mat)
            call zgemm('N','N',N_elements,N_elements,N_elements,dcmplx(-1.d0,0.d0),V,N_elements,w_mat,N_elements,&
                        dcmplx(1.d0,0.d0),factors(i+1)%mat,N_elements)
            do j = 1,N_elements
                !call zcopy(N_elements,V(:,j),1,w,1)
                !call factors(i)%solve(w)
                !call zgemv('N',N_elements,N_elements,dcmplx(-1.d0,0.d0),V,N_elements,w,1,&
                !            dcmplx(1.d0,0.d0),factors(i+1)%mat(:,j),1)
                factors(i+1)%mat(j,j) = factors(i+1)%mat(j,j) + (m_omega(i+1) - shift)
            end do
            call factors(i+1)%factorize()
        end do

    end subroutine block_LU_setup

    subroutine block_LU_solve(b,result,V,N_vector)
        !Now python supplies V on every call. Check if speed/memory usage is affected by storing a copy of V in the module
        integer :: N_vector
        double complex, dimension(N_vector), intent(in) :: b
        double complex, dimension(N_vector), intent(inout) :: result
        double complex, dimension(:,:), intent(in) :: V


        double complex, dimension(:,:), allocatable :: y
        integer :: i,N_elements

        result = dcmplx(0.d0,0.d0)
        N_elements = factors(1)%N_elements

        allocate(y(N_elements,N_blocks))

        call zcopy(N_elements,b(1:N_elements),1,y(:,1),1)
        call factors(1)%solve(y(:,1))

        do i=2,N_blocks
            !b[i*self.N_elements:(i+1)*self.N_elements]-self.V@y[:,i-1]
            call zcopy(N_elements,b(1+(i-1)*N_elements:i*N_elements),1,y(:,i),1)
            call zgemv('N',N_elements,N_elements,dcmplx(-1.d0,0.d0),V,N_elements,y(:,i-1),&
                        1,dcmplx(1.d0,0.d0),y(:,i),1)
            call factors(i)%solve(y(:,i))
        end do

        !result[(self.N_floquet_blocks-1)*self.N_elements:] = y[:,self.N_floquet_blocks-1]
        call zcopy(N_elements,y(:,N_blocks),1,result(1+(N_blocks-1)*N_elements:),1)

        !for i in reversed(range(self.N_floquet_blocks-1)):
        !    #w = self.lgmres(self.Blocks[i],self.Z@result[(i+1)*self.N_elements:(i+2)*self.N_elements])
        !    w = self.factors[i].solve(self.V@result[(i+1)*self.N_elements:(i+2)*self.N_elements])
        !    result[i*self.N_elements:(i+1)*self.N_elements] = y[:,i]-w
        do i=N_blocks-1,1,-1
            call zgemv('N',N_elements,N_elements,dcmplx(1.d0,0.d0),V,N_elements,result(1+i*N_elements:(i+1)*N_elements),1,&
                        dcmplx(0.d0,0.d0),result(1+(i-1)*N_elements: i*N_elements),1)
            call factors(i)%solve(result(1+(i-1)*N_elements: i*N_elements))
            call zaxpy(N_elements,dcmplx(-1.d0,0.d0),y(:,i),1,result(1 + (i-1)*N_elements: i*N_elements),1)
            call zdscal(N_elements,-1.d0,result(1 + (i-1)*N_elements: i*N_elements),1)
        end do

    end subroutine block_LU_solve
end module block_LU