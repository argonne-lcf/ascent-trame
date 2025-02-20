c-----------------------------------------------------------------------
      subroutine nek_ascent_setup()
      include 'SIZE'
      include 'TOTAL'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      real nek_dt
      common /nekascent/ nek_dt
      nek_dt = 0.0

      call ascent_setup(nekcomm)
      end
c-----------------------------------------------------------------------
      subroutine nek_ascent_update()
      include 'SIZE'
      include 'TOTAL'
      include 'LPM'
      common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal

      call ascent_update(istep, time, ndim, nelt, nelv, n, lr, wdsize,
     &       lx1, ly1, lz1, xm1, ym1, zm1,
     &       lx2, ly2, lz2, xm2, ym2, zm2,
     &       vx, vy, vz,
     &       jx, jy, jz, jv0, rpart)
      end
c-----------------------------------------------------------------------
      subroutine nek_ascent_finalize()
      call ascent_finalize()
      end subroutine nek_ascent_finalize
c-----------------------------------------------------------------------
      subroutine nek_ascent_increase_dt()
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      print *, 'Increasing DT'
      print *, 'Before DT', param(12)
      param(12) = param(12) + 0.00001
      print *, 'After DT', param(12)
      end subroutine nek_ascent_increase_dt
c-----------------------------------------------------------------------
      subroutine nek_ascent_decrease_dt()
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      print *, 'Decreasing DT'
      print *, 'Before DT', param(12)
      param(12) = param(12) - 0.00001
      print *, 'After DT', param(12)
      end subroutine nek_ascent_decrease_dt
c-----------------------------------------------------------------------
      subroutine nek_ascent_get_dt()
      INCLUDE 'SIZE'
      INCLUDE 'INPUT'
      print *, 'Current DT', param(12)
      end subroutine nek_ascent_get_dt
c-----------------------------------------------------------------------
      subroutine lpm_user_particle_distribution()
      include 'SIZE'
      include 'TOTAL'
      include 'LPM'

c     remember each rank only has a localy copy      
      integer i,j, array_number
      real   square_spacing, sphere_diam
c     Box params
      real   Box_xmin, Box_ymin, Box_zmin
      real   Box_xrange, Box_yrange, Box_zrange
      integer nnn
      common /Box_param/Box_xrange, Box_yrange, Box_zrange

      rdx            =  0.1/40/5
      array_number   =  1
      x_spacing      =  0.02
      sphere_diam    =  0.01

      Box_xmin       = -0.04
      Box_ymin       =  0.00
      Box_zmin       = -0.04    


      open(unit=9999999,file="particles1.dat")
      read(9999999,*) nnn, xlow, xhigh, zlow, zhigh
      close(9999999)

      nn=0      
      do i = 1, array_number
         do j = 1, 1
            do k = 1,  array_number
               nn = nn + 1
               ibm_center(nn,1) = 0.
               ibm_center(nn,2) = 0. 
               ibm_center(nn,3) = 0.
               ibm_diam(nn)     = 2.0
               n_dh(nn)         = nnn ! determine n_markers
            enddo
         enddo
      enddo

      ! check
      if(nid.eq.0) then
         if (nn.ne.num_of_IBMpart) then
           write(6,'(A,2I4)')"IBM Particle number/array does not match"
     $      ,nn, num_of_IBMpart
            call exitt
         endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine lpm_user_marker_distribution
      include 'SIZE'
      include 'TOTAL'
      include 'LPM'
      
c     Box params
      real    Box_xmin, Box_ymin, Box_zmin
      real    Box_xrange, Box_yrange, Box_zrange

      real ai, bi,ci, di
      real nnn

      common /Box_param/Box_xrange, Box_yrange, Box_zrange

      integer seq_numbering
      real    r,h,n_dh_l

      hcyl = 0.05

      rpi = 4.*atan(1.0)
      seq_numbering = 1
      if(seq_numbering.eq.1) then
            ! ndef: if(nid <  ndef), n_IBMpart+1
            !       if(nid >= ndef), n_IBMpart+0
         if(nid.lt.ndef) nsi = nid * n_IBMpart
         if(nid.ge.ndef) nsi = ndef * (n_IBMpart + 1) + 
     $                        (nid - ndef) * n_IBMpart
         print*,"nid,nsi,ndef",nid,nsi,ndef
      endif

      ! assign Queen
      do nn=1, n_IBMpart            ! number of particles in current process
         rpart(jx + 0, nn) = ibm_center(nsi + nn, 1)
         rpart(jx + 1, nn) = ibm_center(nsi + nn, 2)
         rpart(jx + 2, nn) = ibm_center(nsi + nn, 3)
         r                 = ibm_diam  (nsi + nn ) / 2.0
         n_dh_l            = n_dh(nsi+nn) 
         h                 = 2 * r / n_dh_l

         n_l(nn) = int(2*rpi*r/h) * int(hcyl/h)
	 if(nid.eq.1) print*, "h,ndh", h, n_dh_l, n_l(nn)
         ! diameter and volume
         rpart(jdp,  nn)   =  2 * r
         rpart(jvol, nn)   =  pi * (r**2) * hcyl 
         ! for time lagging terms
         do j =0,2
            rpart(jx1 + j , nn) = rpart(jx + j, nn)
            rpart(jx2 + j , nn) = rpart(jx + j, nn)
            rpart(jx3 + j , nn) = rpart(jx + j, nn) 
         enddo

         if(nid.eq.0)write(6,2029) nid, nn, n_l(nn), 
     $   (rpart(jx+j,nn),j=0,2),rpart(jdp,nn),rpart(jvol,nn)  
 2029 format(I4,' Queen #',2I4,' xyz',3F8.3,',dp ='F8.3,', vol =',E12.4)
      enddo


      open(unit=9999999,file="particles1.dat")
      read(9999999,*) nnn, xlow, xhigh, zlow, zhigh
c      n_dh(1)         = nnn

!     Step 2 calculate the location for each lagrange point on a plane worker
      k = n_IBMpart  ! num_of_IBMpart: total
      do nn = 1, nnn
         r      = ibm_diam  (nsi + nn ) / 2.0d0
         n_dh_l = n_dh(nsi + nn) 
         h      = 2*r / n_dh_l

         xmean = ibm_center(nsi+nn, 1) 
         ymean = ibm_center(nsi+nn, 2)
         zmean = ibm_center(nsi+nn, 3) 
         read(9999999,*) IDP, ai, bi, ci, di

               k = k + 1
               rtheta = real(i)/real(int(2*rpi*r/h)) * 2 * rpi              
               rpart(jx + 0, k) = bi*100*0.3!
               rpart(jx + 1, k) = ci*100*0.3!  ymean + j * h
               rpart(jx + 2, k) = di*100*0.3! zmean + r * sin(rtheta)
               rpart(jvol, k)   =  ai*ai*ai*100*0.3*100*0.3*100*0.3
               rpart(jdp,  k)   =  ai*10*0.3*8 
               do jj =0,2
                  rpart(jx1 + jj , k) = rpart(jx + jj, k)
                  rpart(jx2 + jj , k) = rpart(jx + jj, k)
                  rpart(jx3 + jj , k) = rpart(jx + jj, k)
               enddo
               if(nid.eq.0)write(6,'(A,2I4,3f8.4)')'Worker #',nid,k,
     $              rpart(jx+0,k),rpart(jx+1,k),rpart(jx+2,k)

      enddo


      close(9999999)


      ! update number of points 
      nwe = k
      ! check
      do nn = 1, n_IBMpart 
         if (n_l_max .lt. n_l(nn)) n_l_max=n_l(nn)
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine nek_ascent_load_new_data()
      call lpm_user_particle_distribution
      call load_fld("field.f00001")
      call lpm_init(0)
      end subroutine nek_ascent_load_new_data
c-----------------------------------------------------------------------