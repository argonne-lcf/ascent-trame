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
      subroutine nek_ascent_load_new_data()
      call load_fld("field.f00001")
      call lpm_init(0)
      end subroutine nek_ascent_load_new_data
c-----------------------------------------------------------------------