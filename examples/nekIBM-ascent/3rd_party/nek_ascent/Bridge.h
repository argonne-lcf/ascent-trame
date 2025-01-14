#ifndef NEK5000_BRIDGE_H
#define NEK5000_BRIDGE_H

#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <iostream>
#include <mpi.h>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef UPCASE
#define FORTRAN_NAME(low, up) up
#else
#ifdef UNDERSCORE
#define FORTRAN_NAME(low, up) low##_
#else
#define FORTRAN_NAME(low, up) low
#endif
#endif

#define ascent_setup FORTRAN_NAME(ascent_setup, ASCENT_SETUP)
#define ascent_update FORTRAN_NAME(ascent_update, ASCENT_UPDATE)
#define ascent_finalize FORTRAN_NAME(ascent_finalize, ASCENT_FINALIZE)

    static ascent::Ascent mAscent;

    // Accessible to Fortran
    void ascent_setup(MPI_Comm *comm);
    void ascent_update(int *istep, double *time, int *ndim, int *nelt, int *nelv, int *n, int *lr, int *wdsize,
                       int *lx1, int *ly1, int *lz1, double *xm1, double *ym1, double *zm1,
                       int *lx2, int *ly2, int *lz2, double *xm2, double *ym2, double *zm2,
                       double *vx, double *vy, double *vz,
                       int *jx, int *jy, int *jz, int *jv0, double *rpart);
    void ascent_finalize();

    // Callback registration
    void register_void_callback(std::string callback_name, void (*callback_function)(conduit::Node &, conduit::Node &));
    void register_bool_callback(std::string callback_name, bool (*callback_function)(void));

    // Callbacks
    void get_dt(conduit::Node &params, conduit::Node &output);
    void increase_dt(conduit::Node &params, conduit::Node &output);
    void decrease_dt(conduit::Node &params, conduit::Node &output);
    void reduce_particles(conduit::Node &params, conduit::Node &output);
    void plot_bins(conduit::Node &params, conduit::Node &output);
    void load_new_data(conduit::Node &params, conduit::Node &output);

#ifdef __cplusplus
} // extern "C"
#endif

#endif