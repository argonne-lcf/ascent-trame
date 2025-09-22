#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <string>
#include <cstring>


#include "lbm_mpi.hpp"
#include "paraview_sim.hpp"

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, LbmDQ::LatticeType lattice_type, bool output_vorticity, bool output_velocity);
void exportSimulationStateToVTS(LbmDQ* lbm, const char* filename, double dt, double dx, double physical_density, uint32_t time_steps);
void exportVelocityDiagnostics(int t, int rank, int num_ranks, LbmDQ* lbm, double dt, double dx, double physical_density, uint32_t time_steps);

// global vars for LBM and Barriers
std::vector<Barrier*> barriers;
LbmDQ *lbm;

int main(int argc, char **argv) {
    int rc, rank, num_ranks;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (rc != 0)
    {
	fprintf(stderr, "Error initializing MPI");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    uint32_t dim_x = 400;
    uint32_t dim_y = 128;
    uint32_t dim_z = 128;
    uint32_t time_steps = 10000;

    LbmDQ::LatticeType lattice_type;
    bool model_specified = false;
    bool output_vorticity = false;
    bool output_velocity = false;

    // Check for compile-time defaults (for Makefile compatibility)
#ifdef OUTPUT_VORTICITY
    output_vorticity = (OUTPUT_VORTICITY != 0);
#endif
#ifdef OUTPUT_VELOCITY
    output_velocity = (OUTPUT_VELOCITY != 0);
#endif

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
	if (strcmp(argv[i], "--d3q15") == 0) {
	    lattice_type = LbmDQ::D3Q15;
	    model_specified = true;
	}
	else if (strcmp(argv[i], "--d3q19") == 0) {
	    lattice_type = LbmDQ::D3Q19;
	    model_specified = true;
	}
	else if (strcmp(argv[i], "--d3q27") == 0) {
	    lattice_type = LbmDQ::D3Q27;
	    model_specified = true;
	}
	else if (strcmp(argv[i], "--output-vorticity") == 0) {
	    output_vorticity = true;
	}
	else if (strcmp(argv[i], "--output-velocity") == 0) {
	    output_velocity = true;
	}
	else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
	    if (rank == 0) {
		printf("Usage: %s [OPTIONS]\n", argv[0]);
		printf("Options:\n");
		printf("  --d3q15              Use D3Q15 lattice model\n");
		printf("  --d3q19              Use D3Q19 lattice model\n");
		printf("  --d3q27              Use D3Q27 lattice model\n");
		printf("  --output-vorticity   Enable vorticity output to VTS files\n");
		printf("  --output-velocity    Enable velocity vector output to VTS files\n");
		printf("  --help, -h           Show this help message\n");
	    }
	    MPI_Finalize();
	    return 0;
	}
    }

    if (!model_specified) {
        if (rank == 0) {
	    fprintf(stderr, "Error: Lattice model must be specified. Use --d3q15, --d3q19, or --d3q27");
 	}
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "\nLBM-CFD> running with " << num_ranks << " processes" << std::endl;
        std::cout << "LBM-CFD> resolution=" << dim_x << "x" << dim_y << "x" << dim_z << ", time steps=" << time_steps << std::endl;
        std::cout << "LBM-CFD> using " << (lattice_type == LbmDQ::D3Q15 ? "D3Q15" : (lattice_type == LbmDQ::D3Q19 ? "D3Q19" : "D3Q27")) << " lattice model" << std::endl;
	std::cout << "LBM-CFD> output options: vorticity=" << (output_vorticity ? "enabled" : "disabled")
                  << ", velocity=" << (output_velocity ? "enabled" : "disabled") << std::endl;
    }

    // Run simulation
    runLbmCfdSimulation(rank, num_ranks, dim_x, dim_y, dim_z, time_steps, lattice_type, output_vorticity, output_velocity);
    MPI_Finalize();
    return 0;
}

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, uint32_t time_steps, LbmDQ::LatticeType lattice_type, bool output_vorticity, bool output_velocity)
{
    // simulate corn syrup at 25 C in a 2 m pipe, moving 0.25 m/s
    double physical_density = 1380.0;     // kg/m^3
    double physical_speed = 0.25;         // m/s
    double physical_length = 2.0;         // m
    double physical_viscosity = 1.3806;   // Pa s
    double kinematic_viscosity = physical_viscosity / physical_density;

    double reynolds_number = (physical_density * physical_speed * physical_length) / physical_viscosity;

    double dx = physical_length / (double)dim_x;

    // Calculate time step based on CFL condition
    double cfl_max = 1.0;
    double dt_cfl = cfl_max * dx / physical_speed;

    // Calculate diffusion stability limit
    double dt_diffusion = 0.5 * dx * dx / kinematic_viscosity;

    // Use the more restrictive (smaller) time step
    double dt = std::min(dt_cfl, dt_diffusion);

    // Calculate lattice parameters
    double lattice_cs = 1.0 / sqrt(3.0);  // Lattice speed of sound
    double lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
    double lattice_speed = physical_speed * dt / dx;
    double mach_number = lattice_speed / lattice_cs;

    // Safety checks and corrections
    if (lattice_viscosity < 0.005) {
        double correction = 0.005 / lattice_viscosity;    
        dt *= correction;
        lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
        lattice_speed = physical_speed * dt / dx;
        mach_number = lattice_speed / lattice_cs;
        if (rank == 0) {
            printf("[INFO] Reduced dt by factor %.3f for viscosity stability\n", correction);
        }
    }

    if (mach_number > 0.1) {
        double correction = 0.1 / mach_number;
        dt *= correction;
        lattice_viscosity = kinematic_viscosity * dt / (dx * dx);
        lattice_speed = physical_speed * dt / dx;
        mach_number = lattice_speed / lattice_cs;
        if (rank == 0) {
            printf("[INFO] Reduced dt by factor %.3f for Mach number stability\n", correction);
        }
    }

    // Calculate total simulated time
    double total_simulated_time = time_steps * dt;

    // Output simulation properties
    if (rank == 0)
    {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "LBM-CFD> TIME STEP CALCULATION:" << std::endl;
        std::cout << "  Grid spacing (dx): " << dx << " m" << std::endl;
        std::cout << "  CFL time step: " << dt_cfl << " s" << std::endl;
        std::cout << "  Diffusion time step: " << dt_diffusion << " s" << std::endl;
        std::cout << "  Limiting factor: " << ((dt_cfl < dt_diffusion) ? "CFL" : "Diffusion") << std::endl;
        
        std::cout << "LBM-CFD> PHYSICAL PARAMETERS:" << std::endl;
        std::cout << "  density: " << physical_density << " kg/m^3" << std::endl;
        std::cout << "  dynamic viscosity: " << physical_viscosity << " Pa s" << std::endl;
        std::cout << "  speed: " << physical_speed << " m/s" << std::endl;
        std::cout << "  length: " << physical_length << " m" << std::endl;
        std::cout << "  Reynolds number: " << reynolds_number << std::endl;
        
        std::cout << "LBM-CFD> FINAL SIM PARAMETERS:" << std::endl;
        std::cout << "  Mach number: " << mach_number << std::endl;
        std::cout << "  CFL number: " << cfl_max << std::endl;
        std::cout << "  Total simulated time: " << total_simulated_time << " s" << std::endl;
        
        // Stability warnings
        if (lattice_viscosity < 0.005) {
            std::cout << "*** WARNING: Low lattice viscosity! Simulation may be unstable\n";
        }
        if (mach_number > 0.1) {
            std::cout << "*** WARNING: High Mach number! Simulation may be unstable\n";
        }
        if (lattice_speed > 0.1) {
            std::cout << "*** WARNING: High lattice speed! Simulation may be unstable\n";
        }

    }

    // create LBM object
    lbm = new LbmDQ(dim_x, dim_y, dim_z, dt / dx, rank, num_ranks, lattice_type);

    // BARRIERS: Diamond-shaped (square rotated 45°) bluff body barriers
    barriers.clear();

    // Configuration parameters
    int diamond_x_center = std::max(10, (int)(dim_x / 6));  // Position at 1/6 of domain
    int diamond_y_center = dim_y / 2;                       // Center vertically
    int diamond_size = std::max(5, (int)(dim_y / 8));       // Size based on domain height

    // Optional: Create multiple diamonds for more complex flow
    bool use_multiple_diamonds = true; // set false for one large diamond

    if (use_multiple_diamonds) {
        // Create three diamond obstacles at different positions
        std::vector<std::pair<int, int>> diamond_centers = {
            {diamond_x_center, diamond_y_center},
            {diamond_x_center + (int)(dim_x / 3), diamond_y_center - (int)(dim_y / 6)},
            {diamond_x_center + (int)(2 * dim_x / 3), diamond_y_center + (int)(dim_y / 6)}
        };

        for (const auto& center : diamond_centers) {
            int cx = center.first;
            int cy = center.second;

            // Create diamond shape (rotated square)
            for (int k = 0; k < (int)dim_z; ++k) {
                for (int j = 0; j < (int)dim_y; ++j) {
                    for (int i = 0; i < (int)dim_x; ++i) {
                        // Diamond shape condition: |x-cx| + |y-cy| <= size
                        int dx = abs(i - cx);
                        int dy = abs(j - cy);

                        if (dx + dy <= diamond_size && i >= 0 && i < (int)dim_x &&
                            j >= 0 && j < (int)dim_y && k >= 0 && k < (int)dim_z) {
                            barriers.push_back(new Barrier3D(i, i, j, j, k, k));
                        }
                    }
                }
            }
        }
    } else {
        // Single larger diamond for simpler but still interesting flow
        int large_diamond_size = std::max(8, (int)(dim_y / 5));

        // Create single diamond shape
        for (int k = 0; k < (int)dim_z; ++k) {
            for (int j = 0; j < (int)dim_y; ++j) {
                for (int i = 0; i < (int)dim_x; ++i) {
                    // Diamond shape condition: |x-cx| + |y-cy| <= size
                    int dx = abs(i - diamond_x_center);
                    int dy = abs(j - diamond_y_center);

                    if (dx + dy <= large_diamond_size && i >= 0 && i < (int)dim_x &&
                        j >= 0 && j < (int)dim_y && k >= 0 && k < (int)dim_z) {
                        barriers.push_back(new Barrier3D(i, i, j, j, k, k));
                    }
                }
            }
        }
    }
 
    lbm->initBarrier(barriers);
    lbm->initFluid(physical_speed);
    lbm->checkGuards();

    // sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // run simulation
    int t;
    int output_count = 0;
    double output_frequency = 0.5;  // Output simulation time step progress at specified interval
    double next_output_time = 0.0;
    uint8_t stable, all_stable = 0;

    // Timing variables
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> collide_time(0), stream_time(0), exchange_time(0);
    std::chrono::duration<double> total_iteration_time(0);

    for (t = 0; t < time_steps; t++)
    {
        // enforce inlet boundary at every step
	lbm->updateFluid(physical_speed);
	// Output data at regular intervals
        double current_time = t * dt;  // Current simulation time
        if (current_time >= next_output_time)
	{
            if (rank == 0)
            {
	        std::cout << std::fixed << std::setprecision(3) << "LBM-CFD> time: " << current_time << " / " <<
                             total_simulated_time << " s, time step: " << t << " / " << time_steps << std::endl;
	    }
            stable = lbm->checkStability();
            MPI_Reduce(&stable, &all_stable, 1, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!all_stable && rank == 0)
            {
                fprintf(stderr, "LBM-CFD> Warning: simulation has become unstable (more time steps needed)");
            } 
            output_count++;
            next_output_time = output_count * output_frequency;
        }

	// Export vorticity for visualization
	if (output_vorticity) {
            printSimulationDiagnostics(t, rank, lbm, dt, dx, physical_density, time_steps);
        }
        
        // Export velocity vectors for streamline visualization
        if (output_velocity) {
            exportVelocityDiagnostics(t, rank, num_ranks, lbm, dt, dx, physical_density, time_steps);
        }

	// Time the entire iteration
        start_time = std::chrono::high_resolution_clock::now();

        // Time collide step
        auto collide_start = std::chrono::high_resolution_clock::now();

	lbm->collide(lattice_viscosity, t);

	auto collide_end = std::chrono::high_resolution_clock::now();
        collide_time += std::chrono::duration_cast<std::chrono::duration<double>>(collide_end - collide_start);
       
	// Time stream step
        auto stream_start = std::chrono::high_resolution_clock::now();
	
	lbm->stream();

	auto stream_end = std::chrono::high_resolution_clock::now();
        stream_time += std::chrono::duration_cast<std::chrono::duration<double>>(stream_end - stream_start);

	// Time exchangeBoundaries step
        auto exchange_start = std::chrono::high_resolution_clock::now();

        lbm->exchangeBoundaries();

	auto exchange_end = std::chrono::high_resolution_clock::now();
        exchange_time += std::chrono::duration_cast<std::chrono::duration<double>>(exchange_end - exchange_start);

        end_time = std::chrono::high_resolution_clock::now();
        total_iteration_time += std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
    }

    // Print final timing summary
    if (rank == 0)
    {
        std::cout << "\n================== FINAL TIMING SUMMARY ==================" << std::endl;
        std::cout << std::left << std::setw(15) << "Metric" << "\t" 
                  << std::right << std::setw(12) << "Value" << "\t" 
                  << std::left << std::setw(10) << "Unit" << "\t" 
                  << std::right << std::setw(10) << "Percentage" << std::endl;
        std::cout << std::left << std::setw(15) << "Total steps" << "\t" 
                  << std::right << std::setw(12) << time_steps << "\t" 
                  << std::left << std::setw(10) << "steps" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Collide" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << collide_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (collide_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Stream" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << stream_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (stream_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Exchange" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << exchange_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << std::setprecision(2) << (exchange_time.count() / total_iteration_time.count()) * 100 << "%" << std::endl;
        std::cout << std::left << std::setw(15) << "Total" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << total_iteration_time.count() << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Avg per step" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(6) << total_iteration_time.count() / time_steps << "\t" 
                  << std::left << std::setw(10) << "seconds" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << std::left << std::setw(15) << "Steps/sec" << "\t" 
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << time_steps / total_iteration_time.count() << "\t" 
                  << std::left << std::setw(10) << "steps/sec" << "\t" 
                  << std::right << std::setw(10) << "" << std::endl;
        std::cout << "==========================================================" << std::endl;
    }

    // Clean up    
    delete lbm;
    // Delete all Barrier* objects to prevent memory leaks
    for (Barrier* b : barriers) {
        delete b;
    }
    barriers.clear();
}
