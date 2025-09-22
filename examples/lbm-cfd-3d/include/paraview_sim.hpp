#ifndef PARAVIEW_SIM_HPP
#define PARAVIEW_SIM_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mpi.h>

class LbmDQ;

// Forward declarations for velocity vector export functions
void getBest3DPartition(int num_ranks, int dim_x, int dim_y, int dim_z, int *n_x, int *n_y, int *n_z);
void gatherVelocityComponents(int rank, int num_ranks, LbmDQ* lbm, float* vx_global, float* vy_global, float* vz_global);
void sendVelocityToRank0(LbmDQ* lbm);

void exportSimulationStateToVTS(LbmDQ* lbm, const char* filename, double dt, double dx, double physical_density, uint32_t time_steps) {
    float* gathered = lbm->getGatheredVorticity();
    if (!gathered) {
        fprintf(stderr, "Error: gathered array is NULL!\n");
        return;
    }
    const char* scalar_name = "vorticity";
    double value_scale = 1.0;
    double speed_scale = dx / dt;
    if (std::string(scalar_name) == "speed_m_per_s") {
        value_scale = speed_scale;
    } else if (std::string(scalar_name) == "density") {
        value_scale = physical_density;
    } else if (std::string(scalar_name) == "vorticity") {
        value_scale = speed_scale / dx;
    }
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        return;
    }
    
    // Write VTS XML header
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", total_x-1, total_y-1, total_z-1);
    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", total_x-1, total_y-1, total_z-1);
    
    // Write points
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                fprintf(fp, "          %d %d %d\n", i, j, k);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");
    
    // Write point data
    fprintf(fp, "      <PointData>\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"%s\" format=\"ascii\">\n", scalar_name);
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                int idx = i + total_x * (j + total_y * k);
                fprintf(fp, "          %.6f\n", gathered[idx] * value_scale);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </PointData>\n");
    
    // Close XML structure
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </StructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    
    fclose(fp);
    printf("VTS export completed: %s\n", filename);
}

void exportVelocityVectorsToVTS(int rank, int num_ranks, LbmDQ* lbm, const char* filename, double dt, double dx, double physical_density, uint32_t time_steps) {
    // Get velocity component data from all ranks
    float* vx_data = nullptr;
    float* vy_data = nullptr;
    float* vz_data = nullptr;
    
    if (rank == 0) {
        // Allocate temporary arrays for gathering velocity components
        int total_size = lbm->getTotalDimX() * lbm->getTotalDimY() * lbm->getTotalDimZ();
        vx_data = new float[total_size];
        vy_data = new float[total_size];
        vz_data = new float[total_size];
        
        // Gather velocity components manually from all ranks
        gatherVelocityComponents(rank, num_ranks, lbm, vx_data, vy_data, vz_data);
    }
    
    if (rank != 0) {
        // Send local velocity data to rank 0
        sendVelocityToRank0(lbm);
        return;
    }
    
    // Continue with VTS export on rank 0
    double velocity_scale = dx / dt;  // Convert from lattice units to physical units
    
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s for writing\n", filename);
        delete[] vx_data;
        delete[] vy_data;
        delete[] vz_data;
        return;
    }
    
    // Write VTS XML header
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <StructuredGrid WholeExtent=\"0 %d 0 %d 0 %d\">\n", total_x-1, total_y-1, total_z-1);
    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n", total_x-1, total_y-1, total_z-1);
    
    // Write points
    fprintf(fp, "      <Points>\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                fprintf(fp, "          %d %d %d\n", i, j, k);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </Points>\n");
    
    // Write velocity vector data
    fprintf(fp, "      <PointData>\n");
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int k = 0; k < total_z; ++k) {
        for (int j = 0; j < total_y; ++j) {
            for (int i = 0; i < total_x; ++i) {
                int idx = i + total_x * (j + total_y * k);
                fprintf(fp, "          %.6f %.6f %.6f\n", 
                        vx_data[idx] * velocity_scale, 
                        vy_data[idx] * velocity_scale, 
                        vz_data[idx] * velocity_scale);
            }
        }
    }
    fprintf(fp, "        </DataArray>\n");
    fprintf(fp, "      </PointData>\n");
    
    // Close XML structure
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </StructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    
    fclose(fp);
    printf("VTS velocity export completed: %s\n", filename);
    
    // Clean up
    delete[] vx_data;
    delete[] vy_data;
    delete[] vz_data;
}

// Simple 3D partition algorithm (copied from LbmDQ implementation)
void getBest3DPartition(int num_ranks, int dim_x, int dim_y, int dim_z, int *n_x, int *n_y, int *n_z) {
    int best_x = 1, best_y = 1, best_z = num_ranks;
    int min_diff = num_ranks; // start with a large difference
    
    for (int x = 1; x <= std::min(num_ranks, dim_x); ++x) {
        if (num_ranks % x != 0) continue;
        int remaining = num_ranks / x;
        for (int y = 1; y <= std::min(remaining, dim_y); ++y) {
            if (remaining % y != 0) continue;
            int z = remaining / y;
            if (z > dim_z) continue;
            
            int diff = abs(x - y) + abs(y - z) + abs(z - x);
            if (diff < min_diff) {
                min_diff = diff;
                best_x = x;
                best_y = y;
                best_z = z;
            }
        }
    }
    
    *n_x = best_x;
    *n_y = best_y;
    *n_z = best_z;
}

void gatherVelocityComponents(int rank, int num_ranks, LbmDQ* lbm, float* vx_global, float* vy_global, float* vz_global) {
    
    // Get local velocity data
    float* vx_local = lbm->getVelocityX();
    float* vy_local = lbm->getVelocityY();
    float* vz_local = lbm->getVelocityZ();
    
    int local_x = lbm->getDimX();
    int local_y = lbm->getDimY();
    int local_z = lbm->getDimZ();
    int total_x = lbm->getTotalDimX();
    int total_y = lbm->getTotalDimY();
    int total_z = lbm->getTotalDimZ();
    
    // Calculate domain partitioning (using our own implementation)
    int n_x, n_y, n_z;
    getBest3DPartition(num_ranks, total_x, total_y, total_z, &n_x, &n_y, &n_z);
    
    int chunk_w = total_x / n_x;
    int chunk_h = total_y / n_y;
    int chunk_d = total_z / n_z;
    int extra_w = total_x % n_x;
    int extra_h = total_y % n_y;
    int extra_d = total_z % n_z;
    
    // Copy rank 0's own data
    int col = 0, row = 0, layer = 0;
    int offset_x = col * chunk_w + std::min<int>(col, extra_w);
    int offset_y = row * chunk_h + std::min<int>(row, extra_h);
    int offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
    
    for (int k = 0; k < local_z; ++k) {
        for (int j = 0; j < local_y; ++j) {
            for (int i = 0; i < local_x; ++i) {
                int local_idx = i + local_x * (j + local_y * k);
                int global_idx = (offset_x + i) + total_x * ((offset_y + j) + total_y * (offset_z + k));
                vx_global[global_idx] = vx_local[local_idx];
                vy_global[global_idx] = vy_local[local_idx];
                vz_global[global_idx] = vz_local[local_idx];
            }
        }
    }
    
    // Receive data from other ranks
    MPI_Status status;
    for (int r = 1; r < num_ranks; r++) {
        // Calculate rank r's domain position
        col = r % n_x;
        row = (r / n_x) % n_y;
        layer = r / (n_x * n_y);
        
        // Calculate rank r's dimensions
        int r_dim_x = chunk_w + (col < extra_w ? 1 : 0);
        int r_dim_y = chunk_h + (row < extra_h ? 1 : 0);
        int r_dim_z = chunk_d + (layer < extra_d ? 1 : 0);
        int r_size = r_dim_x * r_dim_y * r_dim_z;
        
        // Allocate temporary buffers
        float* temp_vx = new float[r_size];
        float* temp_vy = new float[r_size];
        float* temp_vz = new float[r_size];
        
        // Receive velocity components from rank r
        MPI_Recv(temp_vx, r_size, MPI_FLOAT, r, 200, MPI_COMM_WORLD, &status);
        MPI_Recv(temp_vy, r_size, MPI_FLOAT, r, 201, MPI_COMM_WORLD, &status);
        MPI_Recv(temp_vz, r_size, MPI_FLOAT, r, 202, MPI_COMM_WORLD, &status);
        
        // Calculate offset for rank r
        offset_x = col * chunk_w + std::min<int>(col, extra_w);
        offset_y = row * chunk_h + std::min<int>(row, extra_h);
        offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
        
        // Copy data to global arrays
        for (int k = 0; k < r_dim_z; ++k) {
            for (int j = 0; j < r_dim_y; ++j) {
                for (int i = 0; i < r_dim_x; ++i) {
                    int local_idx = i + r_dim_x * (j + r_dim_y * k);
                    int global_idx = (offset_x + i) + total_x * ((offset_y + j) + total_y * (offset_z + k));
                    vx_global[global_idx] = temp_vx[local_idx];
                    vy_global[global_idx] = temp_vy[local_idx];
                    vz_global[global_idx] = temp_vz[local_idx];
                }
            }
        }
        
        delete[] temp_vx;
        delete[] temp_vy;
        delete[] temp_vz;
    }
}


void sendVelocityToRank0(LbmDQ* lbm) {
    float* vx_local = lbm->getVelocityX();
    float* vy_local = lbm->getVelocityY();
    float* vz_local = lbm->getVelocityZ();
    
    int local_size = lbm->getDimX() * lbm->getDimY() * lbm->getDimZ();
    
    // Send velocity components to rank 0
    MPI_Send(vx_local, local_size, MPI_FLOAT, 0, 200, MPI_COMM_WORLD);
    MPI_Send(vy_local, local_size, MPI_FLOAT, 0, 201, MPI_COMM_WORLD);
    MPI_Send(vz_local, local_size, MPI_FLOAT, 0, 202, MPI_COMM_WORLD);
}

inline void printSimulationDiagnostics(int t, int rank, LbmDQ* lbm, double dt, double dx, double physical_density, uint32_t time_steps) {
    if (t % 500 == 0 && t <= 10000) {
        lbm->computeVorticity();
        lbm->gatherDataOnRank0(LbmDQ::Vorticity);
        
        if (rank == 0) {
            char vts_filename[128];
            snprintf(vts_filename, sizeof(vts_filename), "paraview/simulation_state_t%05d.vts", t);
            exportSimulationStateToVTS(lbm, vts_filename, dt, dx, physical_density, time_steps);
        }
    }
}

inline void exportVelocityDiagnostics(int t, int rank, int num_ranks, LbmDQ* lbm, double dt, double dx, double physical_density, uint32_t time_steps) {
    // Export velocity vectors for streamline visualization
    if (t % 500 == 0 && t <= 10000) {
        if (rank == 0) {
            char velocity_filename[128];
            snprintf(velocity_filename, sizeof(velocity_filename), "paraview/velocity_vectors_t%05d.vts", t);
            exportVelocityVectorsToVTS(rank, num_ranks, lbm, velocity_filename, dt, dx, physical_density, time_steps);
        } else {
            // Non-rank 0 processes send their velocity data
            exportVelocityVectorsToVTS(rank, num_ranks, lbm, nullptr, dt, dx, physical_density, time_steps);
        }
    }
}

#endif // PARAVIEW_SIM_HPP 
