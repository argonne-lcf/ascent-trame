#ifndef _LBMDQ_MPI_HPP_
#define _LBMDQ_MPI_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <set>
#include <vector>

// Helper class for creating barriers
class Barrier
{
    public:
	int x1, x2, y1, y2, z1, z2;
        Barrier(int x_start, int x_end, int y_start, int y_end, int z_start, int z_end)
            : x1(x_start), x2(x_end), y1(y_start), y2(y_end), z1(z_start), z2(z_end) {}
        virtual ~Barrier() {}
};

class Barrier3D : public Barrier
{
    public:
	Barrier3D(int x_start, int x_end, int y_start, int y_end, int z_start, int z_end)
            : Barrier(x_start, x_end, y_start, y_end, z_start, z_end) {}
        ~Barrier3D() {}
};

// D3Q15 discrete velocities and weights
static const int cD3Q15[15][3] = {
	{0,0,0}, {1,0,0}, {-1,0,0},
	{0,1,0}, {0,-1,0}, {0,0,1},
	{0,0,-1}, {1,1,1}, {-1,1,1},
	{1,-1,1}, {1,1,-1}, {-1,-1,1},
	{-1,1,-1}, {1,-1,-1}, {-1,-1,-1}
};

static const double wD3Q15[15] = {
	2.0/9.0,
	1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
	1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0
};

// D3Q19 discrete velocities and weights
static const int cD3Q19[19][3] = {
    {0,0,0}, {1,0,0}, {-1,0,0},
    {0,1,0}, {0,-1,0}, {0,0,1},
    {0,0,-1}, {1,1,0}, {-1,1,0},
    {1,-1,0}, {-1,-1,0}, {1,0,1},
    {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

static const double wD3Q19[19] = {
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// D3Q27 discrete velocities and weights
static const int cD3Q27[27][3] = {
    {0,0,0},
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1},
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1},
    {1,1,1}, {-1,1,1}, {1,-1,1}, {-1,-1,1},
    {1,1,-1}, {-1,1,-1}, {1,-1,-1}, {-1,-1,-1}
};

static const double wD3Q27[27] = {
    8.0/27.0,
    2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0,
    1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
};

// Lattice-Boltzman Methods CFD simulation
class LbmDQ
{
    public:
        enum FluidProperty {None, Density, Speed, Vorticity};
	enum LatticeType {D3Q15, D3Q19, D3Q27};

	// Guard byte constants
	static constexpr int GUARD_SIZE = 8;
        static constexpr float GUARD_FLOAT = 1e30f;
        static constexpr uint8_t GUARD_BOOL = 0xAB;


    private:
        enum Neighbor {NeighborN, NeighborE, NeighborS, NeighborW, NeighborNE, NeighborNW, NeighborSE, NeighborSW, NeighborUp, NeighborDown};
        enum Column {LeftBoundaryCol, LeftCol, RightCol, RightBoundaryCol};

        int rank;
        int num_ranks;
        uint32_t total_x;
        uint32_t total_y;
	uint32_t total_z;
        uint32_t dim_x;
        uint32_t dim_y;
	uint32_t dim_z;
        uint32_t start_x;
        uint32_t start_y;
	uint32_t start_z;
        uint32_t num_x;
        uint32_t num_y;
	uint32_t num_z;
        int offset_x;
        int offset_y;
	int offset_z;
        uint32_t *rank_local_size;
        uint32_t *rank_local_start;
        double speed_scale;
        int Q;
	LatticeType lattice_type;
	const int (*c)[3];
	const double *w;
	float *f;
        float *density;
        float *velocity_x;
        float *velocity_y;
	float *velocity_z;
        float *vorticity;
        float *speed;
        uint8_t *barrier;
        FluidProperty stored_property;
        float *recv_buf;
        uint8_t *brecv_buf;
        int neighbors[10];
        MPI_Datatype columns_2d[4];
        MPI_Datatype own_scalar;
        MPI_Datatype own_bool;
        MPI_Datatype *other_scalar;
        MPI_Datatype *other_bool;

	MPI_Comm cart_comm;
	MPI_Datatype faceXlo, faceXhi;
	MPI_Datatype faceYlo, faceYhi;
	MPI_Datatype faceZlo, faceZhi;
	MPI_Datatype faceN, faceS;
	MPI_Datatype faceE, faceW;
	MPI_Datatype faceSW, faceNE;
	MPI_Datatype faceNW, faceSE;

        // Add missing member variables
        float *f_0, *f_1, *f_2, *f_3, *f_4, *f_5, *f_6, *f_7, *f_8, *f_9, *f_10, *f_11, *f_12, *f_13, *f_14, *f_15, *f_16, *f_17, *f_18, *f_19, *f_20, *f_21, *f_22, *f_23, *f_24, *f_25, *f_26;
        float *dbl_arrays;
        uint32_t block_width, block_height, block_depth;
	float **fPtr;
	uint32_t array_size;
	
	// Temporary storage for bounce-back streaming
	std::vector<float> f_Old;
	size_t last_size;

	// MPI particle streaming buffers
	struct ParticleBuffer {
		std::vector<float> particles;  // Particle values
		std::vector<int> directions;   // Direction indices
		std::vector<int> positions;    // Target positions (i,j,k packed)

		void clear() {
			particles.clear();
			directions.clear();
			positions.clear();
		}

		void addParticle(float value, int dir, int i, int j, int k) {
			particles.push_back(value);
			directions.push_back(dir);
			positions.push_back(i + 1000 * (j + 1000 * k));  // Pack i,j,k
		}

		void unpackPosition(int packed, int& i, int& j, int& k) {
			k = packed / 1000000;
			packed %= 1000000;
			j = packed / 1000;
			i = packed % 1000;
		}
	};

	ParticleBuffer send_buffers[10];  // One for each neighbor direction
	ParticleBuffer recv_buffers[10];
	
	// Add a member variable to track if corruption has been reported
	bool guard_corruption_reported = false;
	
	// Helper functions
        inline int idx3D(int x, int y, int z) const {
            if (x < 0 || x >= dim_x || y < 0 || y >= dim_y || z < 0 || z >= dim_z) {
		fprintf(stderr, "[Rank %d] ERROR: idx3D out of bounds: x=%d, y=%d, z=%d, dim_x=%d, dim_y=%d, dim_z=%d\n", rank, x, y, z, dim_x, dim_y, dim_z);
		fprintf(stderr, "[Rank %d] ERROR: idx3D out of bounds: x=%d, y=%d, z=%d, dim_x=%d, dim_y=%d, dim_z=%d\n", rank, x, y, z, dim_x, dim_y, dim_z);
		MPI_Abort(MPI_COMM_WORLD, 1);
            }
	    return x + dim_x * (y + dim_y * z);
        }

        inline float& f_at(int d, int x, int y, int z) const {
            int idx = idx3D(x, y, z);
	    return fPtr[d][idx];
        }

        void setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density);
	void getBest3DPartition(int num_ranks, int dim_x, int dim_y, int dim_z, int *n_x, int *n_y, int *n_z);

    public:
        LbmDQ(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks, LatticeType type);
        ~LbmDQ();

        void initBarrier(std::vector<Barrier*> barriers);
        void initFluid(double physical_speed);
        void updateFluid(double physical_speed);
        void collide(double viscosity, int t);
        void stream();
        void bounceBackStream();
        bool checkStability();
        void computeSpeed();
        void computeVorticity();
        void gatherDataOnRank0(FluidProperty property);
        void exchangeBoundaries();
	void exchangeVelocityBoundaries();
	void captureBoundaryParticle(int i, int j, int k, int ni, int nj, int nk, int d, float value);
	uint32_t getDimX();
        uint32_t getDimY();
	uint32_t getDimZ();
        uint32_t getTotalDimX();
        uint32_t getTotalDimY();
	uint32_t getTotalDimZ();
        uint32_t getOffsetX();
        uint32_t getOffsetY();
	uint32_t getOffsetZ();
        uint32_t getStartX();
        uint32_t getStartY();
	uint32_t getStartZ();
        uint32_t getSizeX();
        uint32_t getSizeY();
	uint32_t getSizeZ();
        uint32_t* getRankLocalSize(int rank);
        uint32_t* getRankLocalStart(int rank);
        uint8_t* getBarrier();
	static constexpr int TAG_F  = 100;
	static constexpr int TAG_B  = 105;

	// Local data access
        inline float* getDensity() {
            return density;
        }

        inline float* getVelocityX() {
            return velocity_x;
        }

        inline float* getVelocityY() {
            return velocity_y;
        }

        inline float* getVelocityZ() {
            return velocity_z;
        }

        inline float* getVorticity() {
            return vorticity;
        }

        inline float* getSpeed() {
            return speed;
        }

        // Gathered data access (rank 0 only)
        inline float* getGatheredDensity() {
            if (rank != 0 || stored_property != Density) return NULL;
            return recv_buf;
        }

        inline float* getGatheredVorticity() {
            if (rank != 0 || stored_property != Vorticity) return NULL;
            return recv_buf;
        }

        inline float* getGatheredSpeed() {
            if (rank != 0 || stored_property != Speed) return NULL;
            return recv_buf;
        }

	// Check guard bytes for buffer overruns
        void checkGuards() {
	    bool guard_ok = true;
            for (int i = 0; i < GUARD_SIZE; ++i) {
                if (recv_buf && recv_buf[array_size + i] != GUARD_FLOAT) {
                    fprintf(stderr, "[Rank %d] WARNING: recv_buf guard byte %d corrupted!\n", rank, i);
		    guard_ok = false;
                }
	    }
            for (int i = 0; i < GUARD_SIZE; ++i) {
                if (brecv_buf && brecv_buf[array_size + i] != GUARD_BOOL) {
		    fprintf(stderr, "[Rank %d] WARNING: brecv_buf guard byte %d corrupted!\n", rank, i);
		    guard_ok = false;
                }
            }
	    if (dbl_arrays) {
                uint32_t dbl_arrays_size = dim_x * dim_y * dim_z;
		for (int i = 0; i < GUARD_SIZE; ++i) {
                    if (dbl_arrays[(Q + 6) * dbl_arrays_size + i] != GUARD_FLOAT) {
			fprintf(stderr, "[Rank %d] WARNING: dbl_arrays guard byte %d corrupted!\n", rank, i);
			guard_ok = false;
                    }
                }
            }
            if (!guard_ok) {
	        fprintf(stderr, "[Rank %d] WARNING: Buffer overrun detected in checkGuards!\n", rank);
	    }
	}

        float* getDblArrays() { return dbl_arrays; }
        int getQ() const { return Q; }
	
	// Add a function to print corrupted guard bytes of dbl_arrays
        void printDblArraysGuardCorruption(const char* label, int step = -1) {
	    if (guard_corruption_reported) return;
            if (!dbl_arrays) return;
            uint32_t size = dim_x * dim_y * dim_z;
            for (int i = 0; i < GUARD_SIZE; ++i) {
                float expected = GUARD_FLOAT + i;
                float actual = dbl_arrays[(Q + 6) * size + i];
                if (actual != expected) {
		    fprintf(stderr, "[Rank %d] FIRST GUARD CORRUPTION after %s: dbl_arrays[%d] corrupted! Expected: %f, Actual: %f\n", rank, label, ((Q + 6) * size + i), expected, actual);
		    guard_corruption_reported = true;
                    break;
		}
            }
        }
	// Add public getter for local barrier array
        inline uint8_t* getLocalBarrier() { return barrier; }
        // Add public getter for discrete velocity array c
        inline const int (*getC() const)[3] { return c; }
};

namespace {
    static bool guard_reported[LbmDQ::GUARD_SIZE] = {false};
}

// constructor
LbmDQ::LbmDQ(uint32_t width, uint32_t height, uint32_t depth, double scale, int task_id, int num_tasks, LatticeType type)
{
    rank = task_id;
    num_ranks = num_tasks;
    speed_scale = scale;
    stored_property = None;
    lattice_type = type;

    // set up lattice model
    if (lattice_type == D3Q15) {
	Q = 15;
	c = cD3Q15;
	w = wD3Q15;
    }
    else if (lattice_type == D3Q19) {
	Q = 19;
	c = cD3Q19;
	w = wD3Q19;
    }
    else if (lattice_type == D3Q27) {
	Q = 27;
	c = cD3Q27;
	w = wD3Q27;
    }

    // split up problem space
    int n_x, n_y, n_z, col, row, layer, chunk_w, chunk_h, chunk_d, extra_w, extra_h, extra_d;
    getBest3DPartition(num_ranks, width, height, depth, &n_x, &n_y, &n_z);
    chunk_w = width / n_x;
    chunk_h = height / n_y;
    chunk_d = depth / n_z;
    extra_w = width % n_x;
    extra_h = height % n_y;
    extra_d = depth % n_z;
    col = rank % n_x;
    row = (rank / n_x) % n_y;
    layer = rank / (n_x * n_y);

    num_x = chunk_w + ((col < extra_w) ? 1 : 0);
    num_y = chunk_h + ((row < extra_h) ? 1 : 0);
    num_z = chunk_d + ((layer < extra_d) ? 1 : 0);
    offset_x = col * chunk_w + std::min<int>(col, extra_w);
    offset_y = row * chunk_h + std::min<int>(row, extra_h);
    offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
    start_x = (col == 0) ? 0 : 1;
    start_y = (row == 0) ? 0 : 1;
    start_z = (layer == 0) ? 0 : 1;

    // set up sub grid for simulation
    total_x = width;
    total_y = height;
    total_z = depth;
    block_width = num_x;
    block_height = num_y;
    block_depth = num_z;
    dim_x = block_width;
    dim_y = block_height;
    dim_z = block_depth;

    // Set array_size before allocation
    if (rank == 0) {
        array_size = total_x * total_y * total_z;
    } else {
        array_size = dim_x * dim_y * dim_z;
    }

    // Allocate recv_buf and brecv_buf with guard bytes
    recv_buf = new float[array_size + GUARD_SIZE];
    brecv_buf = new uint8_t[array_size + GUARD_SIZE];
    // Set guard values
    for (int i = 0; i < GUARD_SIZE; ++i) {
        recv_buf[array_size + i] = GUARD_FLOAT;
        brecv_buf[array_size + i] = GUARD_BOOL;
    }

    // MPI Checks - Abort 
    if (num_x == 0 || num_y == 0 || num_z == 0) {
        fprintf(stderr, "[Rank %d] ERROR: Subdomain has zero size! num_x=%d, num_y=%d, num_z=%d\n", rank, num_x, num_y, num_z);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (offset_x < 0 || offset_y < 0 || offset_z < 0) {
        fprintf(stderr, "[Rank %d] ERROR: Negative offsets! offset_x=%d, offset_y=%d, offset_z=%d\n", rank, offset_x, offset_y, offset_z);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    if (offset_x + num_x > width || offset_y + num_y > height || offset_z + num_z > depth) {
        fprintf(stderr, "[Rank %d] ERROR: Subdomain extends beyond global domain!\n", rank);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }

    neighbors[NeighborN] = (row == n_y-1) ? MPI_PROC_NULL : rank + n_x;
    neighbors[NeighborE] = (col == n_x-1) ? MPI_PROC_NULL : rank + 1;
    neighbors[NeighborS] = (row == 0) ? MPI_PROC_NULL : rank - n_x; 
    neighbors[NeighborW] = (col == 0) ? MPI_PROC_NULL : rank - 1;
    neighbors[NeighborNE] = (row == n_y-1 || col == n_x-1) ? MPI_PROC_NULL : rank + n_x + 1;
    neighbors[NeighborNW] = (row == n_y-1 || col == 0) ? MPI_PROC_NULL : rank + n_x - 1;
    neighbors[NeighborSE] = (row == 0 || col == n_x-1) ? MPI_PROC_NULL : rank - n_x + 1;
    neighbors[NeighborSW] = (row == 0 || col == 0) ? MPI_PROC_NULL : rank - n_x - 1;
    
    //new Z neighbors
    neighbors[NeighborUp] = (layer == n_z-1) ? MPI_PROC_NULL : rank + (n_x * n_y);
    neighbors[NeighborDown] = (layer == 0) ? MPI_PROC_NULL : rank - (n_x * n_y);
    
    int dims[3] = {n_z, n_y, n_x};
    int periods[3] = {0, 0, 0};
    int reorder = 0;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);

    // create data types for exchanging data with neighbors
    int sizes3D[3] = {int(total_z), int(total_y), int(total_x)};
    int subsize3D[3] = {int(num_z), int(num_y), int(num_x)};
    int offsets3D[3] = {int(offset_z), int(offset_y), int(offset_x)};

    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_FLOAT, &own_scalar);
    MPI_Type_commit(&own_scalar);
    MPI_Type_create_subarray(3, sizes3D, subsize3D, offsets3D, MPI_ORDER_C, MPI_BYTE, &own_bool);
    MPI_Type_commit(&own_bool);

    other_scalar = new MPI_Datatype[num_ranks];
    other_bool = new MPI_Datatype[num_ranks];

    for (int r = 0; r < num_ranks; r++) {
        int col = r % n_x;
        int row = (r / n_x) % n_y;
        int layer = r / (n_x * n_y);
        int osub[3] = {int(chunk_d + (layer < extra_d)), int(chunk_h + (row < extra_h)), int(chunk_w + (col < extra_w))};
        int ooffset[3] = {int(layer * chunk_d + std::min(layer, extra_d)), 
                         int(row * chunk_h + std::min(row, extra_h)),
                         int(col * chunk_w + std::min(col, extra_w))};

        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_FLOAT, &other_scalar[r]);
        MPI_Type_commit(&other_scalar[r]);
        MPI_Type_create_subarray(3, sizes3D, osub, ooffset, MPI_ORDER_C, MPI_BYTE, &other_bool[r]);
        MPI_Type_commit(&other_bool[r]);
    }

    //X-Faces
    int subsX[3]   = {int(num_z), int(num_y), 1};
    int offsXlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsXhi[3] = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsX, offsXlo, MPI_ORDER_C, MPI_FLOAT, &faceXlo);
    MPI_Type_create_subarray(3, sizes3D, subsX, offsXhi, MPI_ORDER_C, MPI_FLOAT, &faceXhi);
    MPI_Type_commit(&faceXlo);
    MPI_Type_commit(&faceXhi);

    //Y-Faces
    int subsY[3]   = {int(num_z), 1, int(num_x)};
    int offsYlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsYhi[3] = {int(start_z), int(dim_y - start_y - 1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsY, offsYlo, MPI_ORDER_C, MPI_FLOAT, &faceYlo);
    MPI_Type_create_subarray(3, sizes3D, subsY, offsYhi, MPI_ORDER_C, MPI_FLOAT, &faceYhi);
    MPI_Type_commit(&faceYlo);
    MPI_Type_commit(&faceYhi);

    //Z-Faces
    int subsZ[3]   = {1, int(num_y), int(num_x)};
    int offsZlo[3] = {int(start_z), int(start_y), int(start_x)};
    int offsZhi[3] = {int(dim_z - start_z - 1), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsZ, offsZlo, MPI_ORDER_C, MPI_FLOAT, &faceZlo);
    MPI_Type_create_subarray(3, sizes3D, subsZ, offsZhi, MPI_ORDER_C, MPI_FLOAT, &faceZhi);
    MPI_Type_commit(&faceZlo);
    MPI_Type_commit(&faceZhi);


    //North
    int subsN[3]   = {int(num_z), 1, int(num_x)};
    int offsN[3]   = {int(start_z), int(dim_y - start_y -1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsN, offsN, MPI_ORDER_C, MPI_FLOAT, &faceN);
    MPI_Type_commit(&faceN);

    //South
    int subsS[3]   = {int(num_z), 1, int(num_x)};
    int offsS[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsS, offsS, MPI_ORDER_C, MPI_FLOAT, &faceS);
    MPI_Type_commit(&faceS);

    //East
    int subsE[3]   = {int(num_z), int(num_y), 1};
    int offsE[3]   = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsE, offsE, MPI_ORDER_C, MPI_FLOAT, &faceE);
    MPI_Type_commit(&faceE);

    //West
    int subsW[3]   = {int(num_z), int(num_y), 1};
    int offsW[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsW, offsW, MPI_ORDER_C, MPI_FLOAT, &faceW);
    MPI_Type_commit(&faceW);

    //Northeast
    int subsNE[3]   = {int(num_z), 1, 1};
    int offsNE[3]   = {int(start_z), int(dim_y - start_y - 1), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsNE, offsNE, MPI_ORDER_C, MPI_FLOAT, &faceNE);
    MPI_Type_commit(&faceNE);

    //Northwest
    int subsNW[3]   = {int(num_z), 1, 1};
    int offsNW[3]   = {int(start_z), int(dim_y - start_y - 1), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsNW, offsNW, MPI_ORDER_C, MPI_FLOAT, &faceNW);
    MPI_Type_commit(&faceNW);

    //Southeast
    int subsSE[3]   = {int(num_z), 1, 1};
    int offsSE[3]   = {int(start_z), int(start_y), int(dim_x - start_x - 1)};
    MPI_Type_create_subarray(3, sizes3D, subsSE, offsSE, MPI_ORDER_C, MPI_FLOAT, &faceSE);
    MPI_Type_commit(&faceSE);

    //Southwest
    int subsSW[3]   = {int(num_z), 1, 1};
    int offsSW[3]   = {int(start_z), int(start_y), int(start_x)};
    MPI_Type_create_subarray(3, sizes3D, subsSW, offsSW, MPI_ORDER_C, MPI_FLOAT, &faceSW);
    MPI_Type_commit(&faceSW);

    // Number of guard elements
    constexpr int GUARD_SIZE = 8;
    constexpr float GUARD_FLOAT = 1e30f;
    constexpr uint8_t GUARD_BOOL = 0xAB;

    // Print MPI datatype sizes for debugging
    int size_own_scalar = 0, size_own_bool = 0;
    MPI_Type_size(own_scalar, &size_own_scalar);
    MPI_Type_size(own_bool, &size_own_bool);
    for (int r = 0; r < num_ranks; r++) {
        int size_other_scalar = 0, size_other_bool = 0;
        MPI_Type_size(other_scalar[r], &size_other_scalar);
        MPI_Type_size(other_bool[r], &size_other_bool);
    }
    
    uint32_t size = dim_x * dim_y * dim_z;
    size_t total_memory = 0;

    // allocate all float arrays at once
    dbl_arrays = new float[(Q + 6) * size + GUARD_SIZE];
    total_memory += (Q + 6) * size * sizeof(float);
    
    // Set guard values for dbl_arrays
    for (int i = 0; i < GUARD_SIZE; ++i) {
        dbl_arrays[(Q + 6) * size + i] = GUARD_FLOAT + i;
    }

    // Zero-initialize all simulation state arrays to prevent uninitialized value warnings
    std::fill(dbl_arrays, dbl_arrays + (Q + 6) * size, 0.0f);

    // set array pointers
    f_0        = dbl_arrays + (0*size);
    f_1        = dbl_arrays + (1*size);
    f_2        = dbl_arrays + (2*size);
    f_3        = dbl_arrays + (3*size);
    f_4        = dbl_arrays + (4*size);
    f_5        = dbl_arrays + (5*size);
    f_6        = dbl_arrays + (6*size);
    f_7        = dbl_arrays + (7*size);
    f_8        = dbl_arrays + (8*size);
    f_9        = dbl_arrays + (9*size);
    f_10       = dbl_arrays + (10*size);
    f_11       = dbl_arrays + (11*size);
    f_12       = dbl_arrays + (12*size);
    f_13       = dbl_arrays + (13*size);
    f_14       = dbl_arrays + (14*size);
    if (Q == 19) {
	f_15 = dbl_arrays + (15*size);
	f_16 = dbl_arrays + (16*size);
	f_17 = dbl_arrays + (17*size);
	f_18 = dbl_arrays + (18*size);
    }
    if (Q == 27) {
	f_15 = dbl_arrays + (15*size);
        f_16 = dbl_arrays + (16*size);
        f_17 = dbl_arrays + (17*size);
        f_18 = dbl_arrays + (18*size);
        f_19 = dbl_arrays + (19*size);
        f_20 = dbl_arrays + (20*size);
        f_21 = dbl_arrays + (21*size);
        f_22 = dbl_arrays + (22*size);
        f_23 = dbl_arrays + (23*size);
        f_24 = dbl_arrays + (24*size);
        f_25 = dbl_arrays + (25*size);
        f_26 = dbl_arrays + (26*size);
    }

    // initialize f pointer to point to f_0
    f = f_0;
    
    fPtr = new float*[Q];
    fPtr[0] = f_0;
    fPtr[1] = f_1;
    fPtr[2] = f_2;
    fPtr[3] = f_3;
    fPtr[4] = f_4;
    fPtr[5] = f_5;
    fPtr[6] = f_6;
    fPtr[7] = f_7;
    fPtr[8] = f_8;
    fPtr[9] = f_9;
    fPtr[10] = f_10;
    fPtr[11] = f_11;
    fPtr[12] = f_12;
    fPtr[13] = f_13;
    fPtr[14] = f_14;
    if (Q == 19) {
	fPtr[15] = f_15;
	fPtr[16] = f_16;
	fPtr[17] = f_17;
	fPtr[18] = f_18;
    }
    else if (Q == 27) {
        fPtr[15] = f_15;
        fPtr[16] = f_16;
        fPtr[17] = f_17;
        fPtr[18] = f_18;
        fPtr[19] = f_19;
        fPtr[20] = f_20;
        fPtr[21] = f_21;
        fPtr[22] = f_22;
        fPtr[23] = f_23;
        fPtr[24] = f_24;
        fPtr[25] = f_25;
        fPtr[26] = f_26;
    }
    else if (Q == 15) {
	    //No additional assignments
	    //f0->f14 defined above
    }

    density    = dbl_arrays + (Q*size);
    velocity_x = dbl_arrays + ((Q+1)*size);
    velocity_y = dbl_arrays + ((Q+2)*size);
    velocity_z = dbl_arrays + ((Q+3)*size);
    vorticity  = dbl_arrays + ((Q+4)*size);
    speed      = dbl_arrays + ((Q+5)*size);
    
    // allocate boolean array
    barrier = new uint8_t[size];
    total_memory += size * sizeof(uint8_t);

    // Allocate rank_local_size and rank_local_start arrays
    rank_local_size = new uint32_t[2 * num_ranks];
    rank_local_start = new uint32_t[2 * num_ranks];
    
    // Initialize the arrays with the correct values
    for (int r = 0; r < num_ranks; r++) {
        int col = r % n_x;
        int row = (r / n_x) % n_y;
        
        rank_local_size[2*r] = chunk_w + ((col < extra_w) ? 1 : 0);  // x size
        rank_local_size[2*r+1] = chunk_h + ((row < extra_h) ? 1 : 0); // y size
        
        rank_local_start[2*r] = col * chunk_w + std::min<int>(col, extra_w);  // x start
        rank_local_start[2*r+1] = row * chunk_h + std::min<int>(row, extra_h); // y start
    }

    std::fill(recv_buf, recv_buf + array_size, 0.0f);
    std::fill(brecv_buf, brecv_buf + array_size, GUARD_BOOL);

    f_Old.resize(Q * size, 0.0f);
}

// destructor
LbmDQ::~LbmDQ()
{
    checkGuards();

    // Free MPI types
    if (faceXlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceXlo); faceXlo = MPI_DATATYPE_NULL; }
    if (faceXhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceXhi); faceXhi = MPI_DATATYPE_NULL; }
    if (faceYlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceYlo); faceYlo = MPI_DATATYPE_NULL; }
    if (faceYhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceYhi); faceYhi = MPI_DATATYPE_NULL; }
    if (faceZlo != MPI_DATATYPE_NULL) { MPI_Type_free(&faceZlo); faceZlo = MPI_DATATYPE_NULL; }
    if (faceZhi != MPI_DATATYPE_NULL) { MPI_Type_free(&faceZhi); faceZhi = MPI_DATATYPE_NULL; }
    if (faceN != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceN);   faceN = MPI_DATATYPE_NULL; }
    if (faceS != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceS);   faceS = MPI_DATATYPE_NULL; }
    if (faceE != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceE);   faceE = MPI_DATATYPE_NULL; }
    if (faceW != MPI_DATATYPE_NULL)   { MPI_Type_free(&faceW);   faceW = MPI_DATATYPE_NULL; }
    if (faceNE != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceNE);  faceNE = MPI_DATATYPE_NULL; }
    if (faceNW != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceNW);  faceNW = MPI_DATATYPE_NULL; }
    if (faceSE != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceSE);  faceSE = MPI_DATATYPE_NULL; }
    if (faceSW != MPI_DATATYPE_NULL)  { MPI_Type_free(&faceSW);  faceSW = MPI_DATATYPE_NULL; }
    if (own_scalar != MPI_DATATYPE_NULL) { MPI_Type_free(&own_scalar); own_scalar = MPI_DATATYPE_NULL; }
    if (own_bool != MPI_DATATYPE_NULL)   { MPI_Type_free(&own_bool);   own_bool = MPI_DATATYPE_NULL; }

    if (other_scalar != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_scalar[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_scalar[i]);
                other_scalar[i] = MPI_DATATYPE_NULL;
            }
        }
	delete[] other_scalar;
        other_scalar = nullptr;
    }

    if (other_bool != nullptr) {
        for (int i=0; i < num_ranks; i++) {
            if (other_bool[i] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&other_bool[i]);
                other_bool[i] = MPI_DATATYPE_NULL;
            }
        }
        delete[] other_bool;
        other_bool = nullptr;
    }
   
    // Debug before guard check
    bool guard_ok = true;
    for (int i = 0; i < GUARD_SIZE; ++i) {
        if (recv_buf && recv_buf[array_size + i] != GUARD_FLOAT) {
            fprintf(stderr, "[Rank %d] WARNING: recv_buf guard byte %d corrupted! Value: %f", rank, i, recv_buf[array_size + i]);
	    guard_ok = false;
        }
        if (brecv_buf && brecv_buf[array_size + i] != GUARD_BOOL) {
            fprintf(stderr, "[Rank %d] WARNING: brecv_buf guard byte %d corrupted! Value: %hhu", rank, i, brecv_buf[array_size + i]);
	    guard_ok = false;
        }
    }
    if (dbl_arrays) {
	uint32_t dbl_arrays_size = dim_x * dim_y * dim_z;
        for (int i = 0; i < GUARD_SIZE; ++i) {
                float expected = GUARD_FLOAT + i;
            float actual = dbl_arrays[(Q + 6) * dbl_arrays_size + i];
            if (actual != expected) {
                if (!guard_reported[i]) {
                    fprintf(stderr, "[Rank %d] FIRST CORRUPTION: dbl_arrays guard byte %d corrupted! Expected: %f, Actual: %f", rank, i, expected, actual);
		    guard_reported[i] = true;
                }
		fprintf(stderr, "[Rank %d] WARNING: dbl_arrays guard byte %d corrupted! Value: %f", rank, i, actual);
		guard_ok = false;
            }
        }
    }
    if (!guard_ok) {
        fprintf(stderr, "[Rank %d] WARNING: Buffer overrun detected before delete!", rank);
    }
    // Add missing deletes for dynamically allocated arrays
    if (dbl_arrays) { delete[] dbl_arrays; dbl_arrays = nullptr; }
    if (recv_buf) { delete[] recv_buf; recv_buf = nullptr; }
    if (brecv_buf) { delete[] brecv_buf; brecv_buf = nullptr; }
    if (barrier) { delete[] barrier; barrier = nullptr; }
    if (fPtr) { delete[] fPtr; fPtr = nullptr; }
    if (rank_local_size) { delete[] rank_local_size; rank_local_size = nullptr; }
    if (rank_local_start) { delete[] rank_local_start; rank_local_start = nullptr; }
}

// initialize barrier based on selected type
void LbmDQ::initBarrier(std::vector<Barrier*> barriers)
{
    memset(barrier, 0, dim_x * dim_y * dim_z * sizeof(uint8_t));
    for (size_t i = 0; i < barriers.size(); ++i) {
        Barrier* b = barriers[i];
        for (int k = b->z1; k <= b->z2; ++k) {
            int z = k - offset_z;
            if (z < 0 || z >= dim_z) continue;
            for (int j = b->y1; j <= b->y2; ++j) {
                int y = j - offset_y;
                if (y < 0 || y >= dim_y) continue;
                for (int xg = b->x1; xg <= b->x2; ++xg) {
                    int x = xg - offset_x;
                    if (x < 0 || x >= dim_x) continue;
                    int idx = idx3D(x, y, z);
                    if (idx >= 0 && idx < dim_x * dim_y * dim_z) {
                        barrier[idx] = 1;                    
		    }
                }
            }
        }
    }
}

void LbmDQ::initFluid(double physical_speed)
{
    int i, j, k;
    double speed = speed_scale * physical_speed;
    for (k = 0; k < dim_z; k++) {
        for (j = 0; j < dim_y; j++) {
            for (i = 0; i < dim_x; i++) {
                int idx = idx3D(i, j, k);
                
                // Skip barrier locations - barriers are already in place
                if (barrier[idx] == 1) {
                    continue;
                }
                
	    	if ((offset_x + i) == 0) {
                    setEquilibrium(i, j, k, speed, 0.0, 0.0, 1.0); // Inlet face: flow in +X
                } else {
                    setEquilibrium(i, j, k, 0.0, 0.0, 0.0, 1.0); // Rest of domain: at rest
                }
		vorticity[idx] = 0.0;
            }
        }
    }
}

void LbmDQ::updateFluid(double physical_speed)
{
    int j, k;
    double speed = speed_scale * physical_speed;
    static int call_count = 0;

    // Only set equilibrium for the physical inlet (global x=0)
    for (k = 0; k < dim_z; k++) {
        for (j = 0; j < dim_y; j++) {
            int global_x = offset_x + 0;
            if (global_x == 0) {
                // Zou-He velocity inlet: only modify specific directions, preserve density
                int idx = idx3D(0, j, k);
                
                // Calculate density from current distribution
                double rho = 0.0;
                for (int d = 0; d < Q; ++d) {
                    rho += fPtr[d][idx];
                }
                if (rho <= 0.0) rho = 1.0; // Fallback for initialization
                
                // Set velocity components
                double ux = speed;
                double uy = 0.0;
                double uz = 0.0;
                
                // Only update outgoing directions (towards +x) to impose velocity
                double usq = ux*ux + uy*uy + uz*uz;
                for (int d = 0; d < Q; ++d) {
                    if (c[d][0] > 0) { // Only +x directions
                        double cu = 3.0 * (c[d][0] * ux + c[d][1] * uy + c[d][2] * uz);
                        fPtr[d][idx] = w[d] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*usq);
                    }
                }	
            }
        }
    }
    call_count++;
}

// particle collision
void LbmDQ::collide(double viscosity, int t)
{
	int i, j, k, idx;
        double omega = 1.0 / (3.0 * viscosity + 0.5); //reciprocal of relaxation time
	for (k = 0; k < dim_z; k++)
	{
		for (j = 0; j < dim_y; j++)
		{
			for (i = 0; i < dim_x; ++i)
			{
				idx = idx3D(i, j, k);
				if (barrier && barrier[idx]) {
                 		   continue; // skip barrier nodes
                		}

				// Skip inlet boundary cells (they are handled by updateFluid)
				int global_x = offset_x + i;
				if (global_x == 0) {
					continue; // inlet boundary - don't collide
				}

				// Handle outlet boundary (open boundary condition)
				if (global_x == total_x - 1) {
					int prev_idx = idx3D(i-1, j, k);
					if (i > 0) {
						for (int d = 0; d < Q; ++d) {
							if (c[d][0] < 0) { // Only -x directions
								fPtr[d][idx] = fPtr[d][prev_idx];
							}
						}
					}
					continue; // skip collision for outlet
				}
				
				// Handle wall boundaries (no-slip: zero velocity)
				int global_j = offset_y + j;
				int global_k = offset_z + k;
				if (global_j == 0 || global_j == total_y - 1 || global_k == 0 || global_k == total_z - 1) {
					// Wall boundary - set to zero velocity
					density[idx] = 1.0;
					velocity_x[idx] = 0.0;
					velocity_y[idx] = 0.0;
					velocity_z[idx] = 0.0;
					
					// Set equilibrium distribution for wall (zero velocity)
					for (int d = 0; d < Q; ++d) {
						fPtr[d][idx] = w[d] * 1.0; // rho=1, u=0
					}
					continue; // skip collision for walls
				}
				double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
				for (int d = 0; d < Q; ++d)
				{
					double fv = fPtr[d][idx];
					rho += fv;
					ux  += fv * c[d][0];
					uy  += fv * c[d][1];
					uz  += fv * c[d][2];
				}
				density[idx] = rho;
				ux /= rho; uy /= rho; uz /= rho;
				
				// Simple velocity limiting for high-resolution grids
				double speed_magnitude = sqrt(ux*ux + uy*uy + uz*uz);
				if (speed_magnitude > 0.05) {  // Conservative limit for high-res grids
					double scale = 0.05 / speed_magnitude;
					ux *= scale;
					uy *= scale;
					uz *= scale;
				}
				velocity_x[idx] = ux;
				velocity_y[idx] = uy;
				velocity_z[idx] = uz;

				double usqr = ux*ux + uy*uy + uz*uz;
				for (int d = 0; d < Q; ++d)
				{
					double cu = 3.0 * (c[d][0]*ux + c[d][1]*uy + c[d][2]*uz);
					double feq = w[d] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*usqr);
					fPtr[d][idx] += omega * (feq - fPtr[d][idx]);
				}
			}
		}
	}
	printDblArraysGuardCorruption("collideG");
}
	
// Optimized particle streaming - eliminates memory allocation/deallocation
void LbmDQ::stream()
{
	size_t slice = static_cast<size_t>(dim_x) * dim_y * dim_z;

	// Use a static buffer to avoid repeated allocation/deallocation
	static std::vector<std::vector<float>> temp_buffer_per_rank;
	static std::vector<size_t> last_slice_per_rank;

	// Ensure we have enough entries for all ranks
	if (temp_buffer_per_rank.size() <= static_cast<size_t>(rank)) {
		temp_buffer_per_rank.resize(rank + 1);
		last_slice_per_rank.resize(rank + 1);
	}

	// Only resize if needed for this rank
	if (temp_buffer_per_rank[rank].size() != slice) {
		temp_buffer_per_rank[rank].resize(slice);
		last_slice_per_rank[rank] = slice;
	}

	// Store current state in temp buffer for all directions
	static std::vector<std::vector<float>> temp_all_directions;
	if (temp_all_directions.size() <= static_cast<size_t>(rank)) {
		temp_all_directions.resize(rank + 1);
	}
	if (temp_all_directions[rank].size() != Q * slice) {
		temp_all_directions[rank].resize(Q * slice);
	}

	// Copy all distribution functions to temp buffer
	for (int d = 0; d < Q; ++d) {
		std::memcpy(temp_all_directions[rank].data() + d * slice, fPtr[d], slice * sizeof(float));
	}

	// Build opposite direction lookup table
	static std::vector<int> opposite_dir(Q);
	static bool opposite_dir_initialized = false;
	if (!opposite_dir_initialized) {
		for (int d = 0; d < Q; ++d) {
			opposite_dir[d] = -1;
			for (int od = 0; od < Q; ++od) {
				if (c[d][0] == -c[od][0] && c[d][1] == -c[od][1] && c[d][2] == -c[od][2]) {
					opposite_dir[d] = od;
					break;
				}
			}
		}
		opposite_dir_initialized = true;
	}

	// Clear all destination arrays first
	for (int d = 0; d < Q; ++d) {
		std::fill(fPtr[d], fPtr[d] + slice, 0.0f);
	}

	// Handle f[0] rest particles
	for (int idx = 0; idx < slice; ++idx) {
		fPtr[0][idx] = temp_all_directions[rank][0 * slice + idx];
	}

	// Two pass approach: first streaming, then bounce back
	// Pass 1: regular streaming only
	for (int d = 1; d < Q; ++d) {
		for (int k = 0; k < dim_z; ++k) {
			for (int j = 0; j < dim_y; ++j) {
				for (int i = 0; i < dim_x; ++i) {
					int idx = idx3D(i, j, k);
					
					int ni = i + c[d][0];
					int nj = j + c[d][1];
					int nk = k + c[d][2];

					// Check if destination is out of bounds
					if (ni < 0 || ni >= dim_x || nj < 0 || nj >= dim_y || nk < 0 || nk >= dim_z) {
						// Capture particle for MPI exchange
						captureBoundaryParticle(i, j, k, ni, nj, nk, d, temp_all_directions[rank][d * slice + idx]);
						continue;
					}
					
					int nidx = idx3D(ni, nj, nk);
				
					// Check if destination is a barrier
					if (barrier[nidx]) {
						continue; // Handle barrier bounce-back in pass 2
					} else {
						// Normal streaming: move to destination
						fPtr[d][nidx] += temp_all_directions[rank][d * slice + idx];
					}
				}
			}
		}
	}

	// Pass 2: Bounce-back (accumulate properly)
	for (int d = 1; d < Q; ++d) {
		for (int k = 0; k < dim_z; ++k) {
			for (int j = 0; j < dim_y; ++j) {
				for (int i = 0; i < dim_x; ++i) {
					int idx = idx3D(i, j, k);

					int ni = i + c[d][0];
					int nj = j + c[d][1];
					int nk = k + c[d][2];

					bool do_bounceback = false;

					// Check if destination is out of bounds
					if (ni < 0 || ni >= dim_x || nj < 0 || nj >= dim_y || nk < 0 || nk >= dim_z) {
						do_bounceback = true;
					} else {
						int nidx = idx3D(ni, nj, nk);
						// Check if destination is a barrier
						if (barrier[nidx]) {
							do_bounceback = true;
						}
					}

					if (do_bounceback) {
						// Bounce-back: reflect to current cell using opposite direction
						int od = opposite_dir[d];
						if (od >= 0 && od < Q) {
							fPtr[od][idx] += temp_all_directions[rank][d * slice + idx];
						}
					}
				}
			}
		}
	}
	printDblArraysGuardCorruption("stream");
}

// check if simulation has become unstable (if so, more time steps are required)
bool LbmDQ::checkStability()
{
    int i, k, idx;
    bool stable = true;
    int j = dim_y / 2;
    for (k = 0; k < dim_z; k++)
    {
	for (i = 0; i < dim_x; i++)
	    {
	        idx = idx3D(i, j, k);
		if (density[idx] <= 0)
		{
		    stable = false;
		}
	    }
    }
    return stable;
}

// compute speed (magnitude of velocity vector)
void LbmDQ::computeSpeed()
{
    int i, j, k, idx;
    for (k = 0; k < dim_z; k++)
    {
	for (j = 0; j < dim_y; j++)
	{
            for (i = 0; i < dim_x; i++)
            {
		idx = idx3D(i, j, k);
		speed[idx] = sqrt(velocity_x[idx] * velocity_x[idx] + velocity_y[idx] * velocity_y[idx] + velocity_z[idx] * velocity_z[idx]);
	    }
	}
    }
}

// compute vorticity (rotational velocity)
void LbmDQ::computeVorticity()
{
    int i; int j; int k; int idx;

    // Exchange velocity boundary data between MPI ranks first
    exchangeVelocityBoundaries();
    
    // Initialize vorticity to zero
    for (int idx = 0; idx < dim_x * dim_y * dim_z; idx++) {
        vorticity[idx] = 0.0f;
    }

    // Calculate vorticity using central differences
    for (k = 1; k < dim_z - 1; k++)
    {
	for (j = 1; j < dim_y - 1; j++)
	{
	    for (i = 1; i < dim_x - 1; i++)
	    {
		idx = idx3D(i, j, k);
		
		// Get neighboring indices for central difference
		int idx_jp1 = idx3D(i, j + 1, k);
		int idx_jm1 = idx3D(i, j - 1, k);
		int idx_kp1 = idx3D(i, j, k + 1);
		int idx_km1 = idx3D(i, j, k - 1);
		int idx_ip1 = idx3D(i + 1, j, k);
		int idx_im1 = idx3D(i - 1, j, k);

		// Calculate vorticity components using central differences
		double wx = 0.5 * (velocity_z[idx_jp1] - velocity_z[idx_jm1]) - 0.5 * (velocity_y[idx_kp1] - velocity_y[idx_km1]);
		double wy = 0.5 * (velocity_x[idx_kp1] - velocity_x[idx_km1]) - 0.5 * (velocity_z[idx_ip1] - velocity_z[idx_im1]);
		double wz = 0.5 * (velocity_y[idx_ip1] - velocity_y[idx_im1]) - 0.5 * (velocity_x[idx_jp1] - velocity_x[idx_jm1]);

		// Store magnitude of vorticity vector
		vorticity[idx] = sqrt(wx*wx + wy*wy + wz*wz);
	    }
	}
    }
    
    // Handle boundary nodes that couldn't be computed with central differences
    for (k = 0; k < dim_z; k++) {
        for (j = 0; j < dim_y; j++) {
            for (i = 0; i < dim_x; i++) {
                idx = idx3D(i, j, k);
                
                // If this is a boundary node that wasn't computed above
                if (i == 0 || i == dim_x-1 || j == 0 || j == dim_y-1 || k == 0 || k == dim_z-1) {
                    // If we're at a global boundary (not an MPI boundary), use zero vorticity
                    if ((i == 0 && offset_x == 0) || 
                        (i == dim_x-1 && offset_x + dim_x == total_x) ||
                        (j == 0 && offset_y == 0) || 
                        (j == dim_y-1 && offset_y + dim_y == total_y) ||
                        (k == 0 && offset_z == 0) || 
                        (k == dim_z-1 && offset_z + dim_z == total_z)) {
                        vorticity[idx] = 0.0f;
                    }
                    // For MPI boundaries, we can try to compute if we have neighbors
                    else if (vorticity[idx] == 0.0f) {
                        // Find nearest interior point
                        int ni = std::max(1, std::min(i, (int)dim_x - 2));
                        int nj = std::max(1, std::min(j, (int)dim_y - 2));
                        int nk = std::max(1, std::min(k, (int)dim_z - 2));
                        int nidx = idx3D(ni, nj, nk);
                        vorticity[idx] = vorticity[nidx];
                    }
                }
            }
        }
    }
}

// gather all data on rank 0
void LbmDQ::gatherDataOnRank0(FluidProperty property)
{
    float *send_buf = NULL;
    uint8_t *bsend_buf = barrier;
    switch (property)
    {
        case Density:
            send_buf = density;
            break;
        case Speed:
            send_buf = speed;
            break;
        case Vorticity:
            send_buf = vorticity;
            break;
        case None:
            return;
    }
    MPI_Status status;
    uint32_t local_size = dim_x * dim_y * dim_z;
    if (rank == 0)
    {
    	// Compute offset for rank 0's own data in the global buffer
        int n_x, n_y, n_z;
        getBest3DPartition(num_ranks, total_x, total_y, total_z, &n_x, &n_y, &n_z);
        int chunk_w = total_x / n_x;
        int chunk_h = total_y / n_y;
        int chunk_d = total_z / n_z;
        int extra_w = total_x % n_x;
        int extra_h = total_y % n_y;
        int extra_d = total_z % n_z;
        int col = 0; // rank 0
        int row = 0; // rank 0
        int layer = 0; // rank 0
        int offset_x = col * chunk_w + std::min<int>(col, extra_w);
        int offset_y = row * chunk_h + std::min<int>(row, extra_h);
        int offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
        // Compute flat offset in global buffer (C-order)
        uint32_t base = offset_x + total_x * (offset_y + total_y * offset_z);
        
        // Copy own data to correct offset in recv_buf
	for (uint32_t k = 0; k < num_z; ++k) {
                for (uint32_t j = 0; j < num_y; ++j) {
                uint32_t dst_base = base + k * total_x * total_y + j * total_x;
                uint32_t src_base = k * num_y * num_x + j * num_x;
		std::memcpy(recv_buf + dst_base, send_buf + src_base, num_x * sizeof(float));
                std::memcpy(brecv_buf + dst_base, bsend_buf + src_base, num_x * sizeof(uint8_t));
            }
        }
        
        // Receive from other ranks into correct offsets
        for (int r = 1; r < num_ranks; r++) {
            // Compute offset for rank r
            col = r % n_x;
            row = (r / n_x) % n_y;
            layer = r / (n_x * n_y);
            int r_num_x = chunk_w + ((col < extra_w) ? 1 : 0);
            int r_num_y = chunk_h + ((row < extra_h) ? 1 : 0);
            int r_num_z = chunk_d + ((layer < extra_d) ? 1 : 0);
            offset_x = col * chunk_w + std::min<int>(col, extra_w);
            offset_y = row * chunk_h + std::min<int>(row, extra_h);
            offset_z = layer * chunk_d + std::min<int>(layer, extra_d);
            uint32_t rsize = r_num_x * r_num_y * r_num_z;
            // Compute flat offset in global buffer (C-order)
            base = offset_x + total_x * (offset_y + total_y * offset_z);	    
    
	// Receive into a temporary buffer, then copy to correct offset
            std::vector<float> temp_f(rsize);
            std::vector<uint8_t> temp_b(rsize);
            MPI_Recv(temp_f.data(), rsize, MPI_FLOAT, r, TAG_F, cart_comm, &status);
            MPI_Recv(temp_b.data(), rsize, MPI_BYTE,  r, TAG_B, cart_comm, &status);
            // Copy to correct offset in recv_buf/brecv_buf
            for (uint32_t k = 0; k < r_num_z; ++k) {
                for (uint32_t j = 0; j < r_num_y; ++j) {
                    uint32_t dst_base = base + k * total_x * total_y + j * total_x;
                    uint32_t src_base = k * r_num_y * r_num_x + j * r_num_x;
                    std::memcpy(recv_buf + dst_base, temp_f.data() + src_base, r_num_x * sizeof(float));
                    std::memcpy(brecv_buf + dst_base, temp_b.data() + src_base, r_num_x * sizeof(uint8_t));
		}
            }
        }
    }
    else
    {
	MPI_Send(send_buf,  local_size, MPI_FLOAT, 0, TAG_F, cart_comm);
        MPI_Send(bsend_buf, local_size, MPI_BYTE,  0, TAG_B, cart_comm);
    }
    stored_property = property;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimX()
{
    return dim_x;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimY()
{
    return dim_y;
}

// get width of sub-area this task owns (including ghost cells)
uint32_t LbmDQ::getDimZ()
{
    return dim_z;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimX()
{
    return total_x;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimY()
{
    return total_y;
}

// get width of total area of simulation
uint32_t LbmDQ::getTotalDimZ()
{
    return total_z;
}

// get x offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetX()
{
    return offset_x;
}

// get y offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetY()
{
    return offset_y;
}

// get z offset into overall domain where this sub-area esxists
uint32_t LbmDQ::getOffsetZ()
{
    return offset_z;
}

// get x start for valid data (0 if no ghost cell on left, 1 if there is a ghost cell on left)
uint32_t LbmDQ::getStartX()
{
    return start_x;
}

// get y start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmDQ::getStartY()
{
    return start_y;
}

// get z start for valid data (0 if no ghost cell on top, 1 if there is a ghost cell on top)
uint32_t LbmDQ::getStartZ()
{
    return start_z;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeX()
{
    return num_x;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeY()
{
    return num_y;
}

// get width of sub-area this task is responsible for (excluding ghost cells)
uint32_t LbmDQ::getSizeZ()
{
    return num_z;
}

// get the local width and height of a particular rank's data
uint32_t* LbmDQ::getRankLocalSize(int rank)
{
    return rank_local_size + (2 * rank);
}

// get the local x and y start of a particular rank's data
uint32_t* LbmDQ::getRankLocalStart(int rank)
{
    return rank_local_start + (2 * rank);
}

// get barrier array
uint8_t* LbmDQ::getBarrier()
{
    if (rank != 0) return NULL;
    return brecv_buf;
}

// private - set fluid equalibrium
void LbmDQ::setEquilibrium(int x, int y, int z, double new_velocity_x, double new_velocity_y, double new_velocity_z, double new_density)
{
	int idx = idx3D(x, y, z);
	density[idx] = new_density;
	velocity_x[idx] = new_velocity_x;
	velocity_y[idx] = new_velocity_y;
	velocity_z[idx] = new_velocity_z;

	double ux = new_velocity_x;
	double uy = new_velocity_y;
	double uz = new_velocity_z;
	double usq = ux*ux + uy*uy + uz*uz;

	for (int d = 0; d < Q; ++d)
	{
		double cu = 3.0 * (c[d][0] * ux + c[d][1] * uy + c[d][2] * uz);
		f_at(d, x, y, z) = w[d] * new_density * (1.0 + cu + 0.5*cu*cu - 1.5*usq);
	}
}

// private - get 3 factors of a given number that are closest to each other and fit the domain
void LbmDQ::getBest3DPartition(int num_ranks, int dim_x, int dim_y, int dim_z, int *n_x, int *n_y, int *n_z)
{
    int best_x = 1, best_y = 1, best_z = num_ranks;
    int min_diff = num_ranks; // start with a large difference
    bool found = false;
    for (int x = 1; x <= std::min(num_ranks, (int)dim_x); ++x) {
        if (num_ranks % x != 0) continue;
        int yz = num_ranks / x;
        for (int y = 1; y <= std::min(yz, (int)dim_y); ++y) {
            if (yz % y != 0) continue;
            int z = yz / y;
            if (z > (int)dim_z) continue;
            int arr[3] = {x, y, z};
            std::sort(arr, arr + 3);
            int diff = arr[2] - arr[0];
            if (diff < min_diff) {
                min_diff = diff;
                best_x = x;
                best_y = y;
                best_z = z;
                found = true;
            }
        }
    }
    if (!found) {
        fprintf(stderr, "ERROR: Cannot partition %dx%dx%d domain among %d ranks without zero-sized subdomains.", dim_x, dim_y, dim_z, num_ranks);
        fprintf(stderr, "Try using a number of ranks that divides the domain size in at least one dimension.");
	      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    *n_x = best_x;
    *n_y = best_y;
    *n_z = best_z;
}

// Capture particles that need to be sent to neighboring MPI ranks
void LbmDQ::captureBoundaryParticle(int i, int j, int k, int ni, int nj, int nk, int d, float value)
{
    // Determine which neighbor this particle should go to
    int neighbor_idx = -1;
    int target_i = ni, target_j = nj, target_k = nk;

    // Map out-of-bounds coordinates to neighbor domains
    if (ni < 0) {
        neighbor_idx = NeighborW;
        target_i = dim_x - 1;  // Goes to rightmost layer of west neighbor
    } else if (ni >= dim_x) {
        neighbor_idx = NeighborE;
        target_i = 0;  // Goes to leftmost layer of east neighbor
    }

    if (nj < 0) {
        if (neighbor_idx == -1) {
            neighbor_idx = NeighborS;
            target_j = dim_y - 1;  // Goes to top layer of south neighbor
        } else {
            // Corner case - need to handle diagonal neighbors
            if (neighbor_idx == NeighborW) neighbor_idx = NeighborSW;
            else if (neighbor_idx == NeighborE) neighbor_idx = NeighborSE;
            target_j = dim_y - 1;
        }
    } else if (nj >= dim_y) {
        if (neighbor_idx == -1) {
            neighbor_idx = NeighborN;
            target_j = 0;  // Goes to bottom layer of north neighbor
        } else {
            // Corner case - need to handle diagonal neighbors
            if (neighbor_idx == NeighborW) neighbor_idx = NeighborNW;
            else if (neighbor_idx == NeighborE) neighbor_idx = NeighborNE;
            target_j = 0;
        }
    }

    if (nk < 0) {
        neighbor_idx = NeighborDown;
        target_k = dim_z - 1;  // Goes to top layer of down neighbor
    } else if (nk >= dim_z) {
        neighbor_idx = NeighborUp;
        target_k = 0;  // Goes to bottom layer of up neighbor
    }

    // Store particle in appropriate send buffer
    if (neighbor_idx >= 0 && neighbor_idx < 10 && neighbors[neighbor_idx] != MPI_PROC_NULL) {
        send_buffers[neighbor_idx].addParticle(value, d, target_i, target_j, target_k);
    }
}

// private - exchange boundary information between MPI ranks
void LbmDQ::exchangeBoundaries()
{
    // Clear previous receive buffers
    for (int i = 0; i < 10; ++i) {
        recv_buffers[i].clear();
    }

    // Exchange particle buffers with all neighbors
    for (int neighbor_idx = 0; neighbor_idx < 10; ++neighbor_idx) {
        if (neighbors[neighbor_idx] == MPI_PROC_NULL) continue;

        ParticleBuffer& send_buf = send_buffers[neighbor_idx];
        ParticleBuffer& recv_buf = recv_buffers[neighbor_idx];

        // Exchange buffer sizes first
        int send_size = send_buf.particles.size();
        int recv_size = 0;

        MPI_Sendrecv(&send_size, 1, MPI_INT, neighbors[neighbor_idx], 0,
                     &recv_size, 1, MPI_INT, neighbors[neighbor_idx], 0,
                     cart_comm, MPI_STATUS_IGNORE);

        if (recv_size > 0) {
            // Resize receive buffers
            recv_buf.particles.resize(recv_size);
            recv_buf.directions.resize(recv_size);
            recv_buf.positions.resize(recv_size);

            // Exchange particle data
            if (send_size > 0) {
                MPI_Sendrecv(send_buf.particles.data(), send_size, MPI_FLOAT, neighbors[neighbor_idx], 1,
                             recv_buf.particles.data(), recv_size, MPI_FLOAT, neighbors[neighbor_idx], 1,
                             cart_comm, MPI_STATUS_IGNORE);

                MPI_Sendrecv(send_buf.directions.data(), send_size, MPI_INT, neighbors[neighbor_idx], 2,
                             recv_buf.directions.data(), recv_size, MPI_INT, neighbors[neighbor_idx], 2,
                             cart_comm, MPI_STATUS_IGNORE);

                MPI_Sendrecv(send_buf.positions.data(), send_size, MPI_INT, neighbors[neighbor_idx], 3,
                             recv_buf.positions.data(), recv_size, MPI_INT, neighbors[neighbor_idx], 3,
                             cart_comm, MPI_STATUS_IGNORE);
            } else {
                // Only receive
                MPI_Recv(recv_buf.particles.data(), recv_size, MPI_FLOAT, neighbors[neighbor_idx], 1,
                         cart_comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_buf.directions.data(), recv_size, MPI_INT, neighbors[neighbor_idx], 2,
                         cart_comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_buf.positions.data(), recv_size, MPI_INT, neighbors[neighbor_idx], 3,
                         cart_comm, MPI_STATUS_IGNORE);
            }
        } else if (send_size > 0) {
            // Only send
            MPI_Send(send_buf.particles.data(), send_size, MPI_FLOAT, neighbors[neighbor_idx], 1, cart_comm);
            MPI_Send(send_buf.directions.data(), send_size, MPI_INT, neighbors[neighbor_idx], 2, cart_comm);
            MPI_Send(send_buf.positions.data(), send_size, MPI_INT, neighbors[neighbor_idx], 3, cart_comm);
        }
    }

    // Inject received particles into domain
    for (int neighbor_idx = 0; neighbor_idx < 10; ++neighbor_idx) {
        ParticleBuffer& recv_buf = recv_buffers[neighbor_idx];

        for (size_t p = 0; p < recv_buf.particles.size(); ++p) {
            int i, j, k;
            recv_buf.unpackPosition(recv_buf.positions[p], i, j, k);

            // Validate coordinates
            if (i >= 0 && i < dim_x && j >= 0 && j < dim_y && k >= 0 && k < dim_z) {
                int idx = idx3D(i, j, k);
                int d = recv_buf.directions[p];

                // Inject particle
                if (d >= 0 && d < Q) {
                    fPtr[d][idx] = recv_buf.particles[p];
                }
            }
        }
    }

    // Clear send buffers for next timestep
    for (int i = 0; i < 10; ++i) {
        send_buffers[i].clear();
    }
}

// Exchange velocity boundary data between neighboring MPI ranks for vorticity calculation
void LbmDQ::exchangeVelocityBoundaries()
{
    // Handle X-direction boundaries
    for (int k = 0; k < dim_z; k++) {
        for (int j = 0; j < dim_y; j++) {
            // Left boundary (i=0): copy from i=1
            if (dim_x > 1) {
                int boundary_idx = idx3D(0, j, k);
                int interior_idx = idx3D(1, j, k);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }

            // Right boundary (i=dim_x-1): copy from i=dim_x-2
            if (dim_x > 1) {
                int boundary_idx = idx3D(dim_x - 1, j, k);
                int interior_idx = idx3D(dim_x - 2, j, k);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }
        }
    }

    // Handle Y-direction boundaries
    for (int k = 0; k < dim_z; k++) {
        for (int i = 0; i < dim_x; i++) {
            // Back boundary (j=0): copy from j=1
            if (dim_y > 1) {
                int boundary_idx = idx3D(i, 0, k);
                int interior_idx = idx3D(i, 1, k);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }

            // Front boundary (j=dim_y-1): copy from j=dim_y-2
            if (dim_y > 1) {
                int boundary_idx = idx3D(i, dim_y - 1, k);
                int interior_idx = idx3D(i, dim_y - 2, k);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }
        }
    }

    // Handle Z-direction boundaries
    for (int j = 0; j < dim_y; j++) {
        for (int i = 0; i < dim_x; i++) {
            // Bottom boundary (k=0): copy from k=1
            if (dim_z > 1) {
                int boundary_idx = idx3D(i, j, 0);
                int interior_idx = idx3D(i, j, 1);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }

            // Top boundary (k=dim_z-1): copy from k=dim_z-2
            if (dim_z > 1) {
                int boundary_idx = idx3D(i, j, dim_z - 1);
                int interior_idx = idx3D(i, j, dim_z - 2);
                velocity_x[boundary_idx] = velocity_x[interior_idx];
                velocity_y[boundary_idx] = velocity_y[interior_idx];
                velocity_z[boundary_idx] = velocity_z[interior_idx];
            }
        }
    }
}

#endif // _LBMDQ_MPI_HPP_
