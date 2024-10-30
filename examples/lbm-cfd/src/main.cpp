#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>

#ifdef ASCENT_ENABLED
#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#endif

#include "lbmd2q9_mpi.hpp"

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t time_steps, void *ptr);
void createDivergingColorMap(uint8_t *cmap, uint32_t size);
#ifdef ASCENT_ENABLED
void updateAscentData(int rank, int step, double time, conduit::Node &data);
void runAscentInSituTasks(conduit::Node &data, ascent::Ascent *ascent_ptr);
void steeringCallback(conduit::Node &params, conduit::Node &output);
#endif
int32_t readFile(const char *filename, char** data_ptr);

// global vars for LBM and Barriers
std::vector<Barrier*> barriers;
LbmD2Q9 *lbm;

int main(int argc, char **argv) {
    int rc, rank, num_ranks;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (rc != 0)
    {
        std::cerr << "Error initializing MPI" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    uint32_t dim_x = 600;
    uint32_t dim_y = 240;
    uint32_t time_steps = 20000;

    if (rank == 0) std::cout << "LBM-CFD> running with " << num_ranks << " processes" << std::endl;
    if (rank == 0) std::cout << "LBM-CFD> resolution=" << dim_x << "x" << dim_y << ", time steps=" << time_steps << std::endl;

    void *ascent_ptr = NULL;

#ifdef ASCENT_ENABLED
    if (rank == 0) std::cout << "LBM-CFD> Ascent in situ: ENABLED" << std::endl;
    
    // Copy MPI Communicator to use with Ascent
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    // Create Ascent object
    ascent::Ascent ascent;

    // Set Ascent options
    conduit::Node ascent_opts;
    ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
    ascent.open(ascent_opts);

    ascent::register_callback("steeringCallback", steeringCallback);

    ascent_ptr = &ascent;
#endif

    // Run simulation
    runLbmCfdSimulation(rank, num_ranks, dim_x, dim_y, time_steps, ascent_ptr);

#ifdef ASCENT_ENABLED
    ascent.close();
#endif

    MPI_Finalize();

    return 0;
}

void runLbmCfdSimulation(int rank, int num_ranks, uint32_t dim_x, uint32_t dim_y, uint32_t time_steps, void *ptr)
{
    // simulate corn syrup at 25 C in a 2 m pipe, moving 0.75 m/s for 8 sec
    double physical_density = 1380.0;     // kg/m^3
    double physical_speed = 0.75;         // m/s
    double physical_length = 2.0;         // m
    double physical_viscosity = 1.3806;   // Pa s
    double physical_time = 8.0;           // s
    double physical_freq = 0.25;//0.04;          // s
    double reynolds_number = (physical_density * physical_speed * physical_length) / physical_viscosity;
    
    // convert physical properties into simulation properties
    double simulation_dx = physical_length / (double)dim_y;
    double simulation_dt = physical_time / (double)time_steps;
    double simulation_speed_scale = simulation_dt / simulation_dx;
    double simulation_speed = simulation_speed_scale * physical_speed;
    double simulation_viscosity = simulation_dt / (simulation_dx * simulation_dx * reynolds_number);
    
    // output simulation properties
    if (rank == 0)
    {
        std::cout << std::fixed << std::setprecision(6) << "LBM-CFD> speed: " << simulation_speed << ", viscosity: " <<
                     simulation_viscosity << ", reynolds: " << reynolds_number << "\n" << std::endl;
    }
    
    // create LBM object
    lbm = new LbmD2Q9(dim_x, dim_y, simulation_speed_scale, rank, num_ranks);
    
    // initialize simulation
    // barrier: center-gap
    barriers.push_back(new BarrierVertical( 8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, dim_x / 8));
    barriers.push_back(new BarrierVertical( 8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, dim_x / 8 + 1));
    barriers.push_back(new BarrierVertical(13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8));
    barriers.push_back(new BarrierVertical(13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8 + 1));
    // barrier: offset-mid
    //barriers.push_back(new BarrierVertical( 8 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8));
    //barriers.push_back(new BarrierVertical( 8 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8 + 1));
    lbm->initBarrier(barriers);
    lbm->initFluid(physical_speed);
    
    // sync all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // run simulation
    int t;
    double time;
    int output_count = 0;
    double next_output_time = 0.0;
    uint8_t stable, all_stable;
    for (t = 0; t < time_steps; t++)
    {
        // output data at frequency equivalent to `physical_freq` time
        time = t * simulation_dt;
        if (time >= next_output_time)
        {
            if (rank == 0)
            {
                std::cout << std::fixed << std::setprecision(3) << "LBM-CFD> time: " << time << " / " <<
                             physical_time << " , time step: " << t << " / " << time_steps << std::endl;
            }
            stable = lbm->checkStability();
            MPI_Reduce(&stable, &all_stable, 1, MPI_UNSIGNED_CHAR, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!all_stable && rank == 0)
            {
                std::cerr << "LBM-CFD> Warning: simulation has become unstable (more time steps needed)" << std::endl;
            }
            
#ifdef ASCENT_ENABLED
            ascent::Ascent *ascent_ptr = static_cast<ascent::Ascent*>(ptr);
            conduit::Node data;
            updateAscentData(rank, t, time, data);
            runAscentInSituTasks(data, ascent_ptr);
#endif
            output_count++;
            next_output_time = output_count * physical_freq;
        }
        
        // perform one iteration of the simulation
        lbm->collide(simulation_viscosity);
        lbm->stream();
        lbm->bounceBackStream();
    }

    // Clean up
    delete lbm;
}

#ifdef ASCENT_ENABLED
void updateAscentData(int rank, int step, double time, conduit::Node &data)
{
    // Gather data on rank 0
    lbm->computeVorticity();

    uint32_t dim_x = lbm->getDimX();
    uint32_t dim_y = lbm->getDimY();
    uint32_t offset_x = lbm->getOffsetX();
    uint32_t offset_y = lbm->getOffsetY();
    uint32_t prop_size = dim_x * dim_y;

    int *barrier_data = new int[barriers.size() * 4];
    int i;
    for (i = 0; i < barriers.size(); i++)
    {
        barrier_data[4 * i + 0] = barriers[i]->getX1();
        barrier_data[4 * i + 1] = barriers[i]->getY1();
        barrier_data[4 * i + 2] = barriers[i]->getX2();
        barrier_data[4 * i + 3] = barriers[i]->getY2();
    }

    data["state/cycle"] = step;
    data["state/time"] = time;
    data["state/domain_id"] = rank;
    data["state/total_size/w"] = lbm->getTotalDimX();
    data["state/total_size/h"] = lbm->getTotalDimY();
    data["state/num_barriers"] = barriers.size();
    data["state/barriers"].set(barrier_data, barriers.size() * 4);
    
    data["coordsets/coords/type"] = "uniform";
    data["coordsets/coords/dims/i"] = dim_x;
    data["coordsets/coords/dims/j"] = dim_y;

    data["coordsets/coords/origin/x"] = offset_x;
    data["coordsets/coords/origin/y"] = offset_y;
    data["coordsets/coords/spacing/dx"] = 1;
    data["coordsets/coords/spacing/dy"] = 1;

    data["topologies/topo/type"] = "uniform";
    data["topologies/topo/coordset"] = "coords";
    
    data["fields/vorticity/association"] = "vertex";
    data["fields/vorticity/topology"] = "topo";
    data["fields/vorticity/values"].set_external(lbm->getVorticity(), prop_size);

    delete[] barrier_data;
}

void runAscentInSituTasks(conduit::Node &data, ascent::Ascent *ascent_ptr)
{
    ascent_ptr->publish(data);

    conduit::Node actions;
    conduit::Node &add_extracts = actions.append();
    add_extracts["action"] = "add_extracts";
    conduit::Node &extracts = add_extracts["extracts"];

    char *py_script;
    if (readFile("ascent/ascent_trame_bridge.py", &py_script) >= 0)
    {
        extracts["e1/type"] = "python";
        extracts["e1/params/source"] = py_script;
    }

    ascent_ptr->execute(actions);
}

void steeringCallback(conduit::Node &params, conduit::Node &output)
{
    if (params.has_path("task_id") && params.has_path("flow_speed") && params.has_path("num_barriers") && params.has_path("barriers"))
    {
        int rank = (int)params["task_id"].as_int64();
        double flow_speed = params["flow_speed"].as_float64();
        int num_barriers = (int)params["num_barriers"].as_int64();
        int32_t *new_barriers = params["barriers"].as_int32_ptr();
        
        int i;
        barriers.clear();
        for (i = 0; i < num_barriers; i++)
        {
            int x1 = new_barriers[4 * i + 0];
            int y1 = new_barriers[4 * i + 1];
            int x2 = new_barriers[4 * i + 2];
            int y2 = new_barriers[4 * i + 3];
            if (x1 == x2)
            {
                barriers.push_back(new BarrierVertical(std::min(y1, y2), std::max(y1, y2), x1));
            }
            else if (y1 == y2)
            {
                barriers.push_back(new BarrierHorizontal(std::min(x1, x2), std::max(x1, x2), y1));
            }
        }
        lbm->initBarrier(barriers);
        lbm->updateFluid(flow_speed);
    }
}
#endif

int32_t readFile(const char *filename, char** data_ptr)
{
    FILE *fp;
    int err = 0;
#ifdef _WIN32
    err = fopen_s(&fp, filename, "rb");
#else
    fp = fopen(filename, "rb");
#endif
    if (err != 0 || fp == NULL)
    {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int32_t fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    *data_ptr = (char*)malloc(fsize + 1);
    size_t read = fread(*data_ptr, fsize, 1, fp);
    if (read != 1)
    {
        std::cerr << "Error: cannot read " << filename <<std::endl;
        return -1;
    }
    (*data_ptr)[fsize] = '\0';

    fclose(fp);

    return fsize;
}
