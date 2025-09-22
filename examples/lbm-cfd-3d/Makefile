# LBM-CFD Makefile
# Supports compilation, execution, and post-processing automation

# Check for OS (Windows, Linux, Mac OS)
ifeq ($(OS),Windows_NT)
	DETECTED_OS:= Windows
else
	DETECTED_OS:= $(shell uname)
endif

# Set compiler and flags
ifeq ($(DETECTED_OS),Windows)
CXX= g++
MPIRUN= mpiexec
else
CXX= mpic++
MPIRUN= mpirun
endif
CXXFLAGS+= -std=c++14 -O3 -Wall -Wno-sign-compare -DOUTPUT_VORTICITY=$(OUTPUT_VORTICITY) -DOUTPUT_VELOCITY=$(OUTPUT_VELOCITY)

# Set application executable name
APP= lbmcfd3d

# Set source and output directories
SRCDIR= src
OBJDIR= obj
BINDIR= bin

# Simulation parameters (can be overridden)
LATTICE ?= d3q19
N ?= 1

# MPI configuration (can be overridden for different systems)
PPN ?=

# Output configuration parameters
OUTPUT_VORTICITY ?= 0
OUTPUT_VELOCITY ?= 0

# Set up include and library directories
ifeq ($(DETECTED_OS),Windows)
	MPI_INC= $(patsubst %\,%,$(MSMPI_INC))
	MPI_LIB= $(patsubst %\,%,$(MSMPI_LIB64))

	INC= -I"$(MPI_INC)" -I"$(MPI_INC)\x64" -I.\include
	LIB= -L"$(MPI_LIB)" -lmsmpi -lm
else
	INC= -I./include
	LIB= -lm
endif



# Create output directories and set output file names
ifeq ($(DETECTED_OS),Windows)
	mkobjdir:= $(shell if not exist $(OBJDIR) mkdir $(OBJDIR))
	mkbindir:= $(shell if not exist $(BINDIR) mkdir $(BINDIR))

	OBJS= $(addprefix $(OBJDIR)\, main.o)
	EXEC= $(addprefix $(BINDIR)\, $(APP).exe)
else
	mkdirs:= $(shell mkdir -p $(OBJDIR) $(BINDIR))
	
	OBJS= $(addprefix $(OBJDIR)/, main.o)
	EXEC= $(addprefix $(BINDIR)/, $(APP))
endif


# BUILD EVERYTHING
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) -o $@ $^ $(LIB)

ifeq ($(DETECTED_OS),Windows)
$(OBJDIR)\\%.o: $(SRCDIR)\%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
else
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(INC)
endif


# REMOVE OLD FILES
ifeq ($(DETECTED_OS),Windows)
clean:
	if exist $(OBJDIR) rmdir /s /q $(OBJDIR)
	if exist $(BINDIR) rmdir /s /q $(BINDIR)
	del /q paraview\*.vts paraview\*.pvd 2>nul || echo No VTS/PVD files to clean in paraview
else
clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -f paraview/*.vts paraview/*.pvd
	rm -f hostfile.tmp
endif

# PHONY targets
.PHONY: all clean help run pvd complete

# Help target
help:
	@echo "LBM-CFD Makefile Usage:"
	@echo ""
	@echo "  make all             - Compile the application (default)"
	@echo "  make run             - Run simulation with MPI (use in job/qsub)"
	@echo "  make pvd             - Generate PVD files for ParaView"
	@echo "  make complete        - Compile + Run + Generate PVD (use in job/qsub)"
	@echo "  make clean           - Remove compiled files and results"
	@echo ""
	@echo "Parameters:"
	@echo "  LATTICE={d3q15,d3q19,d3q27}  - Lattice model (default: $(LATTICE))"
	@echo "  N=num                        - MPI processes (default: $(N))"
	@echo "  PPN=num                      - Processes per node (default: auto)"
	@echo "  OUTPUT_VORTICITY={0,1}       - Enable vorticity output (default: $(OUTPUT_VORTICITY))"
	@echo "  OUTPUT_VELOCITY={0,1}        - Enable velocity vector output (default: $(OUTPUT_VELOCITY))"

# Build command-line arguments
OUTPUT_FLAGS=
ifeq ($(OUTPUT_VORTICITY),1)
	OUTPUT_FLAGS += --output-vorticity
endif
ifeq ($(OUTPUT_VELOCITY),1)
	OUTPUT_FLAGS += --output-velocity
endif

# Run simulation
run: $(EXEC)
	@echo ""
	@echo "Starting LBM-CFD simulation..."
	@echo "Parameters: $(N) processes, --$(LATTICE) $(OUTPUT_FLAGS)"
ifneq ($(PPN),)
	@echo "PPN: $(PPN) processes per node"
endif
ifeq ($(DETECTED_OS),Windows)
	@echo "MPI Command: $(MPIRUN) -n $(N) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)"
	$(MPIRUN) -n $(N) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)
else
ifneq ($(PPN),)
	@if [ -n "$$PBS_NODEFILE" ]; then \
		echo "Using PBS nodefile: $$PBS_NODEFILE"; \
		echo "MPI Command: $(MPIRUN) -n $(N) --hostfile $$PBS_NODEFILE --ppn $(PPN) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)"; \
		$(MPIRUN) -n $(N) --hostfile $$PBS_NODEFILE --ppn $(PPN) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS); \
	else \
		echo "localhost" > hostfile.tmp; \
		echo "MPI Command: $(MPIRUN) -n $(N) --hostfile hostfile.tmp --ppn $(PPN) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)"; \
		$(MPIRUN) -n $(N) --hostfile hostfile.tmp --ppn $(PPN) $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS); \
		rm -f hostfile.tmp; \
	fi
else
	@if [ -n "$$PBS_NODEFILE" ]; then \
		echo "Using PBS nodefile: $$PBS_NODEFILE"; \
		echo "MPI Command: $(MPIRUN) -n $(N) --hostfile $$PBS_NODEFILE $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)"; \
		$(MPIRUN) -n $(N) --hostfile $$PBS_NODEFILE $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS); \
	else \
		echo "localhost" > hostfile.tmp; \
		echo "MPI Command: $(MPIRUN) -n $(N) --hostfile hostfile.tmp $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS)"; \
		$(MPIRUN) -n $(N) --hostfile hostfile.tmp $(EXEC) --$(LATTICE) $(OUTPUT_FLAGS); \
		rm -f hostfile.tmp; \
	fi
endif
endif
	@echo ""
	@echo "Simulation completed."

# Generate PVD files for ParaView (only if VTS files exist)
pvd:
ifeq ($(DETECTED_OS),Windows)
	@if exist "paraview\*.vts" ( \
		echo "Generating PVD files for ParaView..." && \
		if exist "create_pvd_from_vts.py" ( \
			python "create_pvd_from_vts.py" --input-dir paraview \
		) else ( \
			echo Warning: create_pvd_from_vts.py not found \
		) && \
		echo "PVD files ready for ParaView." \
	) else ( \
		echo "No VTS files found - skipping PVD generation" \
	)
else
	@if ls paraview/*.vts >/dev/null 2>&1; then \
		echo "Generating PVD files for ParaView..."; \
		if [ -f "create_pvd_from_vts.py" ]; then \
			python3 create_pvd_from_vts.py --input-dir paraview; \
		else \
			echo "Warning: create_pvd_from_vts.py not found"; \
		fi; \
		echo "PVD files ready for ParaView."; \
	else \
		echo "No VTS files found - skipping PVD generation"; \
	fi
endif

# Complete workflow: compile + run + generate PVD
complete: all run pvd
	@echo "Complete workflow finished!"
	@echo "Ready for ParaView visualization."
	@echo ""
