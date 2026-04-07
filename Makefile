# ============================================================
#  Makefile  –  DEM Solver  (serial + OpenMP)
#  Usage:
#    make          → build serial  (dem_serial)
#    make omp      → build OpenMP  (dem_omp)
#    make all      → build both
#    make run      → run serial
#    make run_omp  → run OpenMP (16 threads)
#    make clean    → remove binaries and CSVs
# ============================================================

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
OMPFLAGS = -fopenmp

SRC_SERIAL = dem.cpp
SRC_OMP    = dem_omp.cpp

BIN_SERIAL = dem_serial
BIN_OMP    = dem_omp

.PHONY: all serial omp run run_omp clean

all: serial omp

serial: $(SRC_SERIAL)
	$(CXX) $(CXXFLAGS) -o $(BIN_SERIAL) $(SRC_SERIAL)
	@echo "Built: $(BIN_SERIAL)"

omp: $(SRC_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $(BIN_OMP) $(SRC_OMP)
	@echo "Built: $(BIN_OMP)"

run: serial
	./$(BIN_SERIAL)

run_omp: omp
	OMP_NUM_THREADS=16 ./$(BIN_OMP)

clean:
	rm -f $(BIN_SERIAL) $(BIN_OMP) *.csv *.o
