# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
export MV2_ENABLE_AFFINITY=0

# Single GPU
#benchmarks/hpgmg/hpgmg-cuda/build/bin/hpgmg-fv 4 5
#./build/bin/hpgmg-fv 4 5
/dmtcp-cuda/contrib/cuda/benchmarks/hpgmg/hpgmg-cuda/build/bin/hpgmg-fv 4 5
# MPI, one rank per GPU
#mpirun -np 2 ./build/bin/hpgmg-fv 7 8
