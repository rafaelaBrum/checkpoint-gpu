# Building `hpgmg-cuda` on Yaksha:
  cd hpmpg-cuda
  ./build.sh

# Running `hpgmg-cuda` on Yaksha Single GPU. See `hpgmg-cuda/run.sh` for more details:
  cd hpmpg-cuda
  export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
  export OMP_NUM_THREADS=4
  export MV2_ENABLE_AFFINITY=0
  ./build/bin/hpgmg-fv 4 5

