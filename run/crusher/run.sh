 #srun -n 1 -- make jai > prof_flat_n1.out
 #srun -n 2 -- make jai > prof_flat_n2.out
 #srun -n 4 -- make jai > prof_flat_n4.out
 #srun -n 8 -- make jai > prof_flat_n8.out

echo "4000 * 2000 grid test - started"

echo "compiling fortran mpi version"
make fort

echo "compiling fortran mpi openacc version"
make fortacc

echo "run fortran mpi version with 64 ranks"
srun -c 1 -m *:block -n 64 --  ./miniweather_fort.exe

echo "run fortran mpi openacc version with 1 rank"
srun -c 1 -m *:block -n 1 --  ./miniweather_fortacc.exe

echo "run fortran mpi openacc version with 8 ranks"
srun -c 1 -m *:block -n 8 --  ./miniweather_fortacc.exe

echo "run julia mpi version with 64 ranks"
srun -c 1 -m *:block -n 64 --  make julia

echo "run julia mpi openacc version with 1 rank"
srun -c 1 -m *:block -n 1 --  make jai

echo "4000 * 2000 grid test - ended"
