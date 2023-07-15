make fort
sleep 1
srun -n 56 ./miniweather_fort.exe
sleep 1

make fortacc
sleep 1
./miniweather_fortacc.exe
sleep 1

make fortomp
sleep 1
./miniweather_fortomp.exe
sleep 1

srun -n 56 -- make julia
sleep 1

srun -n 56 -- make jai_fortran
sleep 1
srun -n 56 -- make jai_fortran
sleep 1

make jai_fomp1
sleep 1
make jai_fomp1
sleep 1

make jai_facc1
sleep 1
make jai_facc1
sleep 1
