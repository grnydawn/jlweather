
# user info
ACCOUNT=<your account>
USERNAME=<your username>

# change working directory
WORK_DIR="/lustre/orion/${ACCOUNT}/proj-shared/${USERNAME}/jaiwork"

# un-comment one of the following commands:

make julia   WORK_DIR="${WORK_DIR}"

#make jai_all   WORK_DIR="${WORK_DIR}" FORTRAN="ftn -fPIC -shared -h noacc,noomp" 
#make jai_all   WORK_DIR="${WORK_DIR}" FOPENACC="ftn -shared -fPIC -h acc,noomp" 
#make jai_all   WORK_DIR="${WORK_DIR}" FOMPTARGET="ftn -shared -fPIC -h omp,noacc" 
#make jai_all   WORK_DIR="${WORK_DIR}" CUDA="nvcc --linker-options=\"-fPIC\" --compiler-options=\"-fPIC\" --shared -g" 
#make jai_all   WORK_DIR="${WORK_DIR}" HIP="hipcc -shared -fPIC -lamdhip64 -g" 

#make jai_fortran   WORK_DIR="${WORK_DIR}" FORTRAN="ftn -fPIC -shared -h noacc,noomp" 

#make jai_fhip1   WORK_DIR="${WORK_DIR}" FORTRAN="ftn -fPIC -shared -h noacc,noomp" HIP="hipcc -shared -fPIC -lamdhip64 -g"
#make jai_fhip2   WORK_DIR="${WORK_DIR}" FORTRAN="ftn -fPIC -shared -h noacc,noomp" HIP="hipcc -shared -fPIC -lamdhip64 -g"

#make fortran FORTRAN="ftn -h noacc,noomp"

#make fopenacc FOPENACC="ftn -h acc,noomp"

#make fomptarget FOMPTARGET="ftn -h omp,noacc"

#make julia_manual WORK_DIR="${WORK_DIR}" FOPENACC="ftn -fPIC -shared -h acc,noomp"
