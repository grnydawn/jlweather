# example command: srun  -n 2 -- make

#SIM_TIME := 10.0
SIM_TIME := 100.0
#SIM_TIME := 1000.0

OUT_FREQ := 10.0
#OUT_FREQ := 100.0

#NX := 100
#NZ := 50

NX  := 400
NZ  := 200

#NX  := 1000
#NZ  := 500

#NX  := 1400
#NZ  := 700

#NX  := 2000
#NZ  := 1000

#NX  := 3000
#NZ  := 1500

#NX  := 4000
#NZ  := 2000

#SIM_TIME := 10.0 #10.0
#OUT_FREQ := 10.0 # 10.0
#NX := 6400 # 640 
#NZ := 100 # 100

DATA_SPEC_COLLISION       := 1
DATA_SPEC_THERMAL         := 2
DATA_SPEC_GRAVITY_WAVES   := 3
DATA_SPEC_DENSITY_CURRENT := 5
DATA_SPEC_INJECTION       := 6

DATA_SPEC := ${DATA_SPEC_COLLISION}
##DATA_SPEC := ${DATA_SPEC_THERMAL}
#DATA_SPEC := ${DATA_SPEC_GRAVITY_WAVES} # not working
#DATA_SPEC := ${DATA_SPEC_DENSITY_CURRENT}
#DATA_SPEC := ${DATA_SPEC_INJECTION}

ACCEL_TYPE := fortran_openacc
WORK_DIR  := /lustre/orion/cli115/proj-shared/grnydawn/jaiwork
DEBUG_DIR := ${WORK_DIR}
OUT_FILE  := /lustre/orion/cli115/proj-shared/grnydawn/output.nc

JULIA := julia --project=. --check-bounds=no
#JULIA := julia --project=.

SRCDIR :=../../src
JAIFORTRANSRC := ${SRCDIR}/jai/fortran/miniWeather_accel.jl
JAIFORTHIP0SRC := ${SRCDIR}/jai/forthip0/miniWeather_accel.jl
JAIFORTHIP1SRC := ${SRCDIR}/jai/forthip1/miniWeather_accel.jl
JAIFORTHIP2SRC := ${SRCDIR}/jai/forthip2/miniWeather_accel.jl
JAIJULIA2SRC := ${SRCDIR}/jai/forthip2/miniWeather_julia.jl
JAIFORTHIP3SRC := ${SRCDIR}/jai/forthip3/miniWeather_accel.jl
JAIFORTOMPSRC := ${SRCDIR}/jai/fortran_omptarget/miniWeather_accel.jl
JAIFORTACCSRC := ${SRCDIR}/jai/fortran_openacc/miniWeather_accel.jl
JAIJULIA3SRC := ${SRCDIR}/jai/forthip3/miniWeather_julia.jl
JLSRC := ${SRCDIR}/julia/miniWeather_mpi.jl
FORTSRC := ${SRCDIR}/fortran/miniWeather_mpi.F90
FORTACCSRC := ${SRCDIR}/fortran/miniWeather_mpi_openacc.F90
FORTOMPSRC := ${SRCDIR}/fortran/miniWeather_mpi_openmp45.F90
MANUALJLSRC := ${SRCDIR}/manual/miniWeather_openacc.jl
MANUALFORTSRC := ${SRCDIR}/manual/openacc_driver_${NX}x${NZ}.F90

ARGS := -s ${SIM_TIME} -x ${NX} -z ${NZ} -f ${OUT_FREQ} \
		-d ${DATA_SPEC} -o ${OUT_FILE} -w ${WORK_DIR} \
		-b ${DEBUG_DIR}

CC := cc
CXX := CC
FC := ftn

INCLUDES := -I${OLCF_PARALLEL_NETCDF_ROOT}/include
LIBS := -L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf
MACROS := -D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}

F_FLAGS := ${INCLUDES} ${LIBS} ${MACROS}

jai_fortran:
	echo "*** JAI FORTRAN ***"
	${JULIA} --color=yes -- ${JAIFORTRANSRC} ${ARGS}

jai_hip0:
	echo "*** JAI HIP ***"
	${JULIA} --color=yes -- ${JAIFORTHIP0SRC} ${ARGS}

jai_forthip1:
	${JULIA} --color=yes -- ${JAIFORTHIP1SRC} ${ARGS}

jai_forthip2:
	${JULIA} --color=yes -- ${JAIFORTHIP2SRC} ${ARGS}

jai_julia2:
	${JULIA} --color=yes -- ${JAIJULIA2SRC} ${ARGS}

jai_forthip3:
	${JULIA} --color=yes -- ${JAIFORTHIP3SRC} -p fortran_openacc ${ARGS}

jai_fomp1:
	echo "*** JAI FORTRAN OPENMP TARGET ***"
	${JULIA} --color=yes -- ${JAIFORTOMPSRC} ${ARGS}

jai_facc1:
	echo "*** JAI FORTRAN OPENACC ***"
	${JULIA} --color=yes -- ${JAIFORTACCSRC} ${ARGS}

jai_julia3:
	${JULIA} --color=yes -- ${JAIJULIA3SRC} ${ARGS}

julia:
	echo "*** JULIA ***"
	${JULIA} --color=yes -- ${JLSRC} ${ARGS}

debug:
	${JULIA} --color=yes --check-bounds=yes -O0 ${SRC} ${ARGS}

fortran: ${FORTSRC}
	echo "*** FORTRAN ***"
	${FC} ${F_FLAGS} -DNO_INFORM -h noacc,noomp -o miniweather_fort.exe $<

fortacc: ${FORTACCSRC}
	echo "*** FORTRAN OPENACC ***"
	${FC} ${F_FLAGS} -DNO_INFORM -h acc,noomp -o miniweather_fortacc.exe $<

fortomp: ${FORTOMPSRC}
	echo "*** FORTRAN OPENMP TARGET ***"
	${FC} ${F_FLAGS} -DNO_INFORM -h noacc,omp -o miniweather_fortomp.exe $<

manual: ${MANUALFORTSRC}
	ftn -shared -fPIC -h acc,noomp -o ${WORK_DIR}/jopenacc.so $<
	${JULIA} ${MANUALJLSRC} ${ARGS}

plot:
	${JULIA} miniW_output_display.jl

all: fort fortacc julia jai
	./miniweather_fort.exe
	./ miniweather_fortacc.exe

clean:
	rm -rf *.so core *.mod .jaitmp *.o *.s *.exe output.nc data_roofline results.csv
