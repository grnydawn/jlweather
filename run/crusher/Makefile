
FORTRAN    ?=
FOPENACC   ?=
FOMPTARGET ?=
CUDA       ?=
HIP        ?=

ARGS :=

ifneq ("${FORTRAN}", "")
    ARGS := ${ARGS} --fortran="${FORTRAN}"
endif

ifneq ("${FOPENACC}", "")
    ARGS := ${ARGS} --fopenacc="${FOPENACC}"
endif

ifneq ("${FOMPTARGET}", "")
    ARGS := ${ARGS} --fomptarget="${FOMPTARGET}"
endif

ifneq ("${CUDA}", "")
    ARGS := ${ARGS} --cuda="${CUDA}"
endif

ifneq ("${HIP}", "")
    ARGS := ${ARGS} --hip="${HIP}"
endif

#SIM_TIME ?= 10.0
SIM_TIME ?= 100.0
#SIM_TIME ?= 1000.0

OUT_FREQ ?= 10.0
#OUT_FREQ ?= 100.0

#NX ?= 100
#NZ ?= 50

NX  ?= 400
NZ  ?= 200

#NX  ?= 1000
#NZ  ?= 500

#NX  ?= 1400
#NZ  ?= 700

#NX  ?= 2000
#NZ  ?= 1000

#NX  ?= 3000
#NZ  ?= 1500

#NX  ?= 4000
#NZ  ?= 2000

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

WORK_DIR  ?= ./jaiwork
DEBUG_DIR := ${WORK_DIR}
OUT_FILE  := ${WORK_DIR}/output.nc

JULIA := julia --project=. --check-bounds=no
#JULIA := julia --project=.

SRCDIR :=../../src
JAIALLSRC := ${SRCDIR}/jai/fortran_openacc_omptarget_cuda_hip/miniWeather_accel.jl
JAIFORTRANSRC := ${SRCDIR}/jai/fortran/miniWeather_accel.jl
JAIFORTHIP1SRC := ${SRCDIR}/jai/fortran_hip_ver1/miniWeather_accel.jl
JAIFORTHIP2SRC := ${SRCDIR}/jai/fortran_hip_ver2/miniWeather_accel.jl
JAIFORTOMPSRC := ${SRCDIR}/jai/fortran_omptarget/miniWeather_accel.jl
JAIFORTACCSRC := ${SRCDIR}/jai/fortran_openacc/miniWeather_accel.jl
JULIASRC := ${SRCDIR}/julia/miniWeather_mpi.jl
FORTSRC := ${SRCDIR}/fortran/miniWeather_mpi.F90
FORTACCSRC := ${SRCDIR}/fortran/miniWeather_mpi_openacc.F90
FORTOMPSRC := ${SRCDIR}/fortran/miniWeather_mpi_openmp45.F90
MANUALJLSRC := ${SRCDIR}/manual/miniWeather_openacc.jl
MANUALFORTSRC := ${SRCDIR}/manual/openacc_driver_${NX}x${NZ}.F90

SIM_OPTS := -s ${SIM_TIME} -x ${NX} -z ${NZ} -f ${OUT_FREQ} \
		-d ${DATA_SPEC} -o ${OUT_FILE} -w ${WORK_DIR} \
		-b ${DEBUG_DIR}
ARGS := ${ARGS} ${SIM_OPTS}

FC ?= gfortran
MACROS := -D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}

F_FLAGS ?= -I${OLCF_PARALLEL_NETCDF_ROOT}/include -L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf ${MACROS}

jai_all:
	echo "*** JAI ALL ***"
	${JULIA} ${JAIALLSRC} ${ARGS}

jai_fortran:
	echo "*** JAI FORTRAN ***"
	${JULIA} ${JAIFORTRANSRC} ${ARGS}

jai_fhip1:
	echo "*** JAI FORTRAN HIP 1 ***"
	${JULIA} ${JAIFORTHIP1SRC} ${ARGS}

jai_fhip2:
	echo "*** JAI FORTRAN HIP 2 ***"
	${JULIA} ${JAIFORTHIP2SRC} ${ARGS}

julia:
	echo "*** JULIA ***"
	${JULIA} ${JULIASRC} ${ARGS}

fortran: ${FORTSRC}
	echo "*** FORTRAN ***"
	${FORTRAN} ${MACROS} -o miniweather_fort.exe $<
	./miniweather_fort.exe

fopenacc: ${FORTACCSRC}
	echo "*** FORTRAN OPENACC ***"
	${FOPENACC} ${MACROS} -o miniweather_fortacc.exe $<
	./miniweather_fortacc.exe

fomptarget: ${FORTOMPSRC}
	echo "*** FORTRAN OPENMP TARGET ***"
	${FOMPTARGET} ${MACROS} -o miniweather_fortomp.exe $<
	./miniweather_fortomp.exe

julia_manual: ${MANUALFORTSRC}
	echo "*** JULIA OPENACC MANUAL ***"
	${FOPENACC} -o ${WORK_DIR}/jopenacc.so $<
	${JULIA} ${MANUALJLSRC} ${SIM_OPTS}

clean:
	rm -rf *.so core *.mod .jaitmp jaiwork *.o *.s *.exe output.nc data_roofline results.csv
