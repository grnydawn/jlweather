# jlweather

cmake_minimum_required(VERSION 3.10)

project(Jlweather VERSION 0.1)

enable_language(C CXX Fortran)

############################################################
## Set Parameters
############################################################

if ("${ACCOUNT}" STREQUAL "")
  SET(ACCOUNT cli133)
endif()


if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  # ORG SET(SIM_TIME 400)
  SET(SIM_TIME 100)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()
SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")
SET(TEST_DEFS "-D_NX=100 -D_NZ=50 -D_SIM_TIME=400 -D_OUT_FREQ=400 -D_DATA_SPEC=DATA_SPEC_THERMAL")


set(MINIWEATHER_SOURCE_DIR  ${CMAKE_SOURCE_DIR}/../../src)
set(MINIWEATHER_RUN_DIR  ${CMAKE_SOURCE_DIR}/../../run)
set(FORTRAN_SOURCE_DIR  ${MINIWEATHER_SOURCE_DIR}/fortran)
set(JULIA_SOURCE_DIR  ${MINIWEATHER_SOURCE_DIR}/julia)
set(JAI_SOURCE_DIR  ${MINIWEATHER_SOURCE_DIR}/jai)
set(JAI_INC_SOURCE_DIR  ${MINIWEATHER_SOURCE_DIR}/jai_incremental)

execute_process(COMMAND julia --version OUTPUT_VARIABLE JULIA_VERSION)
IF ("${JULIA_VERSION}" STREQUAL "")
	MESSAGE(FATAL_ERROR "ERROR: julia is not available.")
ENDIF()

execute_process(COMMAND pnetcdf_version OUTPUT_VARIABLE PNETCDF_VERSION)
IF ("${PNETCDF_VERSION}" STREQUAL "")
	MESSAGE(FATAL_ERROR "ERROR: pnetcdf is not available.")
ENDIF()

if ("${MACHINE_NAME}" STREQUAL "")
	execute_process(COMMAND hostname OUTPUT_VARIABLE HOST_NAME_RAW)

	IF ("${HOST_NAME_RAW}" STREQUAL "")
		MESSAGE(FATAL_ERROR "ERROR: MACHINE_NAME is not available.")

	ELSEIF (${HOST_NAME_RAW} MATCHES crusher)
		set(MACHINE_NAME "crusher")

	ELSEIF (UNIX)
		set(MACHINE_NAME "unix")

	ELSE ()
		MESSAGE(FATAL_ERROR "ERROR: not supported machine.")

	ENDIF()

ENDIF()

if ("${MACHINE_NAME}" STREQUAL "crusher")
	set(JULIA_PROJECT_DIR "${MINIWEATHER_RUN_DIR}/crusher")
	SET(WORK_DIR "/gpfs/alpine/${ACCOUNT}/proj-shared/$ENV{USER}/jlworkdir")

elseif ("${MACHINE_NAME}" STREQUAL "unix")
	#set(JULIA_PROJECT_DIR "${MINIWEATHER_RUN_DIR}/unix")
	set(JULIA_PROJECT_DIR "${MINIWEATHER_RUN_DIR}/crusher")
	SET(WORK_DIR "$ENV{HOME}/temp/jltemp")

endif()

SET(NUM_DATA_SPEC_THERMAL 2)
SET(JULIA_TEST_DEFS -x 100 -z 50 -s 400 -f 400 -d ${NUM_DATA_SPEC_THERMAL} -o ${WORK_DIR}/output.nc -w ${WORK_DIR} -b ${WORK_DIR})

#MESSAGE(WARNING "DEBUG: ${CMAKE_Fortran_COMPILER}")
#MESSAGE(WARNING "DEBUG: ${CMAKE_Fortran_COMPILER_ID}")
#MESSAGE(WARNING "DEBUG: ${MACHINE_NAME}")
#MESSAGE(WARNING "DEBUG: ${JULIA_PROJECT_DIR}")

enable_testing()

add_executable(mpi ${FORTRAN_SOURCE_DIR}/miniWeather_mpi.F90)
set_target_properties(mpi PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

add_executable(mpi_test ${FORTRAN_SOURCE_DIR}/miniWeather_mpi.F90)
set_target_properties(mpi_test PROPERTIES COMPILE_FLAGS "${TEST_DEFS}")

add_test(NAME Fortran_TEST COMMAND ${CMAKE_SOURCE_DIR}/check_output.sh ./mpi_test 1e-13 4.5e-5 )
add_test(NAME Julia_TEST COMMAND julia --project=${JULIA_PROJECT_DIR} ${JULIA_SOURCE_DIR}/miniWeather_mpi.jl ${JULIA_TEST_DEFS})
add_test(NAME JAI_Fortran_TEST COMMAND julia --project=${JULIA_PROJECT_DIR} ${JAI_SOURCE_DIR}/miniWeather_accel.jl ${JULIA_TEST_DEFS})
add_test(NAME JAI_HIP1_TEST COMMAND julia --project=${JULIA_PROJECT_DIR} ${JAI_INC_SOURCE_DIR}/miniWeather_accel.jl ${JULIA_TEST_DEFS})
