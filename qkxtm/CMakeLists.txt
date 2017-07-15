
include_directories(.)
include_directories(../lib)
include_directories(../include)
include_directories(../tests)
include_directories(../lib/code_pieces)
include_directories(${QKXTM_QUDA_HOME}/lib)
include_directories(${QKXTM_QUDA_HOME}/include)
include_directories(${QKXTM_QUDA_HOME}/tests)
include_directories(${QKXTM_HDF5HOME}/include)
include_directories(${QKXTM_LIMEHOME}/include)
include_directories(SYSTEM ${QKXTM_GSLHOME}/include)
if(QKXTM_ARPACK)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/SRC/MPI)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/UTIL/MPI)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/SRC)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/UTIL)
endif(QKXTM_ARPACK)  

#build a common library for all test utilities
set(QKXTM_COMMON ${QKXTM_QUDA_HOME}/tests/gtest-all.cc ${QKXTM_QUDA_HOME}/tests/misc.cpp QKXTM_util.cpp)
cuda_add_library(qkxtm STATIC ${QKXTM_COMMON})
set(QKXTM_LIBS QKXTMquda qkxtm ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS} cuda)

if(QUDA_QIO)
  LIST(APPEND QKXTM_LIBS ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  LIST(APPEND QKXTM_LIBS ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  LIST(APPEND QKXTM_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  find_library(MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
  LIST(APPEND QKXTM_LIBS ${MAGMA})
endif()

if(QUDA_NUMA_NVML)
  LIST(APPEND QKXTM_LIBS ${NVML_LIBRARY})
endif()

if(QKXTM_ARPACK)
  LIST(APPEND QKXTM_LIBS ${QKXTM_PARPACK_LIB})
  LIST(APPEND QKXTM_LIBS ${QKXTM_ARPACK_LIB})
endif()

LIST(APPEND QKXTM_LIBS ${QKXTM_GSL_LIB})
LIST(APPEND QKXTM_LIBS ${QKXTM_HDF5_LIB})
LIST(APPEND QKXTM_LIBS ${QKXTM_LIME_LIB})

if(QKXTM_OPENBLAS)
  LIST(APPEND QKXTM_LIBS ${QKXTM_OPENBLAS_GOMP})
  LIST(APPEND QKXTM_LIBS ${QKXTM_OPENBLAS_LIB})
endif(QKXTM_OPENBLAS)

if(QKXTM_MKL)
  LIST(APPEND QKXTM_LIBS ${QKXTM_MKL_LIB_GFORTRAN})
  LIST(APPEND QKXTM_LIBS ${QKXTM_MKL_LIB_IOMP5})
  LIST(APPEND QKXTM_LIBS ${QKXTM_MKL_LIB_LP64})
  LIST(APPEND QKXTM_LIBS ${QKXTM_MKL_LIB_THREAD})
  LIST(APPEND QKXTM_LIBS ${QKXTM_MKL_LIB_CORE})
endif(QKXTM_MKL)

#define QKXTM executables

if(QKXTM_2FLAVMG)
  #CalcMG_2pt3pt_EvenOdd
  # This function will calculate hadronic correlation functions using a 
  # handful of randomly separated, gaussian smeared point sources, 
  #constructed with a smeared gauge field. Even/Odd preconditioning is 
  # employed, as well as QUDA's MG solver routines.
  cuda_add_executable(CalcMG_2pt3pt_EvenOdd CalcMG_2pt3pt_EvenOdd.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(CalcMG_2pt3pt_EvenOdd ${QKXTM_LIBS})
endif()  

#MG_Bench
# This function will invert a single flavour, Gaussian smeared source
# using the MG preconditioner.
cuda_add_executable(MG_Bench MG_Bench.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
target_link_libraries(MG_Bench ${QKXTM_LIBS})

if(QKXTM_ARPACK)

#CalcMG_Loops_w_oneD_TSM_wExact
# This function will calculate the All-to-All propagator with a covariant 
# derivate, and correlation functions thereof. It calculates the 
# eigenvectors of the full operator and the corresponding correlations 
#from that part. It then projects out that exact part and employs 
# undiluted stochastic sources, Even/Odd preconditioning, functionality 
# for using the Trunacted Solver Method, as well as QUDA's MG solver 
# routines for the remainder.
cuda_add_executable(CalcMG_Loops_w_oneD_TSM_wExact CalcMG_Loops_w_oneD_TSM_wExact.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
target_link_libraries(CalcMG_Loops_w_oneD_TSM_wExact ${QKXTM_LIBS})

#CalcLowModeProjection
# Ths function will calculate the low modes of the massles M^{dagger}M 
# dirac operator, and then sum the low modes to obtain the
# Pontryagin Index.
cuda_add_executable(CalcLowModeProjection CalcLowModeProjection.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
target_link_libraries(CalcLowModeProjection ${QKXTM_LIBS})

endif()
