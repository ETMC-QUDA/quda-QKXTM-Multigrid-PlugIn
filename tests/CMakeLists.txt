
include_directories(.)
include_directories(../lib)
include_directories(../include)
include_directories(../tests)
include_directories(../lib/code_pieces_Kepler)
include_directories(${QKXTM_QUDA_HOME}/lib)
include_directories(${QKXTM_QUDA_HOME}/include)
include_directories(${QKXTM_QUDA_HOME}/tests)
include_directories(${QKXTM_HDF5HOME}/include)
include_directories(${QKXTM_LIMEHOME}/include)
include_directories(SYSTEM ${QKXTM_GSLHOME}/include)
#if(QKXTM_ARPACK)
include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/SRC/MPI)
include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/UTIL/MPI)
include_directories(SYSTEM ${QKXTM_ARPACKHOME}/SRC)
include_directories(SYSTEM ${QKXTM_ARPACKHOME}/UTIL)
#endif(QKXTM_ARPACK)  

#build a common library for all test utilities
set(QUDA_TEST_COMMON ${QKXTM_QUDA_HOME}/tests/gtest-all.cc ${QKXTM_QUDA_HOME}/tests/test_util.cpp ${QKXTM_QUDA_HOME}/tests/misc.cpp)
cuda_add_library(quda_test STATIC ${QUDA_TEST_COMMON})
set(TEST_LIBS QKXTMquda quda_test)


MACRO (QUDA_CHECKBUILDTEST mytarget qudabuildtests)
  IF(NOT ${qudabuildtests})
    set_property(TARGET ${mytarget} PROPERTY EXCLUDE_FROM_ALL 1)
  ENDIF()
ENDMACRO()

if(QUDA_QIO)
  LIST(APPEND TEST_LIBS ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  LIST(APPEND TEST_LIBS ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  LIST(APPEND TEST_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  find_library(MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
  LIST(APPEND TEST_LIBS ${MAGMA})
endif()

if(QUDA_NUMA_NVML)
  LIST(APPEND TEST_LIBS ${NVML_LIBRARY})
endif()

if(QKXTM_ARPACK)
  LIST(APPEND TEST_LIBS ${QKXTM_PARPACK_LIB})
  LIST(APPEND TEST_LIBS ${QKXTM_ARPACK_LIB})
endif(QKXTM_ARPACK)
LIST(APPEND TEST_LIBS ${QKXTM_GSL_LIB})
LIST(APPEND TEST_LIBS ${QKXTM_HDF5_LIB})
LIST(APPEND TEST_LIBS ${QKXTM_LIME_LIB})
if(QKXTM_OPENBLAS)
  LIST(APPEND TEST_LIBS ${QKXTM_OPENBLAS_LIB})
endif()

#define tests

if(QUDA_DIRAC_WILSON OR QUDA_DIRAC_CLOVER OR QUDA_DIRAC_TWISTED_MASS OR QUDA_DIRAC_TWISTED_CLOVER OR QUDA_DIRAC_DOMAIN_WALL)
  cuda_add_executable(dslash_test ${QKXTM_QUDA_HOME}/tests/dslash_test.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(dslash_test ${TEST_LIBS} )
  QUDA_CHECKBUILDTEST(dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(invert_test ${QKXTM_QUDA_HOME}/tests/invert_test.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(invert_test QUDA_BUILD_ALL_TESTS)


  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(invertmsrc_test ${QKXTM_QUDA_HOME}/tests/invertmsrc_test.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
    target_link_libraries(invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()

endif()

#cuda_add_executable(deflation_test ${QKXTM_QUDA_HOME}/tests/deflation_test.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
#target_link_libraries(deflation_test ${TEST_LIBS})
#QUDA_CHECKBUILDTEST(deflation_test QUDA_BUILD_ALL_TESTS)

if(QUDA_DIRAC_STAGGERED)
  cuda_add_executable(staggered_dslash_test ${QKXTM_QUDA_HOME}/tests/staggered_dslash_test.cpp ${QKXTM_QUDA_HOME}/tests/staggered_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(staggered_dslash_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_dslash_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(staggered_invert_test ${QKXTM_QUDA_HOME}/tests/staggered_invert_test.cpp ${QKXTM_QUDA_HOME}/tests/staggered_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(staggered_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(staggered_invert_test QUDA_BUILD_ALL_TESTS)

  if(QUDA_BLOCKSOLVER)
    cuda_add_executable(staggered_invertmsrc_test staggered_invertmsrc_test.cpp  ${QKXTM_QUDA_HOME}/tests/staggered_dslash_reference.cpp  ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
    target_link_libraries(staggered_invertmsrc_test ${TEST_LIBS})
    QUDA_CHECKBUILDTEST(staggered_invertmsrc_test QUDA_BUILD_ALL_TESTS)
  endif()
endif()

if(QUDA_MULTIGRID)
  cuda_add_executable(multigrid_invert_test ${QKXTM_QUDA_HOME}/tests/multigrid_invert_test.cpp ${QKXTM_QUDA_HOME}/tests/wilson_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/clover_reference.cpp ${QKXTM_QUDA_HOME}/tests/domain_wall_dslash_reference.cpp ${QKXTM_QUDA_HOME}/tests/blas_reference.cpp)
  target_link_libraries(multigrid_invert_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(multigrid_invert_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(multigrid_benchmark_test ${QKXTM_QUDA_HOME}/tests/multigrid_benchmark_test.cu)
  target_link_libraries(multigrid_benchmark_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(multigrid_benchmark_test QUDA_BUILD_ALL_TESTS)
endif()

cuda_add_executable(su3_test ${QKXTM_QUDA_HOME}/tests/su3_test.cpp)
target_link_libraries(su3_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(su3_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(pack_test ${QKXTM_QUDA_HOME}/tests/pack_test.cpp)
target_link_libraries(pack_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(pack_test QUDA_BUILD_ALL_TESTS)

cuda_add_executable(blas_test ${QKXTM_QUDA_HOME}/tests/blas_test.cu)
target_link_libraries(blas_test ${TEST_LIBS})
QUDA_CHECKBUILDTEST(blas_test QUDA_BUILD_ALL_TESTS)

if(QUDA_LINK_ASQTAD OR QUDA_LINK_HISQ)
  cuda_add_executable(llfat_test ${QKXTM_QUDA_HOME}/tests/llfat_test.cpp ${QKXTM_QUDA_HOME}/tests/llfat_reference.cpp)
  target_link_libraries(llfat_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(llfat_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_LINK_HISQ)
  cuda_add_executable(unitarize_link_test ${QKXTM_QUDA_HOME}/tests/unitarize_link_test.cpp)
  target_link_libraries(unitarize_link_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(unitarize_link_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_GAUGE)
  cuda_add_executable(gauge_force_test ${QKXTM_QUDA_HOME}/tests/gauge_force_test.cpp ${QKXTM_QUDA_HOME}/tests/gauge_force_reference.cpp)
  target_link_libraries(gauge_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_force_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_ASQTAD)
  cuda_add_executable(fermion_force_test ${QKXTM_QUDA_HOME}/tests/fermion_force_test.cpp ${QKXTM_QUDA_HOME}/tests/fermion_force_reference.cpp)
  target_link_libraries(fermion_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(fermion_force_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_GAUGE_ALG)
  cuda_add_executable(gauge_alg_test ${QKXTM_QUDA_HOME}/tests/gauge_alg_test.cpp)
  target_link_libraries(gauge_alg_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(gauge_alg_test QUDA_BUILD_ALL_TESTS)
endif()

if(QUDA_FORCE_HISQ)
  cuda_add_executable(hisq_paths_force_test ${QKXTM_QUDA_HOME}/tests/hisq_paths_force_test.cpp ${QKXTM_QUDA_HOME}/tests/hisq_force_reference.cpp ${QKXTM_QUDA_HOME}/tests/hisq_force_reference2.cpp ${QKXTM_QUDA_HOME}/tests/fermion_force_reference.cpp   )
  target_link_libraries(hisq_paths_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_paths_force_test QUDA_BUILD_ALL_TESTS)

  cuda_add_executable(hisq_unitarize_force_test ${QKXTM_QUDA_HOME}/tests/hisq_unitarize_force_test.cpp ${QKXTM_QUDA_HOME}/tests/hisq_force_reference.cpp )
  target_link_libraries(hisq_unitarize_force_test ${TEST_LIBS})
  QUDA_CHECKBUILDTEST(hisq_unitarize_force_test QUDA_BUILD_ALL_TESTS)
endif()
