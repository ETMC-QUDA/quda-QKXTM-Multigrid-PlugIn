# all files for quda -- needs some cleanup
set (QUDA_OBJS 
  ${QUDA_HOME}/lib/dirac_coarse.cpp ${QUDA_HOME}/lib/dslash_coarse.cu ${QUDA_HOME}/lib/coarse_op.cu ${QUDA_HOME}/lib/coarsecoarse_op.cu
  ${QUDA_HOME}/lib/multigrid.cpp ${QUDA_HOME}/lib/transfer.cpp ${QUDA_HOME}/lib/transfer_util.cu ${QUDA_HOME}/lib/inv_bicgstab_quda.cpp
  ${QUDA_HOME}/lib/inv_bicgstabl_quda.cpp
  ${QUDA_HOME}/lib/prolongator.cu ${QUDA_HOME}/lib/restrictor.cu ${QUDA_HOME}/lib/gauge_phase.cu ${QUDA_HOME}/lib/timer.cpp ${QUDA_HOME}/lib/malloc.cpp
  ${QUDA_HOME}/lib/solver.cpp ${QUDA_HOME}/lib/inv_bicgstab_quda.cpp ${QUDA_HOME}/lib/inv_cg_quda.cpp
  ${QUDA_HOME}/lib/inv_multi_cg_quda.cpp ${QUDA_HOME}/lib/inv_eigcg_quda.cpp ${QUDA_HOME}/lib/gauge_ape.cu
  ${QUDA_HOME}/lib/gauge_stout.cu ${QUDA_HOME}/lib/gauge_plaq.cu
  ${QUDA_HOME}/lib/inv_gcr_quda.cpp ${QUDA_HOME}/lib/inv_mr_quda.cpp ${QUDA_HOME}/lib/inv_sd_quda.cpp ${QUDA_HOME}/lib/inv_xsd_quda.cpp
  ${QUDA_HOME}/lib/inv_pcg_quda.cpp ${QUDA_HOME}/lib/inv_mre.cpp ${QUDA_HOME}/lib/util_quda.cpp
  ${QUDA_HOME}/lib/color_spinor_field.cpp ${QUDA_HOME}/lib/color_spinor_util.cu ${QUDA_HOME}/lib/color_spinor_pack.cu
  ${QUDA_HOME}/lib/cpu_color_spinor_field.cpp ${QUDA_HOME}/lib/cuda_color_spinor_field.cu ${QUDA_HOME}/lib/dirac.cpp
  ${QUDA_HOME}/lib/clover_field.cpp ${QUDA_HOME}/lib/covd.cpp ${QUDA_HOME}/lib/lattice_field.cpp ${QUDA_HOME}/lib/gauge_field.cpp
  ${QUDA_HOME}/lib/cpu_gauge_field.cpp ${QUDA_HOME}/lib/cuda_gauge_field.cu ${QUDA_HOME}/lib/extract_gauge_ghost.cu
  ${QUDA_HOME}/lib/extract_gauge_ghost_mg.cu ${QUDA_HOME}/lib/max_gauge.cu ${QUDA_HOME}/lib/gauge_update_quda.cu
  ${QUDA_HOME}/lib/dirac_clover.cpp ${QUDA_HOME}/lib/dirac_wilson.cpp ${QUDA_HOME}/lib/dirac_staggered.cpp
  ${QUDA_HOME}/lib/dirac_improved_staggered.cpp ${QUDA_HOME}/lib/dirac_domain_wall.cpp
  ${QUDA_HOME}/lib/dirac_domain_wall_4d.cpp ${QUDA_HOME}/lib/dirac_mobius.cpp ${QUDA_HOME}/lib/dirac_twisted_clover.cpp
  ${QUDA_HOME}/lib/dirac_twisted_mass.cpp ${QUDA_HOME}/lib/tune.cpp ${QUDA_HOME}/lib/fat_force_quda.cpp
  ${QUDA_HOME}/lib/llfat_quda.cu ${QUDA_HOME}/lib/gauge_force.cu
  ${QUDA_HOME}/lib/field_strength_tensor.cu ${QUDA_HOME}/lib/clover_quda.cu ${QUDA_HOME}/lib/dslash_quda.cu ${QUDA_HOME}/lib/covDev.cu
  ${QUDA_HOME}/lib/dslash_wilson.cu ${QUDA_HOME}/lib/dslash_clover.cu ${QUDA_HOME}/lib/dslash_clover_asym.cu
  ${QUDA_HOME}/lib/dslash_twisted_mass.cu ${QUDA_HOME}/lib/dslash_ndeg_twisted_mass.cu
  ${QUDA_HOME}/lib/dslash_twisted_clover.cu ${QUDA_HOME}/lib/dslash_domain_wall.cu
  ${QUDA_HOME}/lib/dslash_domain_wall_4d.cu ${QUDA_HOME}/lib/dslash_mobius.cu ${QUDA_HOME}/lib/dslash_staggered.cu
  ${QUDA_HOME}/lib/dslash_improved_staggered.cu ${QUDA_HOME}/lib/dslash_pack.cu ${QUDA_HOME}/lib/blas_quda.cu
  ${QUDA_HOME}/lib/multi_blas_quda.cu ${QUDA_HOME}/lib/multi_reduce_quda.cu
  ${QUDA_HOME}/lib/copy_quda.cu ${QUDA_HOME}/lib/reduce_quda.cu ${QUDA_HOME}/lib/face_buffer.cpp ${QUDA_HOME}/lib/face_gauge.cpp
  ${QUDA_HOME}/lib/comm_common.cpp ${QUDA_HOME}/lib/${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  ${QUDA_HOME}/lib/clover_deriv_quda.cu ${QUDA_HOME}/lib/clover_invert.cu ${QUDA_HOME}/lib/copy_gauge_extended.cu
  ${QUDA_HOME}/lib/extract_gauge_ghost_extended.cu ${QUDA_HOME}/lib/copy_color_spinor.cu
  ${QUDA_HOME}/lib/copy_color_spinor_dd.cu ${QUDA_HOME}/lib/copy_color_spinor_ds.cu
  ${QUDA_HOME}/lib/copy_color_spinor_dh.cu ${QUDA_HOME}/lib/copy_color_spinor_ss.cu
  ${QUDA_HOME}/lib/copy_color_spinor_sd.cu ${QUDA_HOME}/lib/copy_color_spinor_sh.cu
  ${QUDA_HOME}/lib/copy_color_spinor_hd.cu ${QUDA_HOME}/lib/copy_color_spinor_hs.cu
  ${QUDA_HOME}/lib/copy_color_spinor_hh.cu ${QUDA_HOME}/lib/copy_color_spinor_mg_dd.cu
  ${QUDA_HOME}/lib/copy_color_spinor_mg_ds.cu ${QUDA_HOME}/lib/copy_color_spinor_mg_sd.cu
  ${QUDA_HOME}/lib/copy_color_spinor_mg_ss.cu ${QUDA_HOME}/lib/copy_gauge_double.cu ${QUDA_HOME}/lib/copy_gauge_single.cu
  ${QUDA_HOME}/lib/copy_gauge_half.cu ${QUDA_HOME}/lib/copy_gauge.cu ${QUDA_HOME}/lib/copy_gauge_mg.cu ${QUDA_HOME}/lib/copy_clover.cu
  ${QUDA_HOME}/lib/staggered_oprod.cu ${QUDA_HOME}/lib/clover_trace_quda.cu ${QUDA_HOME}/lib/ks_force_quda.cu ${QUDA_HOME}/lib/staggered_oprod_new.cu
  ${QUDA_HOME}/lib/hisq_paths_force_quda.cu ${QUDA_HOME}/lib/fermion_force_quda.cu
  ${QUDA_HOME}/lib/unitarize_force_quda.cu ${QUDA_HOME}/lib/unitarize_links_quda.cu ${QUDA_HOME}/lib/milc_interface.cpp
  ${QUDA_HOME}/lib/extended_color_spinor_utilities.cu ${QUDA_HOME}/lib/eig_lanczos_quda.cpp
  ${QUDA_HOME}/lib/ritz_quda.cpp ${QUDA_HOME}/lib/eig_solver.cpp ${QUDA_HOME}/lib/blas_magma.cu ${QUDA_HOME}/lib/misc_helpers.cu
  ${QUDA_HOME}/lib/inv_mpcg_quda.cpp ${QUDA_HOME}/lib/inv_mpbicgstab_quda.cpp ${QUDA_HOME}/lib/inv_gmresdr_quda.cpp
  ${QUDA_HOME}/lib/pgauge_exchange.cu ${QUDA_HOME}/lib/pgauge_init.cu ${QUDA_HOME}/lib/pgauge_heatbath.cu ${QUDA_HOME}/lib/random.cu
  ${QUDA_HOME}/lib/gauge_fix_ovr_extra.cu ${QUDA_HOME}/lib/gauge_fix_fft.cu ${QUDA_HOME}/lib/gauge_fix_ovr.cu
  ${QUDA_HOME}/lib/pgauge_det_trace.cu ${QUDA_HOME}/lib/clover_outer_product.cu
  ${QUDA_HOME}/lib/clover_sigma_outer_product.cu ${QUDA_HOME}/lib/momentum.cu ${QUDA_HOME}/lib/qcharge_quda.cu
  ${QUDA_HOME}/lib/quda_memcpy.cpp ${QUDA_HOME}/lib/version.cpp
  qudaQKXTM_Kepler_kernels.cu qudaQKXTM_interface.cpp )

## split source into cu and cpp files
FOREACH(item ${QUDA_OBJS})
  STRING(REGEX MATCH ".+\\.cu$" item_match ${item})
  IF(item_match)
    LIST(APPEND QUDA_CU_OBJS ${item})
  ENDIF(item_match)
ENDFOREACH(item ${QUDA_OBJS})

LIST(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  LIST(APPEND QUDA_OBJS quda_fortran.F90)
  set_source_files_properties(quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now
# QUDA_OBJS all c, cpp, fortran sources

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(GITVERSION)
  find_file(QUDA_GITDIR NAMES .git PATHS ${CMAKE_SOURCE_DIR} NO_DEFAULT_PATH)
  set_property(SOURCE version.cpp PROPERTY OBJECT_DEPENDS  ${QUDA_GITDIR}/logs/HEAD)
endif()
mark_as_advanced(QUDA_GITDIR)

# include_directories
include_directories(${QUDA_HOME}/lib/dslash_core)
include_directories(${QUDA_HOME}/lib/)
#QKXTM: Begin DMH
include_directories(code_pieces_Kepler)
include_directories(SYSTEM ${QUDA_QKXTM_GSLHOME}/include)
include_directories(SYSTEM ${QUDA_QKXTM_HDF5HOME}/src)
include_directories(SYSTEM ${QUDA_QKXTM_LIMEHOME}/include)
if(QUDA_QKXTM_ARPACK)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/PARPACK/SRC/MPI)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/PARPACK/UTIL/MPI)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/SRC)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/UTIL)
endif(QUDA_QKXTM_ARPACK)
#QKXTM: End DMH

# generate a cmake object library for all cpp files first
add_library(QKXTMquda_cpp OBJECT ${QUDA_OBJS})

# add some deifnitions that cause issues with cmake 3.7 and nvcc only to cpp files
target_compile_definitions(QKXTMquda_cpp PUBLIC -DQUDA_HASH="${HASH}")
IF(GITVERSION)
  target_compile_definitions(QKXTMquda_cpp PUBLIC -DGITVERSION="${GITVERSION}")
ENDIF()

# make one library
if(QUDA_BUILD_SHAREDLIB)
    set_target_properties(QKXTMquda_cpp PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    cuda_add_library(QKXTMquda SHARED $<TARGET_OBJECTS:QKXTMquda_cpp> ${QUDA_CU_OBJS} )
else()
  cuda_add_library(QKXTMquda STATIC $<TARGET_OBJECTS:QKXTMquda_cpp> ${QUDA_CU_OBJS})
endif()
target_include_directories(QKXTMquda PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

#DMH
LIST(APPEND QKXTMquda ${QKXTM_GSL_LIB})
LIST(APPEND QKXTMquda ${QKXTM_HDF5_LIB})
LIST(APPEND QKXTMquda ${QKXTM_LIME_LIB})
if(QUDA_QKXTM_ARPACK)
  LIST(APPEND QKXTMquda ${QKXTM_PARPACK_LIB})
  LIST(APPEND QKXTMquda ${QKXTM_ARPACK_LIB})
endif(QUDA_QKXTM_ARPACK)
#DMH

# add dependencies for linking
# if CUDA_CUDA_LIBRARY (driver api) is not found look for stubs
FIND_LIBRARY(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)

target_link_libraries(QKXTMquda ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS} ${CUDA_CUDA_LIBRARY})

if(QUDA_QIO)
  target_link_libraries(QKXTMquda ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  target_link_libraries(QKXTMquda ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  target_link_libraries(QKXTMquda ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  target_link_libraries(QKXTMquda ${MAGMA})
endif()

if(QUDA_ARPACK)
  target_link_libraries(QKXTMquda ${PARPACK_OR_ARPACK})
endif()

if(QUDA_NUMA_NVML)
target_link_libraries(QKXTMquda ${NVML_LIBRARY})
endif()


# malloc.cpp uses both the driver and runtime api
# So we need to find the CUDA_CUDA_LIBRARY (driver api) or the stub version
find_library(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)
target_link_libraries(QKXTMquda ${CUDA_CUDA_LIBRARY})

# until we define an install step copy the include directory to the build directory
ADD_CUSTOM_COMMAND(TARGET QKXTMquda POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
# some hackery to prevent having old shared / static builds of quda messing with the current build
ADD_CUSTOM_COMMAND(TARGET QKXTMquda PRE_LINK COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/libQKXTMquda.a ${CMAKE_CURRENT_BINARY_DIR}/libQKXTMquda.so)

install(TARGETS QKXTMquda EXPORT QKXTMqudaTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
include(CMakePackageConfigHelpers)
write_basic_package_version_file( 
  "${CMAKE_CURRENT_BINARY_DIR}/qudaConfigVersion.cmake" 
  VERSION ${QUDA_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(EXPORT QKXTMqudaTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/QKXTMqudaTargets.cmake" NAMESPACE quda::)
set(ConfigPackageLocation lib/cmake/quda/)
install(EXPORT QKXTMqudaTargets NAMESPACE quda:: DESTINATION ${ConfigPackageLocation})

add_custom_target(gen ${PYTHON_EXECUTABLE} ${QUDA_HOME}/lib/generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} ${QUDA_HOME}/lib/generate/wrap.py -g -o ${QUDA_HOME}/lib/nvtx_pmpi.c ${QUDA_HOME}/lib/generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )
