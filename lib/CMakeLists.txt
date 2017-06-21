# all files for quda -- needs some cleanup
set (QUDA_OBJS 
  ${QKXTM_QUDA_HOME}/lib/dirac_coarse.cpp
  ${QKXTM_QUDA_HOME}/lib/dslash_coarse.cu
  ${QKXTM_QUDA_HOME}/lib/coarse_op.cu
  ${QKXTM_QUDA_HOME}/lib/coarsecoarse_op.cu
  ${QKXTM_QUDA_HOME}/lib/coarse_op_preconditioned.cu
  ${QKXTM_QUDA_HOME}/lib/multigrid.cpp
  ${QKXTM_QUDA_HOME}/lib/transfer.cpp
  ${QKXTM_QUDA_HOME}/lib/transfer_util.cu
  ${QKXTM_QUDA_HOME}/lib/inv_bicgstab_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_bicgstabl_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/prolongator.cu
  ${QKXTM_QUDA_HOME}/lib/restrictor.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_phase.cu
  ${QKXTM_QUDA_HOME}/lib/timer.cpp
  ${QKXTM_QUDA_HOME}/lib/malloc.cpp
  ${QKXTM_QUDA_HOME}/lib/solver.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_cg_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_multi_cg_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_eigcg_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/gauge_ape.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_stout.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_plaq.cu
  ${QKXTM_QUDA_HOME}/lib/inv_cg3_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_cg3ne_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_gcr_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_mr_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_sd_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_xsd_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_pcg_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_mre.cpp
  qudaQKXTM_interface.cpp 
  ${QKXTM_QUDA_HOME}/lib/util_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/color_spinor_field.cpp
  ${QKXTM_QUDA_HOME}/lib/color_spinor_util.cu
  ${QKXTM_QUDA_HOME}/lib/color_spinor_pack.cu
  ${QKXTM_QUDA_HOME}/lib/cpu_color_spinor_field.cpp
  ${QKXTM_QUDA_HOME}/lib/cuda_color_spinor_field.cu
  ${QKXTM_QUDA_HOME}/lib/dirac.cpp
  ${QKXTM_QUDA_HOME}/lib/clover_field.cpp
  ${QKXTM_QUDA_HOME}/lib/covd.cpp
  ${QKXTM_QUDA_HOME}/lib/lattice_field.cpp
  ${QKXTM_QUDA_HOME}/lib/gauge_field.cpp
  ${QKXTM_QUDA_HOME}/lib/cpu_gauge_field.cpp
  ${QKXTM_QUDA_HOME}/lib/cuda_gauge_field.cu
  ${QKXTM_QUDA_HOME}/lib/extract_gauge_ghost.cu
  ${QKXTM_QUDA_HOME}/lib/extract_gauge_ghost_mg.cu
  ${QKXTM_QUDA_HOME}/lib/max_gauge.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_update_quda.cu
  ${QKXTM_QUDA_HOME}/lib/dirac_clover.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_wilson.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_staggered.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_improved_staggered.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_domain_wall.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_domain_wall_4d.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_mobius.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_twisted_clover.cpp
  ${QKXTM_QUDA_HOME}/lib/dirac_twisted_mass.cpp
  ${QKXTM_QUDA_HOME}/lib/tune.cpp
  ${QKXTM_QUDA_HOME}/lib/fat_force_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/llfat_quda.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_force.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_random.cu
  ${QKXTM_QUDA_HOME}/lib/field_strength_tensor.cu
  ${QKXTM_QUDA_HOME}/lib/clover_quda.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_quda.cu
  ${QKXTM_QUDA_HOME}/lib/covDev.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_wilson.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_clover.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_clover_asym.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_twisted_mass.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_ndeg_twisted_mass.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_twisted_clover.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_domain_wall.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_domain_wall_4d.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_mobius.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_staggered.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_improved_staggered.cu
  ${QKXTM_QUDA_HOME}/lib/dslash_pack.cu
  ${QKXTM_QUDA_HOME}/lib/blas_quda.cu
  ${QKXTM_QUDA_HOME}/lib/multi_blas_quda.cu
  ${QKXTM_QUDA_HOME}/lib/copy_quda.cu
  ${QKXTM_QUDA_HOME}/lib/reduce_quda.cu
  ${QKXTM_QUDA_HOME}/lib/multi_reduce_quda.cu
  ${QKXTM_QUDA_HOME}/lib/face_buffer.cpp
  ${QKXTM_QUDA_HOME}/lib/face_gauge.cpp
  ${QKXTM_QUDA_HOME}/lib/comm_common.cpp
  ${COMM_OBJS} ${NUMA_AFFINITY_OBJS} ${QIO_UTIL}
  ${QKXTM_QUDA_HOME}/lib/clover_deriv_quda.cu
  ${QKXTM_QUDA_HOME}/lib/clover_invert.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge_extended.cu
  ${QKXTM_QUDA_HOME}/lib/extract_gauge_ghost_extended.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor.cu
  ${QKXTM_QUDA_HOME}/lib/spinor_gauss.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_dd.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_ds.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_dh.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_ss.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_sd.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_sh.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_hd.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_hs.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_hh.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_dd.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_ds.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_sd.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_ss.cu
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_hs.cu 
  ${QKXTM_QUDA_HOME}/lib/copy_color_spinor_mg_sh.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge_double.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge_single.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge_half.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge.cu
  ${QKXTM_QUDA_HOME}/lib/copy_gauge_mg.cu
  ${QKXTM_QUDA_HOME}/lib/copy_clover.cu
  ${QKXTM_QUDA_HOME}/lib/staggered_oprod.cu
  ${QKXTM_QUDA_HOME}/lib/staggered_oprod_new.cu
  ${QKXTM_QUDA_HOME}/lib/clover_trace_quda.cu
  ${QKXTM_QUDA_HOME}/lib/ks_force_quda.cu
  ${QKXTM_QUDA_HOME}/lib/hisq_paths_force_quda.cu
  ${QKXTM_QUDA_HOME}/lib/fermion_force_quda.cu
  ${QKXTM_QUDA_HOME}/lib/unitarize_force_quda.cu
  ${QKXTM_QUDA_HOME}/lib/unitarize_links_quda.cu
  ${QKXTM_QUDA_HOME}/lib/milc_interface.cpp
  ${QKXTM_QUDA_HOME}/lib/extended_color_spinor_utilities.cu
  ${QKXTM_QUDA_HOME}/lib/eig_lanczos_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/ritz_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/eig_solver.cpp
  ${QKXTM_QUDA_HOME}/lib/blas_cublas.cu 
  ${QKXTM_QUDA_HOME}/lib/blas_magma.cu
  ${QKXTM_QUDA_HOME}/lib/misc_helpers.cu
  ${QKXTM_QUDA_HOME}/lib/inv_mpcg_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_mpbicgstab_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/inv_gmresdr_quda.cpp
  ${QKXTM_QUDA_HOME}/lib/pgauge_exchange.cu
  ${QKXTM_QUDA_HOME}/lib/pgauge_init.cu
  ${QKXTM_QUDA_HOME}/lib/pgauge_heatbath.cu
  ${QKXTM_QUDA_HOME}/lib/random.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_fix_ovr_extra.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_fix_fft.cu
  ${QKXTM_QUDA_HOME}/lib/gauge_fix_ovr.cu
  ${QKXTM_QUDA_HOME}/lib/pgauge_det_trace.cu
  ${QKXTM_QUDA_HOME}/lib/clover_outer_product.cu
  ${QKXTM_QUDA_HOME}/lib/clover_sigma_outer_product.cu
  ${QKXTM_QUDA_HOME}/lib/momentum.cu
  ${QKXTM_QUDA_HOME}/lib/qcharge_quda.cu
  ${QKXTM_QUDA_HOME}/lib/quda_memcpy.cpp
  ${QKXTM_QUDA_HOME}/lib/quda_arpack_interface.cpp
  ${QKXTM_QUDA_HOME}/lib/deflation.cpp
  ${QKXTM_QUDA_HOME}/lib/version.cpp
  qudaQKXTM_kernels.cu 
  )

## split source into cu and cpp files
FOREACH(item ${QUDA_OBJS})
  STRING(REGEX MATCH ".+\\.cu$" item_match ${item})
  IF(item_match)
    LIST(APPEND QUDA_CU_OBJS ${item})
  ENDIF(item_match)
ENDFOREACH(item ${QUDA_OBJS})

LIST(REMOVE_ITEM QUDA_OBJS ${QUDA_CU_OBJS})

if(BUILD_FORTRAN_INTERFACE)
  LIST(APPEND QUDA_OBJS ${QKXTM_QUDA_HOME}/lib/quda_fortran.F90)
  set_source_files_properties(${QKXTM_QUDA_HOME}/lib/quda_fortran.F90 PROPERTIES OBJECT_OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/quda_fortran.mod)
endif()

# QUDA_CU_OBJS shoudl contain all cuda files now
# QUDA_OBJS all c, cpp, fortran sources

# if we have a git version make version.cpp depend on git head so that it is rebuild if the git sha changed
if(GITVERSION)
  find_file(QUDA_GITDIR NAMES ${QKXTM_QUDA_HOME}/.git PATHS ${QKXTM_QUDA_HOME} NO_DEFAULT_PATH)
  set_property(SOURCE ${QKXTM_QUDA_HOME}/lib/version.cpp PROPERTY OBJECT_DEPENDS  ${QUDA_GITDIR}/logs/HEAD)
endif()
mark_as_advanced(QUDA_GITDIR)

# include_directories
include_directories(${QKXTM_QUDA_HOME}/lib/dslash_core)
include_directories(${QKXTM_QUDA_HOME}/lib/)
#QKXTM: Begin DMH
include_directories(code_pieces)
include_directories(SYSTEM ${QKXTM_GSLHOME}/include)
include_directories(SYSTEM ${QKXTM_HDF5HOME}/src)
include_directories(SYSTEM ${QKXTM_LIMEHOME}/include)
if(QKXTM_ARPACK)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/SRC/MPI)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/PARPACK/UTIL/MPI)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/SRC)
  include_directories(SYSTEM ${QKXTM_ARPACKHOME}/UTIL)
endif(QKXTM_ARPACK)
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
if(QKXTM_ARPACK)
  LIST(APPEND QKXTMquda ${QKXTM_PARPACK_LIB})
  LIST(APPEND QKXTMquda ${QKXTM_ARPACK_LIB})
endif(QKXTM_ARPACK)
#DMH

# add dependencies for linking
# if CUDA_CUDA_LIBRARY (driver api) is not found look for stubs
FIND_LIBRARY(CUDA_CUDA_LIBRARY cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib/ ${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs)

target_link_libraries(QKXTMquda ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS} ${CUDA_CUDA_LIBRARY})

if(QUDA_QIO)
  target_link_libraries(QKXTMquda ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QDPJIT)
  target_link_libraries(quda ${QDP_LIB} ${QIO_LIB} ${LIME_LIB} ${QMP_LIB} ${MPI_CXX_LIBRARIES})
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


add_custom_target(gen ${PYTHON_EXECUTABLE} ${QKXTM_QUDA_HOME}/lib/generate/gen.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating dslash_core files" )

add_custom_target(mpi_nvtx ${PYTHON_EXECUTABLE} ${QKXTM_QUDA_HOME}/lib/generate/wrap.py -g -o ${QKXTM_QUDA_HOME}/lib/nvtx_pmpi.c ${QKXTM_QUDA_HOME}/lib/generate/nvtx.w
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating mpi_nvtx wrapper" )
