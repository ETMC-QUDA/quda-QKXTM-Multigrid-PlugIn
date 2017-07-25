#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <QKXTM_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <qudaQKXTM.h>

//========================================================================//
//====== P A R A M E T E R   S E T T I N G S   A N D   C H E C K S =======//
//========================================================================//

//-----------------//
// QUDA Parameters //
//-----------------//
extern QudaDslashType dslash_type;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision  prec_sloppy;
extern QudaPrecision  prec_precondition;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass; // mass of Dirac operator
extern double kappa; // kappa of Dirac operator
extern double mu;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];
extern QudaMassNormalization normalization; // mass normalization of Dirac operators

extern int niter;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

//------------------//
// QKXTM Parameters //
//------------------//
extern char latfile_smeared[];
extern char verbosity_level[];
extern int traj;
extern bool isEven;
extern int Q_sq;
extern double kappa;
extern char prop_path[];
extern double csw;

//-C.K. ARPACK Parameters
extern int PolyDeg;
extern int nEv;
extern int nKv;
extern char *spectrumPart;
extern bool isACC;
extern double tolArpack;
extern int maxIterArpack;
extern int modeArpack;
extern char arpack_logfile[];
extern double amin;
extern double amax;
extern bool isEven;
extern bool isFullOp;


namespace quda {
  extern void setTransferGPU(bool);
}

void display_test_info() {
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  
  return ;
  
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
}

void setInvertParam(QudaInvertParam &inv_param) {

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1.0 + 3.0/anisotropy);
  }
  
  printfQuda("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);


  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = csw*inv_param.kappa;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;
  
  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = QUDA_NORMOP_SOLVE;

  if(isEven) {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    printfQuda("### Running for the Even-Even Operator\n");
  }
  else {
    printfQuda("### Running for the Odd-Odd Operator\n");
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD_ASYMMETRIC;
  }
  
  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine 
  // convergence
  inv_param.residual_type = 
    static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq; 
  
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = 1e-4;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;


  if(strcmp(verbosity_level,"verbose")==0) 
    inv_param.verbosity = QUDA_VERBOSE;
  else if(strcmp(verbosity_level,"summarize")==0) 
    inv_param.verbosity = QUDA_SUMMARIZE;
  else if(strcmp(verbosity_level,"silent")==0) 
    inv_param.verbosity = QUDA_SILENT;
  else{
    warningQuda("Unknown verbosity level %s. Proceeding with QUDA_VERBOSE verbosity level\n",verbosity_level);
    inv_param.verbosity = QUDA_VERBOSE;
  }
}


//=======================================================================//
//== C O N T A I N E R   A N D   Q U D A   I N I T I A L I S A T I O N  =//
//=======================================================================//

int main(int argc, char **argv)
{
  using namespace quda;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    printfQuda("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  //QKXTM: qkxtm specfic inputs
  //--------------------------------------------------------------------
  //-C.K. Pass ARPACK parameters to arpackInfo  
  qudaQKXTM_arpackInfo arpackInfo;
  arpackInfo.PolyDeg = PolyDeg;
  arpackInfo.nEv = nEv;
  arpackInfo.nKv = nKv;
  arpackInfo.isACC = isACC;
  arpackInfo.tolArpack = tolArpack;
  arpackInfo.modeArpack;
  arpackInfo.maxIterArpack = maxIterArpack;
  strcpy(arpackInfo.arpack_logfile,arpack_logfile);
  arpackInfo.amin = amin;
  arpackInfo.amax = amax;
  arpackInfo.isEven = isEven;
  arpackInfo.isFullOp = isFullOp;

  if(strcmp(spectrumPart,"SR")==0) arpackInfo.spectrumPart = SR;
  else if(strcmp(spectrumPart,"LR")==0) arpackInfo.spectrumPart = LR;
  else if(strcmp(spectrumPart,"SM")==0) arpackInfo.spectrumPart = SM;
  else if(strcmp(spectrumPart,"LM")==0) arpackInfo.spectrumPart = LM;
  else if(strcmp(spectrumPart,"SI")==0) arpackInfo.spectrumPart = SI;
  else if(strcmp(spectrumPart,"LI")==0) arpackInfo.spectrumPart = LI;
  else{
    printf("Error: Your spectrumPart option is suspicious\n");
    exit(-1);
  }

  //-C.K. General QKXTM information
  qudaQKXTMinfo info;
  info.lL[0] = xdim;
  info.lL[1] = ydim;
  info.lL[2] = zdim;
  info.lL[3] = tdim;
  info.Q_sq = Q_sq;
  info.isEven = isEven;

  // QUDA parameters begin here.
  //-----------------------------------------------------------------
  if ( dslash_type != QUDA_TWISTED_MASS_DSLASH && 
       dslash_type != QUDA_TWISTED_CLOVER_DSLASH ){
    printfQuda("This test is only for twisted mass or twisted clover operator\n");
    exit(-1);
  }
  
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  //-C.K. Operator parameters for the Full Operator deflation
  QudaInvertParam evInvParam = newQudaInvertParam();
  setInvertParam(evInvParam);

  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);
  size_t sSize = (evInvParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? 
    sizeof(double) : sizeof(float);

  void *gauge[4];

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  // load in the command line supplied gauge field
  readLimeGauge(gauge, latfile, &gauge_param, &evInvParam, 
		gridsize_from_cmdline);
  applyBoundaryCondition(gauge, V/2 ,&gauge_param);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);
  //Print remaining info to stdout
  init_qudaQKXTM(&info);
  printf_qudaQKXTM();

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  for(int i = 0 ; i < 4 ; i++){
    free(gauge[i]);
  } 

  printfQuda("Before clover term\n");
  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) 
    loadCloverQuda(NULL, NULL, &evInvParam);
  printfQuda("After clover term\n");
  
  //Launch calculation.
  calcLowModeProjection(&evInvParam, arpackInfo);
  
  freeGaugeQuda();
  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
