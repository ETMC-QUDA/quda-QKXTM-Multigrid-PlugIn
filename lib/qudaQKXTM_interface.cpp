
#include <interface_quda.cpp>
#include <qudaQKXTM_Field.cpp>
#include <qudaQKXTM_Gauge.cpp>
#include <qudaQKXTM_Vector.cpp>
#include <qudaQKXTM_Propagator.cpp>
#include <qudaQKXTM_Contraction.cpp>
#ifdef HAVE_ARPACK
#include <qudaQKXTM_Deflation.cpp>
#include <qudaQKXTM_Loops.cpp>
#endif

#include <qudaQKXTM_utils.cpp>
#include <QKXTM_util.h>
///////////////////////
// QKXTM MG Routines //
///////////////////////


void MG_bench(void **gaugeSmeared, void **gauge, 
	      QudaGaugeParam *gauge_param, 
	      QudaInvertParam *param,
	      qudaQKXTMinfo info){
  
  bool flag_eo;
  double t1,t2,t3,t4,t5,t6;
  
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_DIRECT_SOLVE) 
    errorQuda("This function works only with Direct, unpreconditioned "
	      "solves (This will get better...)");

  if(param->inv_type != QUDA_GCR_INVERTER) 
    errorQuda("This function works only with GCR method");

  bool pc_solution = false;
  bool pc_solve = false;
  bool mat_solution = 
    (param->solution_type == QUDA_MAT_SOLUTION) || 
    (param->solution_type == QUDA_MATPC_SOLUTION);
  bool direct_solve = true;

  void *input_vector = malloc(GK_localL[0]*
			      GK_localL[1]*
			      GK_localL[2]*
			      GK_localL[3]*spinorSiteSize*sizeof(double));

  void *output_vector = malloc(GK_localL[0]*
			       GK_localL[1]*
			       GK_localL[2]*
			       GK_localL[3]*spinorSiteSize*sizeof(double));
  
  
  QKXTM_Gauge<double> *K_gaugeSmeared = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);

  QKXTM_Vector<double> *K_vector = 
    new QKXTM_Vector<double>(BOTH,VECTOR);

  QKXTM_Vector<double> *K_guess = 
    new QKXTM_Vector<double>(BOTH,VECTOR);

  printfQuda("Memory allocation was successfull\n");

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("This function works only with ukqcd gamma basis\n");
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("This function works only with colors inside the spins\n");


  if (!initialized)
    errorQuda("QUDA not initialized");

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
    printQudaInvertParam(param);
  
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  K_gaugeSmeared->packGauge(gaugeSmeared);
  K_gaugeSmeared->loadGauge();
  K_gaugeSmeared->calculatePlaq();

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? 
		       sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, 
			    param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);
  
  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  
  blas::zero(*x);
  blas::zero(*b);
  DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);

  int my_src[4];

  printfQuda("\nThe total number of source-positions is %d\n",0);

  //------------------------------------------------------------------------
  
  for(int isource = 0 ; isource < 1 ; isource++){
    t3 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       0, 0, 0, 0, 0);
    
    printfQuda("Forward Inversions:\n");
    t1 = MPI_Wtime();
    for(int isc = 0 ; isc < 12 ; isc++){

      t4 = MPI_Wtime();

      ///////////////////////////////
      // Forward prop for up quark //
      ///////////////////////////////

      memset(input_vector,0,
	     X[0]*X[1]*X[2]*X[3]*
	     spinorSiteSize*sizeof(double));
      //Ensure mu is positive:
      if(param->mu < 0) param->mu *= -1.0;

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = (0 - comm_coords(default_topo)[i] * X[i]);
      
      if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	  (my_src[1]>=0) && (my_src[1]<X[1]) && 
	  (my_src[2]>=0) && (my_src[2]<X[2]) && 
	  (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + 
	   my_src[3]*X[2]*X[1]*X[0]*24 + 
	   my_src[2]*X[1]*X[0]*24 + 
	   my_src[1]*X[0]*24 + 
	   my_src[0]*24 + 
	   isc*2 ) = 1.0;
      
      

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      K_guess->uploadToCuda(b,flag_eo);
      dirac.prepare(in,out,*x,*b,param->solution_type);

      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, 
				     mPre, profileInvert);
      
      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet
      
      printfQuda(" up - %02d: \n",isc);
      (*solve)(*out,*in);
      solverParam.updateInvertParam(*param);
      dirac.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      
      delete solve;

      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d, for source = %d finished in time %f sec\n",isc,0,t2-t4);

    }
    // Close loop over 12 spin-color    
  }

  free(input_vector);
  free(output_vector);
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete h_x;
  delete h_b;
  delete x;
  delete b;

  printfQuda("...Done\n");

  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

#ifdef QKXTM_2FLAVMG

void calcMG_threepTwop_EvenOdd(void **gauge_APE, void **gauge, 
			       QudaGaugeParam *gauge_param, 
			       QudaInvertParam *param, 
			       qudaQKXTMinfo info, 
			       char *filename_twop, 
			       char *filename_threep, 
			       WHICHPARTICLE NUCLEON){
  
  bool flag_eo;
  double t1,t2,t3,t4,t5,t6;
  double tx1,tx2,summ_tx12=0.; // needed to time gaussian smearing routines
  double tx3,tx4,summ_tx34=0.; // needed to time just the inversion time
  char fname[256];
  sprintf(fname, "calcMG_threepTwop_EvenOdd");

  //======================================================================//
  //= P A R A M E T E R   C H E C K S   A N D   I N I T I A L I S I O N ==//
  //======================================================================//
  
  if (!initialized)
    errorQuda("%s: QUDA not initialized", fname);

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
    printQudaInvertParam(param);
  
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if(param->solve_type != QUDA_DIRECT_PC_SOLVE) 
    errorQuda("%s: This function works only with Direct solve and even odd preconditioning", fname);
  
  if(param->inv_type != QUDA_GCR_INVERTER) 
    errorQuda("%s: This function works only with GCR method", fname);

  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("%s: This function works only with ukqcd gamma basis\n", fname);
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("%s: This function works only with colors inside the spins\n", fname);

  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD )
    flag_eo = false;

  int my_src[4];
  char filename_mesons[257];
  char filename_baryons[257];

  info.thrp_type[0] = "ultra_local";
  info.thrp_type[1] = "noether";
  info.thrp_type[2] = "oneD";

  info.thrp_proj_type[0] = "G4";
  info.thrp_proj_type[1] = "G5G123";
  info.thrp_proj_type[2] = "G5G1";
  info.thrp_proj_type[3] = "G5G2";
  info.thrp_proj_type[4] = "G5G3";

  info.baryon_type[0] = "nucl_nucl";
  info.baryon_type[1] = "nucl_roper";
  info.baryon_type[2] = "roper_nucl";
  info.baryon_type[3] = "roper_roper";
  info.baryon_type[4] = "deltapp_deltamm_11";
  info.baryon_type[5] = "deltapp_deltamm_22";
  info.baryon_type[6] = "deltapp_deltamm_33";
  info.baryon_type[7] = "deltap_deltaz_11";
  info.baryon_type[8] = "deltap_deltaz_22";
  info.baryon_type[9] = "deltap_deltaz_33";

  info.meson_type[0] = "pseudoscalar";
  info.meson_type[1] = "scalar";
  info.meson_type[2] = "g5g1";
  info.meson_type[3] = "g5g2";
  info.meson_type[4] = "g5g3";
  info.meson_type[5] = "g5g4";
  info.meson_type[6] = "g1";
  info.meson_type[7] = "g2";
  info.meson_type[8] = "g3";
  info.meson_type[9] = "g4";

  printfQuda("\nThe total number of source-positions is %d\n",info.Nsources);

  int nRun3pt = 0;
  for(int i=0;i<info.Nsources;i++) nRun3pt += info.run3pt_src[i];

  int NprojMax = 0;
  if(nRun3pt==0) printfQuda("%s: Will NOT perform the three-point function for any of the source positions\n", fname);
  else if (nRun3pt>0){
    printfQuda("Will perform the three-point function for %d source positions, for the following source-sink separations and projectors:\n",nRun3pt);
    for(int its=0;its<info.Ntsink;its++){
      if(info.Nproj[its] >= NprojMax) NprojMax = info.Nproj[its];
      
      printfQuda(" sink-source = %d:\n",info.tsinkSource[its]);
      for(int p=0;p<info.Nproj[its];p++) 
	printfQuda("  %s\n",info.thrp_proj_type[info.proj_list[its][p]]);
    }
  }
  else errorQuda("%s: Check your option for running the three-point function! Exiting.\n", fname);
  //-C.K. Determine whether to write the correlation functions in position/momentum space, and
  //- determine whether to write the correlation functions in High-Momenta Form
  CORR_SPACE CorrSpace = info.CorrSpace; 
  bool HighMomForm = info.HighMomForm;   

  printfQuda("\n");
  if(CorrSpace==POSITION_SPACE && HighMomForm){
    warningQuda("High-Momenta Form not applicable when writing in position-space! Switching to standard form...\n");
    HighMomForm = false;
  }

  //-C.K. We do these to switches so that the run does not go wasted.
  //-C.K. (ASCII format can be obtained with another third-party program, if desired)
  if( (CorrSpace==POSITION_SPACE || HighMomForm) && info.CorrFileFormat==ASCII_FORM ){
    if(CorrSpace==POSITION_SPACE) warningQuda("ASCII format not supported for writing the correlation functions in position-space!\n");
    if(HighMomForm) warningQuda("ASCII format not supported for High-Momenta Form!\n");
    printfQuda("Switching to HDF5 format...\n");
    info.CorrFileFormat = HDF5_FORM;
  }
  FILE_WRITE_FORMAT CorrFileFormat = info.CorrFileFormat;
  
  printfQuda("Will write the correlation functions in %s-space!\n" , (CorrSpace == POSITION_SPACE) ? "position" : "momentum");
  printfQuda("Will write the correlation functions in %s!\n"       , HighMomForm ? "High-Momenta Form" : "Normal Form");
  printfQuda("Will write the correlation functions in %s format!\n", (CorrFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda("\n");
  

  //======================================================================//
  //================ M E M O R Y   A L L O C A T I O N ===================// 
  //======================================================================//


  //-Allocate the Two-point and Three-point function data buffers
  long int alloc_size;
  if(CorrSpace==MOMENTUM_SPACE) alloc_size = GK_localL[3]*GK_Nmoms;
  else if(CorrSpace==POSITION_SPACE) alloc_size = GK_localVolume;

  //-Three-Point function
  double *corrThp_local   = (double*)calloc(alloc_size  *16*2,sizeof(double));
  double *corrThp_noether = (double*)calloc(alloc_size*4   *2,sizeof(double));
  double *corrThp_oneD    = (double*)calloc(alloc_size*4*16*2,sizeof(double));
  if(corrThp_local == NULL || 
     corrThp_noether == NULL || 
     corrThp_oneD == NULL) 
    errorQuda("%s: Cannot allocate memory for Three-point function write Buffers.", fname);
  
  //-Two-point function
  double (*corrMesons)[2][N_MESONS] = 
    (double(*)[2][N_MESONS]) calloc(alloc_size*2*N_MESONS*2,sizeof(double));
  double (*corrBaryons)[2][N_BARYONS][4][4] = 
    (double(*)[2][N_BARYONS][4][4]) calloc(alloc_size*2*N_BARYONS*4*4*2,sizeof(double));
  if(corrMesons == NULL || 
     corrBaryons == NULL) 
    errorQuda("%s: Cannot allocate memory for 2-point function write Buffers.", fname);
  

  //-HDF5 buffers for the three-point and two-point function
  double *Thrp_local_HDF5   = 
    (double*) malloc(2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
  double *Thrp_noether_HDF5 = 
    (double*) malloc(2* 4*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
  double **Thrp_oneD_HDF5   = 
    (double**) malloc(4*sizeof(double*));
  for(int mu=0;mu<4;mu++){
    Thrp_oneD_HDF5[mu] = 
      (double*) malloc(2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
  }   

  double *Twop_baryons_HDF5 = 
    (double*) malloc(2*16*alloc_size*2*N_BARYONS*sizeof(double));
  double *Twop_mesons_HDF5  = 
    (double*) malloc(2   *alloc_size*2*N_MESONS *sizeof(double));
  
  //Settings for HDF5 data write format
  if( CorrFileFormat==HDF5_FORM ){
    if( Thrp_local_HDF5 == NULL ) 
      errorQuda("%s: Cannot allocate memory for Thrp_local_HDF5.\n",fname);
    if( Thrp_noether_HDF5 == NULL ) 
      errorQuda("%s: Cannot allocate memory for Thrp_noether_HDF5.\n",fname);

    memset(Thrp_local_HDF5  , 0, 2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
    memset(Thrp_noether_HDF5, 0, 2* 4*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));

    if( Thrp_oneD_HDF5 == NULL ) 
      errorQuda("%s: Cannot allocate memory for Thrp_oneD_HDF5.\n",fname);
    for(int mu=0;mu<4;mu++){
      if( Thrp_oneD_HDF5[mu] == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_oned_HDF5[%d].\n",fname,mu);      
      memset(Thrp_oneD_HDF5[mu], 0, 2*16*alloc_size*2*info.Ntsink*NprojMax*sizeof(double));
    }
    
    if( Twop_baryons_HDF5 == NULL ) 
      errorQuda("%s: Cannot allocate memory for Twop_baryons_HDF5.\n",fname);
    if( Twop_mesons_HDF5  == NULL ) 
      errorQuda("%s: Cannot allocate memory for Twop_mesons_HDF5.\n",fname);

    memset(Twop_baryons_HDF5, 0, 2*16*alloc_size*2*N_BARYONS*sizeof(double));
    memset(Twop_mesons_HDF5 , 0, 2   *alloc_size*2*N_MESONS *sizeof(double));
  }

  //QKXTM specific objects
  QKXTM_Gauge<float> *K_gaugeContractions = 
    new QKXTM_Gauge<float>(BOTH,GAUGE);
  QKXTM_Gauge<double> *K_gaugeSmeared = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
  QKXTM_Vector<double> *K_vector = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<double> *K_guess = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<float> *K_temp = 
    new QKXTM_Vector<float>(BOTH,VECTOR);

  QKXTM_Propagator<float> *K_prop_up = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator<float> *K_prop_down = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR); 
  QKXTM_Propagator<float> *K_seqProp = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D<float> *K_prop3D_up = 
    new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D<float> *K_prop3D_down = 
    new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);

  QKXTM_Contraction<float> *K_contract = 
    new QKXTM_Contraction<float>();

  printfQuda("QKXTM memory allocation was successfull\n");

  //======================================================================//
  //================ P R O B L E M   C O N S T R U C T ===================// 
  //======================================================================//

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);
  
  //Unsmeared gauge field used for contractions
  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  printfQuda("Unsmeared:\n");
  K_gaugeContractions->calculatePlaq();
  
  //Smeared gauge field used for source construction
  K_gaugeSmeared->packGauge(gauge_APE);
  K_gaugeSmeared->loadGauge();
  printfQuda("Smeared:\n");
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = ((param->solution_type == QUDA_MAT_SOLUTION) || 
		       (param->solution_type == QUDA_MATPC_SOLUTION));
  bool direct_solve = true;

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? 
		       sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  //ensure mu is +ve
  if(param->mu < 0) param->mu *= -1.0;
  Dirac *dUP = NULL;
  Dirac *dSloppyUP = NULL;
  Dirac *dPreUP = NULL;
  // create the dirac operator
  createDirac(dUP, dSloppyUP, dPreUP, *param, pc_solve);
  Dirac &diracUP = *dUP;
  Dirac &diracSloppyUP = *dSloppyUP;
  Dirac &diracPreUP = *dPreUP;

  //ensure mu is -ve
  if(param->mu > 0) param->mu *= -1.0;
  
  Dirac *dDN = NULL;
  Dirac *dSloppyDN = NULL;
  Dirac *dPreDN = NULL;
  // create the dirac operator
  createDirac(dDN, dSloppyDN, dPreDN, *param, pc_solve);
  Dirac &diracDN = *dDN;
  Dirac &diracSloppyDN = *dSloppyDN;
  Dirac &diracPreDN = *dPreDN;

  //revert to +ve mu
  if(param->mu < 0) param->mu *= -1.0;
 
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(GK_localL[0]*
			      GK_localL[1]*
			      GK_localL[2]*
			      GK_localL[3]*spinorSiteSize*sizeof(double));
  
  void *output_vector = malloc(GK_localL[0]*
			       GK_localL[1]*
			       GK_localL[2]*
			       GK_localL[3]*spinorSiteSize*sizeof(double));
  
  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, 
			    param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);
  
  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  
  // Create Operators
  DiracM mUP(diracUP), mSloppyUP(diracSloppyUP), mPreUP(diracPreUP);
  DiracM mDN(diracDN), mSloppyDN(diracSloppyDN), mPreDN(diracPreDN);
 
  // Create Solvers
  if(param->mu < 0) param->mu *= -1.0;
  param->preconditioner = param->preconditionerUP;
  SolverParam solverParamU(*param);
  Solver *solveU = Solver::create(solverParamU, mUP, mSloppyUP, 
				  mPreUP, profileInvert);

  if(param->mu > 0) param->mu *= -1.0;
  param->preconditioner = param->preconditionerDOWN;
  SolverParam solverParamD(*param);
  Solver *solveD = Solver::create(solverParamD, mDN, mSloppyDN, 
				  mPreDN, profileInvert);
  
  //======================================================================//
  //================ P R O B L E M   E X E C U T I O N  ==================// 
  //======================================================================//
 
  //We loop over all source positions and spin-colour componenets

  for(int isource = 0 ; isource < info.Nsources ; isource++){
    t5 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       isource,
	       info.sourcePosition[isource][0],
	       info.sourcePosition[isource][1],
	       info.sourcePosition[isource][2],
	       info.sourcePosition[isource][3]);
    
    if( CorrFileFormat==ASCII_FORM ){
      sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",
	      filename_twop,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);      
      sprintf(filename_baryons,"%s.baryons.SS.%02d.%02d.%02d.%02d.dat",
	      filename_twop,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);
    }
    else if( CorrFileFormat==HDF5_FORM ){
      char *str;
      if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
      else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");
      sprintf(filename_mesons ,"%s_mesons_%s_SS.%02d.%02d.%02d.%02d.h5" ,
	      filename_twop,str,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);
      sprintf(filename_baryons,"%s_baryons_%s_SS.%02d.%02d.%02d.%02d.h5",
	      filename_twop,str,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);
    }
    
    if(info.check_files){
      bool checkMesons, checkBaryons;
      checkMesons = exists_file(filename_mesons);
      checkBaryons = exists_file(filename_baryons);
      if( (checkMesons == true) && (checkBaryons == true) ) continue;
    }
    
    printfQuda("Forward Inversions:\n");
    t1 = MPI_Wtime();
    for(int isc = 0 ; isc < 12 ; isc++){

      t4 = MPI_Wtime();

      ///////////////////////////////
      // Forward prop for up quark //
      ///////////////////////////////
      memset(input_vector,0,
	     X[0]*X[1]*X[2]*X[3]*
	     spinorSiteSize*sizeof(double));

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = (info.sourcePosition[isource][i] - 
		     comm_coords(default_topo)[i] * X[i]);
      
      if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	  (my_src[1]>=0) && (my_src[1]<X[1]) && 
	  (my_src[2]>=0) && (my_src[2]<X[2]) && 
	  (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + 
	   my_src[3]*X[2]*X[1]*X[0]*24 + 
	   my_src[2]*X[1]*X[0]*24 + 
	   my_src[1]*X[0]*24 + 
	   my_src[0]*24 + 
	   isc*2 ) = 1.0;
      
      //Ensure mu is +ve
      if(param->mu < 0) param->mu *= -1.0;
      mapNormalToEvenOdd(input_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
      tx1 = MPI_Wtime();
      performWuppertalnStep(output_vector, input_vector, param, GK_nsmearGauss, GK_alphaGauss);
      tx2 = MPI_Wtime();
      summ_tx12 += tx2-tx1;
      mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
      K_guess->packVector((double*) output_vector);
      K_guess->loadVector();
      K_guess->uploadToCuda(b,flag_eo);
      blas::zero(*x);
      diracUP.prepare(in,out,*x,*b,param->solution_type);

      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet
      
      printfQuda(" up - %02d: \n",isc);
      tx3 = MPI_Wtime();
      (*solveU)(*out,*in);
      tx4 = MPI_Wtime();
      summ_tx34 += tx4-tx3;
      solverParamU.updateInvertParam(*param);
      diracUP.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      
      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);

      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",
		 isc,isource,t2-t4);

      if(isc == 0) saveTuneCache();
      
      /////////////////////////////////
      // Forward prop for down quark //
      /////////////////////////////////
      memset(input_vector,0,
	     X[0]*X[1]*X[2]*X[3]*
	     spinorSiteSize*sizeof(double));

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = (info.sourcePosition[isource][i] - 
		     comm_coords(default_topo)[i] * X[i]);

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	  (my_src[1]>=0) && (my_src[1]<X[1]) && 
	  (my_src[2]>=0) && (my_src[2]<X[2]) && 
	  (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + 
	   my_src[3]*X[2]*X[1]*X[0]*24 + 
	   my_src[2]*X[1]*X[0]*24 + 
	   my_src[1]*X[0]*24 + 
	   my_src[0]*24 + 
	   isc*2 ) = 1.0;

      //Ensure mu is -ve
      if(param->mu > 0) param->mu *= -1.0;
      K_guess->uploadToCuda(b,flag_eo);
      blas::zero(*x);
      diracDN.prepare(in,out,*x,*b,param->solution_type);
      printfQuda(" dn - %02d: \n",isc);
      tx3 = MPI_Wtime();
      (*solveD)(*out,*in);
      tx4 = MPI_Wtime();
      summ_tx34 += tx4-tx3;
      solverParamD.updateInvertParam(*param);
      diracDN.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);

      t4 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);

    } 
    // Close loop over 12 spin-color
    
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Gaussian Smearing (For just the source point, all spin-color components): %f sec. \n",summ_tx12);
    summ_tx12=0.;
    printfQuda("TIME_REPORT - Just Inversions (24 in total): %f sec. \n", summ_tx34);
    summ_tx34=0.;
    printfQuda("TIME_REPORT - Forward Inversions (Total including smearing, inversions and others): %f sec.\n\n",t2-t1);
    
    ////////////////////////////////////
    // Smearing on the 3D propagators //
    ////////////////////////////////////
    
    if(info.run3pt_src[isource]){
      
      //-C.K: Loop over the number of sink-source separations
      int my_fixSinkTime;
      char *filename_threep_base;
      for(int its=0;its<info.Ntsink;its++){
	my_fixSinkTime = 
	  (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] 
	  - comm_coords(default_topo)[3] * X[3];
	
	t1 = MPI_Wtime();
	K_temp->zero_device();
	checkCudaError();
	if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	  K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	  K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
	}
	comm_barrier();

	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    ////////
	    // up //
	    ////////
	    K_temp->zero_device();
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_temp->copyPropagator3D(*K_prop3D_up,
				       my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_prop3D_up->absorbVectorTimeSlice(*K_temp,
						 my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_temp->zero_device();
	    //////////
	    // down //
	    //////////
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_temp->copyPropagator3D(*K_prop3D_down,
				       my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_prop3D_down->absorbVectorTimeSlice(*K_temp,
						   my_fixSinkTime,nu,c2);
	    comm_barrier();
	    
	    K_temp->zero_device();	
	  }
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT - 3d Props preparation for sink-source[%d]=%d: %f sec\n",
		   its,info.tsinkSource[its],t2-t1);
	
	for(int proj=0;proj<info.Nproj[its];proj++){
	  WHICHPROJECTOR PID = (WHICHPROJECTOR) info.proj_list[its][proj];
	  char *proj_str;
	  asprintf(&proj_str,"%s",
		   info.thrp_proj_type[info.proj_list[its][proj]]);
	  
	  printfQuda("\n# Three-point function calculation for source-position = %d, sink-source = %d, projector %s begins now\n",
		     isource,info.tsinkSource[its],proj_str);
	  
	  if( CorrFileFormat==ASCII_FORM ){
	    asprintf(&filename_threep_base,"%s_tsink%d_proj%s",
		     filename_threep,info.tsinkSource[its],proj_str);
	    printfQuda("The three-point function ASCII base name is: %s\n",
		       filename_threep_base);
	  }
	  
	  
	  //////////////////////////////////////
	  // Sequential propagator for part 1 //
	  //////////////////////////////////////
	  printfQuda("Sequential Inversions, flavor %s:\n",
		     NUCLEON == NEUTRON ? "dn" : "up");
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();
	      if(NUCLEON == PROTON){
		//Ensure mu is -ve:
		if(param->mu > 0) param->mu *= -1.0;		
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		  K_contract->seqSourceFixSinkPart1(*K_temp,
						    *K_prop3D_up, 
						    *K_prop3D_down, 
						    my_fixSinkTime, nu, c2, 
						    PID, NUCLEON);
	      }
	      else if(NUCLEON == NEUTRON){
		//Ensure mu is +ve:
		if(param->mu < 0) param->mu *= -1.0;		
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		  K_contract->seqSourceFixSinkPart1(*K_temp,
						    *K_prop3D_down, 
						    *K_prop3D_up, 
						    my_fixSinkTime, nu, c2, 
						    PID, NUCLEON);
	      }
	      comm_barrier();
	      K_temp->conjugate();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      //Scale up vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      	      K_guess->uploadToCuda(b,flag_eo);

	  
	      if(NUCLEON == PROTON){
		diracDN.prepare(in,out,*x,*b,param->solution_type);
		//Ensure mu is -ve
		if(param->mu > 0) param->mu *= -1.0;
	      
		K_vector->downloadFromCuda(in,flag_eo);
		K_vector->download();
		K_guess->uploadToCuda(out,flag_eo); 
		// initial guess is ready
	      
		printfQuda("%02d - \n",nu*3+c2);
		(*solveD)(*out,*in);
		solverParamD.updateInvertParam(*param);
		diracDN.reconstruct(*x,*b,param->solution_type);

	      }
	      else if(NUCLEON == NEUTRON){
		diracUP.prepare(in,out,*x,*b,param->solution_type);
		//Ensure mu is +ve
		if(param->mu < 0) param->mu *= -1.0;
		
		K_vector->downloadFromCuda(in,flag_eo);
		K_vector->download();
		K_guess->uploadToCuda(out,flag_eo); 
		// initial guess is ready
	      
		printfQuda("%02d - \n",nu*3+c2);
		(*solveU)(*out,*in);
		solverParamU.updateInvertParam(*param);
		diracUP.reconstruct(*x,*b,param->solution_type);

	      }

	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }
	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	      
	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for seq prop part 1 = %d, source = %d at sink-source = %d, projector %s is: %f sec\n",
			 nu*3+c2,isource,info.tsinkSource[its],proj_str,t4-t3);
	    }
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, flavor %s: %f sec\n",
		     NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  
	  /////////////////////////////
	  // Contractions for part 1 //
	  /////////////////////////////
	  t1 = MPI_Wtime();
	  if(NUCLEON == PROTON) 
	    K_contract->contractFixSink(*K_seqProp, *K_prop_up, 
					*K_gaugeContractions,
					corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 1, 
					isource, CorrSpace);
	  if(NUCLEON == NEUTRON) 
	    K_contract->contractFixSink(*K_seqProp, *K_prop_down, 
					*K_gaugeContractions, 
					corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 1, 
					isource, CorrSpace);
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Three-point Contractions, flavor %s: %f sec\n",
		     NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  
	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 1, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, proj %s, flavor %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],proj_str,
		       NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd;
	    if(NUCLEON == PROTON ) uOrd = 0;
	    if(NUCLEON == NEUTRON) uOrd = 1;
	    
	    int thrp_sign = ( info.tsinkSource[its] + GK_sourcePosition[isource][3] ) >= GK_totalL[3] ? -1 : +1;
	    
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, proj, 
					   thrp_sign, THRP_LOCAL, CorrSpace,
					   HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, proj, 
					   thrp_sign,THRP_NOETHER,CorrSpace,
					   HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_HDF5[mu],
					     (void*)corrThp_oneD, mu, 
					     uOrd, its, info.Ntsink, proj, 
					     thrp_sign, THRP_ONED, CorrSpace,
					     HighMomForm);
	    
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - 3-point function for flavor %s copied to HDF5 write buffers in %f sec.\n", NUCLEON == NEUTRON ? "dn" : "up",t2-t1);
	  }
	  
	  //////////////////////////////////////
	  // Sequential propagator for part 2 //
	  //////////////////////////////////////
	  printfQuda("Sequential Inversions, flavor %s:\n",
		     NUCLEON == NEUTRON ? "up" : "dn");
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();
	      if(NUCLEON == PROTON){
		//Ensure mu is +ve
		if(param->mu < 0) param->mu *= -1.0;		
		if( ( my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		  K_contract->seqSourceFixSinkPart2(*K_temp,
						    *K_prop3D_up, 
						    my_fixSinkTime, nu, c2, 
						    PID, NUCLEON);
	      }
	      else if(NUCLEON == NEUTRON){
		//Ensure mu is -ve
		if(param->mu > 0) param->mu *= -1.0;
		if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		  K_contract->seqSourceFixSinkPart2(*K_temp,
						    *K_prop3D_down, 
						    my_fixSinkTime, nu, c2, 
						    PID, NUCLEON);
	      }
	      comm_barrier();
	      K_temp->conjugate();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      // Scale vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_guess->uploadToCuda(b,flag_eo);
	      
	      if(NUCLEON == PROTON){
		diracUP.prepare(in,out,*x,*b,param->solution_type);
		//Ensure mu is +ve
		if(param->mu < 0) param->mu *= -1.0;
		
		K_vector->downloadFromCuda(in,flag_eo);
		K_vector->download();
		K_guess->uploadToCuda(out,flag_eo); 
		// initial guess is ready
		
		printfQuda("%02d - ",nu*3+c2);
		(*solveU)(*out,*in);
		solverParamU.updateInvertParam(*param);
		diracUP.reconstruct(*x,*b,param->solution_type);

	      }
	      else if(NUCLEON == NEUTRON){
		diracDN.prepare(in,out,*x,*b,param->solution_type);
		//Ensure mu is -ve
		if(param->mu > 0) param->mu *= -1.0;
		
		K_vector->downloadFromCuda(in,flag_eo);
		K_vector->download();
		K_guess->uploadToCuda(out,flag_eo); 
		// initial guess is ready
		
		printfQuda("%02d - ",nu*3+c2);
		(*solveD)(*out,*in);
		solverParamD.updateInvertParam(*param);
		diracDN.reconstruct(*x,*b,param->solution_type);
		
	      }
	      
	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }

	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);

	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for seq prop part 2 = %d, source = %d at sink-source = %d, projector %s is: %f sec\n", 
			 nu*3+c2,isource,info.tsinkSource[its],
			 proj_str,t4-t3);
	    }
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "up" : "dn",t2-t1);

	  /////////////////////////////
	  // Contractions for part 2 //
	  /////////////////////////////
	  t1 = MPI_Wtime();
	  if(NUCLEON == PROTON) 
	    K_contract->contractFixSink(*K_seqProp, 
					*K_prop_down, 
					*K_gaugeContractions, 
					corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 2, 
					isource, CorrSpace);
	  if(NUCLEON == NEUTRON) 
	    K_contract->contractFixSink(*K_seqProp, *K_prop_up, 
					*K_gaugeContractions, 
					corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 2, 
					isource, CorrSpace);
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Three-point Contractions, flavor %s: %f sec\n",NUCLEON == NEUTRON ? "up" : "dn",t2-t1);

	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, NUCLEON, 2, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, proj %s, flavor %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],proj_str,
		       NUCLEON == NEUTRON ? "up" : "dn",t2-t1);
	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd;
	    if(NUCLEON == PROTON ) uOrd = 1;
	    if(NUCLEON == NEUTRON) uOrd = 0;
	    
	    int thrp_sign = (info.tsinkSource[its] + GK_sourcePosition[isource][3] ) >= GK_totalL[3] ? -1 : +1;

	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, 
					   proj, thrp_sign, THRP_LOCAL, 
					   CorrSpace, HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, 
					   proj, thrp_sign, THRP_NOETHER, 
					   CorrSpace, HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_HDF5[mu], 
					     (void*)corrThp_oneD,mu, 
					     uOrd, its, info.Ntsink, 
					     proj, thrp_sign, THRP_ONED, 
					     CorrSpace, HighMomForm);
	    
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - 3-point function for flavor %s copied to HDF5 write buffers in %f sec.\n", 
		       NUCLEON == NEUTRON ? "up" : "dn",t2-t1);
	  }	  
	} 
	// End loop over projectors
      }
      // End loop over sink-source separations      
      
      //-C.K. Write the three-point function in HDF5 format
      if( CorrFileFormat==HDF5_FORM ){
	char *str;
	if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
	else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");

	t1 = MPI_Wtime();
	asprintf(&filename_threep_base,"%s_%s_%s_SS.%02d.%02d.%02d.%02d.h5",
		 filename_threep, 
		 (NUCLEON == PROTON) ? "proton" : "neutron", str, 
		 GK_sourcePosition[isource][0],
		 GK_sourcePosition[isource][1],
		 GK_sourcePosition[isource][2],
		 GK_sourcePosition[isource][3]);
	printfQuda("\nThe three-point function HDF5 filename is: %s\n",
		   filename_threep_base);
	
	K_contract->writeThrpHDF5((void*) Thrp_local_HDF5, 
				  (void*) Thrp_noether_HDF5, 
				  (void**)Thrp_oneD_HDF5, 
				  filename_threep_base, info, 
				  isource, NUCLEON);
	
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d written in HDF5 format in %f sec.\n",isource,t2-t1);
      }
      
      printfQuda("\n");
    }
    // End loop if running for the specific isource
    
    ///////////////////////////////////
    // Smear the forward propagators //
    ///////////////////////////////////

    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
    	K_temp->copyPropagator(*K_prop_up,nu,c2);
    	K_vector->castFloatToDouble(*K_temp);
	K_vector->download();
	mapNormalToEvenOdd((void*) K_vector->H_elem() , *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	performWuppertalnStep(output_vector, (void*) K_vector->H_elem(), param, GK_nsmearGauss, GK_alphaGauss);
	mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	K_guess->packVector((double*) output_vector);
	K_guess->loadVector();
    	K_temp->castDoubleToFloat(*K_guess);
    	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
    	K_temp->copyPropagator(*K_prop_down,nu,c2);
    	K_vector->castFloatToDouble(*K_temp);
	K_vector->download();
	mapNormalToEvenOdd((void*) K_vector->H_elem() , *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	performWuppertalnStep(output_vector, (void*) K_vector->H_elem(), param, GK_nsmearGauss, GK_alphaGauss);
	mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	K_guess->packVector((double*) output_vector);
	K_guess->loadVector();
    	K_temp->castDoubleToFloat(*K_guess);
    	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractBaryons(*K_prop_up,*K_prop_down, corrBaryons, 
				isource, CorrSpace);
    K_contract->contractMesons (*K_prop_up,*K_prop_down, corrMesons, 
				isource, CorrSpace);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Two-point Contractions: %f sec\n",t2-t1);
    
    //======================================================================//
    //===================== W R I T E   D A T A  ===========================//
    //======================================================================//

    printfQuda("\nThe baryons two-point function %s filename is: %s\n",
	       (CorrFileFormat==ASCII_FORM) ? "ASCII" : "HDF5",
	       filename_baryons);
    printfQuda("The mesons two-point function %s filename is: %s\n" ,
	       (CorrFileFormat==ASCII_FORM) ? "ASCII" : "HDF5",
	       filename_mesons);
    
    if( CorrFileFormat==ASCII_FORM ){
      t1 = MPI_Wtime();
      K_contract->writeTwopBaryons_ASCII(corrBaryons, filename_baryons, 
					 isource, CorrSpace);
      K_contract->writeTwopMesons_ASCII (corrMesons , filename_mesons , 
					 isource, CorrSpace);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Mesons and Baryons for source-position = %d written in ASCII format in %f sec.\n",
		 isource,t2-t1);
    }    
    else if( CorrFileFormat==HDF5_FORM ){
      t1 = MPI_Wtime();
      K_contract->copyTwopBaryonsToHDF5_Buf((void*)Twop_baryons_HDF5, 
					    (void*)corrBaryons, isource, 
					    CorrSpace, HighMomForm);
      K_contract->copyTwopMesonsToHDF5_Buf ((void*)Twop_mesons_HDF5 , 
					    (void*)corrMesons, CorrSpace,
					    HighMomForm);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Two-point function for baryons and mesons copied to HDF5 write buffers in %f sec.\n",t2-t1);
      
      t1 = MPI_Wtime();
      K_contract->writeTwopBaryonsHDF5((void*) Twop_baryons_HDF5, 
				       filename_baryons, info, isource);
      K_contract->writeTwopMesonsHDF5 ((void*) Twop_mesons_HDF5, 
				       filename_mesons , info, isource);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Baryons and Mesons for source-position = %d written in HDF5 format in %f sec.\n",
		 isource,t2-t1);
    }
    
    t6 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d Completed in %f sec. ###\n",isource, 
	       info.sourcePosition[isource][0],
	       info.sourcePosition[isource][1],
	       info.sourcePosition[isource][2],
	       info.sourcePosition[isource][3],
	       t6-t5);
  } 
  // close loop over source positions

  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//  
  
  printfQuda("\nCleaning up...\n");
  free(corrThp_local);
  free(corrThp_noether);
  free(corrThp_oneD);
  free(corrMesons);
  free(corrBaryons);

  if( CorrFileFormat==HDF5_FORM ){
    free(Thrp_local_HDF5);
    free(Thrp_noether_HDF5);
    for(int mu=0;mu<4;mu++) free(Thrp_oneD_HDF5[mu]);
    free(Thrp_oneD_HDF5);
    free(Twop_baryons_HDF5);
    free(Twop_mesons_HDF5);
  }

  free(input_vector);
  free(output_vector);
  delete K_temp;
  delete K_contract;
  delete K_prop_down;
  delete K_prop_up;
  delete dUP;
  delete dSloppyUP;
  delete dPreUP;
  delete dDN;
  delete dSloppyDN;
  delete dPreDN;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;
  delete solveU;
  delete solveD;

  printfQuda("...Done\n");
  
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

void calcMG_threepTwop_Mesons(void **gauge_APE, void **gauge, 
			      QudaGaugeParam *gauge_param, 
			      QudaInvertParam *param, 
			      qudaQKXTMinfo info, 
			      char *filename_twop, 
			      char *filename_threep, 
			      WHICHPARTICLE MESON){
  
  bool flag_eo;
  double t1,t2,t3,t4,t5,t6;
  double tx1,tx2,summ_tx12=0.; // needed to time gaussian smearing routines
  double tx3,tx4,summ_tx34=0.; // needed to time just the inversion time
  char fname[256];
  sprintf(fname, "calcMG_threepTwop_Meson");

  //======================================================================//
  //= P A R A M E T E R   C H E C K S   A N D   I N I T I A L I S I O N ==//
  //======================================================================//
  
  if (!initialized)
    errorQuda("%s: QUDA not initialized", fname);

  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
    printQudaInvertParam(param);
  
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  
  if((param->inv_type != QUDA_GCR_INVERTER) && (param->inv_type != QUDA_CG_INVERTER) )
    errorQuda("%s: This function works only with GCR/CG solver", fname);  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("%s: This function works only with ukqcd gamma basis\n", fname);
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("%s: This function works only with colors inside the spins\n", fname);

  if( param->inv_type == QUDA_GCR_INVERTER )
    if(param->solve_type != QUDA_DIRECT_PC_SOLVE) 
      errorQuda("%s: If this function is using gcr inverter, itworks only with Direct solve and even odd preconditioning", fname);

  if( param->matpc_type == QUDA_MATPC_EVEN_EVEN )
    flag_eo = true;
  else if(param->matpc_type == QUDA_MATPC_ODD_ODD )
    flag_eo = false;

  int my_src[4];
  char filename_mesons[257];

  info.thrp_type[0] = "ultra_local";
  info.thrp_type[1] = "noether";
  info.thrp_type[2] = "oneD";

  info.meson_type[0] = "pseudoscalar";
  info.meson_type[1] = "scalar";
  info.meson_type[2] = "g5g1";
  info.meson_type[3] = "g5g2";
  info.meson_type[4] = "g5g3";
  info.meson_type[5] = "g5g4";
  info.meson_type[6] = "g1";
  info.meson_type[7] = "g2";
  info.meson_type[8] = "g3";
  info.meson_type[9] = "g4";

  printfQuda("\nThe total number of source-positions is %d\n",info.Nsources);

  int nRun3pt = 0;
  for(int i=0;i<info.Nsources;i++) nRun3pt += info.run3pt_src[i];

  if(nRun3pt==0) printfQuda("%s: Will NOT perform the three-point function for any of the source positions\n", fname);
  else if (nRun3pt>0){
    printfQuda("Will perform the three-point function for %d source positions, for the following source-sink separations:\n",nRun3pt);
    for(int its=0;its<info.Ntsink;its++){
      printfQuda(" sink-source = %d:\n",info.tsinkSource[its]);
    }
  }
  else errorQuda("%s: Check your option for running the three-point function! Exiting.\n", fname);
  //-C.K. Determine whether to write the correlation functions in position/momentum space, and
  //- determine whether to write the correlation functions in High-Momenta Form
  CORR_SPACE CorrSpace = info.CorrSpace; 
  bool HighMomForm = info.HighMomForm;   

  printfQuda("\n");
  if(CorrSpace==POSITION_SPACE && HighMomForm){
    warningQuda("High-Momenta Form not applicable when writing in position-space! Switching to standard form...\n");
    HighMomForm = false;
  }

  //-C.K. We do these to switches so that the run does not go wasted.
  //-C.K. (ASCII format can be obtained with another third-party program, if desired)
  if( (CorrSpace==POSITION_SPACE || HighMomForm) && info.CorrFileFormat==ASCII_FORM ){
    if(CorrSpace==POSITION_SPACE) warningQuda("ASCII format not supported for writing the correlation functions in position-space!\n");
    if(HighMomForm) warningQuda("ASCII format not supported for High-Momenta Form!\n");
    printfQuda("Switching to HDF5 format...\n");
    info.CorrFileFormat = HDF5_FORM;
  }
  FILE_WRITE_FORMAT CorrFileFormat = info.CorrFileFormat;
  
  printfQuda("Will write the correlation functions in %s-space!\n" , (CorrSpace == POSITION_SPACE) ? "position" : "momentum");
  printfQuda("Will write the correlation functions in %s!\n"       , HighMomForm ? "High-Momenta Form" : "Normal Form");
  printfQuda("Will write the correlation functions in %s format!\n", (CorrFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda("\n");
  

  //======================================================================//
  //================ M E M O R Y   A L L O C A T I O N ===================// 
  //======================================================================//


  //-Allocate the Two-point and Three-point function data buffers
  long int alloc_size;
  if(CorrSpace==MOMENTUM_SPACE) alloc_size = GK_localL[3]*GK_Nmoms;
  else if(CorrSpace==POSITION_SPACE) alloc_size = GK_localVolume;

  //-Three-Point function
  double *corrThp_local   = (double*)calloc(alloc_size  *16*2,sizeof(double));
  double *corrThp_noether = (double*)calloc(alloc_size*4   *2,sizeof(double));
  double *corrThp_oneD    = (double*)calloc(alloc_size*4*16*2,sizeof(double));
  if(corrThp_local == NULL ||
     corrThp_noether == NULL || 
     corrThp_oneD == NULL
     )
    errorQuda("%s: Cannot allocate memory for Three-point function write Buffers.", fname);
  
  //-Two-point function
  double (*corrMesons)[2][N_MESONS] = 
    (double(*)[2][N_MESONS]) calloc(alloc_size*2*N_MESONS*2,sizeof(double));
  if(corrMesons == NULL) 
    errorQuda("%s: Cannot allocate memory for 2-point function write Buffers.", fname);
  

  //-HDF5 buffers for the three-point and two-point function
  double *Thrp_local_pion_HDF5;   
  double *Thrp_noether_pion_HDF5; 
  double **Thrp_oneD_pion_HDF5;   

  if( MESON == PION || MESON == ALL_MESONS ) {
    Thrp_local_pion_HDF5 = (double*) malloc(2*16*alloc_size*2*info.Ntsink*sizeof(double));
    Thrp_noether_pion_HDF5 = (double*) malloc(2* 4*alloc_size*2*info.Ntsink*sizeof(double));
    Thrp_oneD_pion_HDF5 = (double**) malloc(4*sizeof(double*));
    for(int mu=0;mu<4;mu++){
      Thrp_oneD_pion_HDF5[mu] = 
	(double*) malloc(2*16*alloc_size*2*info.Ntsink*sizeof(double));
    }   
  }

  double *Thrp_local_kaon_HDF5;   
  double *Thrp_noether_kaon_HDF5; 
  double **Thrp_oneD_kaon_HDF5;   

  if( MESON == KAON || MESON == ALL_MESONS ) {
    Thrp_local_kaon_HDF5 = (double*) malloc(2*16*alloc_size*2*info.Ntsink*sizeof(double));
    Thrp_noether_kaon_HDF5 = (double*) malloc(2* 4*alloc_size*2*info.Ntsink*sizeof(double));
    Thrp_oneD_kaon_HDF5 = (double**) malloc(4*sizeof(double*));
    for(int mu=0;mu<4;mu++){
      Thrp_oneD_kaon_HDF5[mu] = 
	(double*) malloc(2*16*alloc_size*2*info.Ntsink*sizeof(double));
    }   
  }
  
  double *Twop_mesons_HDF5  = 
    (double*) malloc(2   *alloc_size*2*N_MESONS *sizeof(double));
  
  //Settings for HDF5 data write format
  if( CorrFileFormat==HDF5_FORM ){
    if( MESON == PION || MESON == ALL_MESONS ) {
      if( Thrp_local_pion_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_local_HDF5.\n",fname);
      if( Thrp_noether_pion_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_noether_HDF5.\n",fname);

      memset(Thrp_local_pion_HDF5  , 0, 2*16*alloc_size*2*info.Ntsink*sizeof(double));
      memset(Thrp_noether_pion_HDF5, 0, 2* 4*alloc_size*2*info.Ntsink*sizeof(double));

      if( Thrp_oneD_pion_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_oneD_HDF5.\n",fname);
      for(int mu=0;mu<4;mu++){
	if( Thrp_oneD_pion_HDF5[mu] == NULL ) 
	  errorQuda("%s: Cannot allocate memory for Thrp_oned_HDF5[%d].\n",fname,mu);      
	memset(Thrp_oneD_pion_HDF5[mu], 0, 2*16*alloc_size*2*info.Ntsink*sizeof(double));
      }
    }

    if( MESON == KAON || MESON == ALL_MESONS ) {    
      if( Thrp_local_kaon_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_local_HDF5.\n",fname);
      if( Thrp_noether_kaon_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_noether_HDF5.\n",fname);

      memset(Thrp_local_kaon_HDF5  , 0, 2*16*alloc_size*2*info.Ntsink*sizeof(double));
      memset(Thrp_noether_kaon_HDF5, 0, 2* 4*alloc_size*2*info.Ntsink*sizeof(double));

      if( Thrp_oneD_kaon_HDF5 == NULL ) 
	errorQuda("%s: Cannot allocate memory for Thrp_oneD_HDF5.\n",fname);
      for(int mu=0;mu<4;mu++){
	if( Thrp_oneD_kaon_HDF5[mu] == NULL ) 
	  errorQuda("%s: Cannot allocate memory for Thrp_oned_HDF5[%d].\n",fname,mu);      
	memset(Thrp_oneD_kaon_HDF5[mu], 0, 2*16*alloc_size*2*info.Ntsink*sizeof(double));
      }
    }
    
    if( Twop_mesons_HDF5  == NULL ) 
      errorQuda("%s: Cannot allocate memory for Twop_mesons_HDF5.\n",fname);

    memset(Twop_mesons_HDF5 , 0, 2   *alloc_size*2*N_MESONS *sizeof(double));
  }

  //QKXTM specific objects
  QKXTM_Gauge<float> *K_gaugeContractions = 
    new QKXTM_Gauge<float>(BOTH,GAUGE);
  QKXTM_Gauge<double> *K_gaugeSmeared = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
  QKXTM_Vector<double> *K_vector = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<double> *K_guess = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<float> *K_temp = 
    new QKXTM_Vector<float>(BOTH,VECTOR);

  QKXTM_Propagator<float> *K_prop_up = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR);
  QKXTM_Propagator<float> *K_prop_down = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR); 
  QKXTM_Propagator<float> *K_prop_strangePlus = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR); 
  QKXTM_Propagator<float> *K_prop_strangeMinus = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR); 
  QKXTM_Propagator<float> *K_seqProp = 
    new QKXTM_Propagator<float>(BOTH,PROPAGATOR);

  QKXTM_Propagator3D<float> *K_prop3D_up = 
    new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D<float> *K_prop3D_down = 
    new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);
  QKXTM_Propagator3D<float> *K_prop3D_strangePlus;
  QKXTM_Propagator3D<float> *K_prop3D_strangeMinus;

  if( MESON == KAON || MESON == ALL_MESONS ) {
    K_prop3D_strangePlus = 
      new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);

    K_prop3D_strangeMinus = 
      new QKXTM_Propagator3D<float>(BOTH,PROPAGATOR3D);
  }
  QKXTM_Contraction<float> *K_contract = 
    new QKXTM_Contraction<float>();

  printfQuda("QKXTM memory allocation was successfull\n");

  //======================================================================//
  //================ P R O B L E M   C O N S T R U C T ===================// 
  //======================================================================//

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);
  
  //Unsmeared gauge field used for contractions
  K_gaugeContractions->packGauge(gauge);
  K_gaugeContractions->loadGauge();
  printfQuda("Unsmeared:\n");
  K_gaugeContractions->calculatePlaq();
  
  //Smeared gauge field used for source construction
  K_gaugeSmeared->packGauge(gauge_APE);
  K_gaugeSmeared->loadGauge();
  printfQuda("Smeared:\n");
  K_gaugeSmeared->calculatePlaq();

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? 
		       sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 
			 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  //ensure mu is up flavor
  param->mu = param->mu_l;
  Dirac *dUP = NULL;
  Dirac *dSloppyUP = NULL;
  Dirac *dPreUP = NULL;
  // create the dirac operator
  createDirac(dUP, dSloppyUP, dPreUP, *param, pc_solve);
  Dirac &diracUP = *dUP;
  Dirac &diracSloppyUP = *dSloppyUP;
  Dirac &diracPreUP = *dPreUP;

  //ensure mu is down flavor
  param->mu *= -1.0 * param->mu_l;
  
  Dirac *dDOWN = NULL;
  Dirac *dSloppyDOWN = NULL;
  Dirac *dPreDOWN = NULL;
  // create the dirac operator
  createDirac(dDOWN, dSloppyDOWN, dPreDOWN, *param, pc_solve);
  Dirac &diracDOWN = *dDOWN;
  Dirac &diracSloppyDOWN = *dSloppyDOWN;
  Dirac &diracPreDOWN = *dPreDOWN;

  //ensure mu is strange plus flavor
  param->mu *= param->mu_s;
  
  Dirac *dSTRANGEPLUS = NULL;
  Dirac *dSloppySTRANGEPLUS = NULL;
  Dirac *dPreSTRANGEPLUS = NULL;
  if( MESON == KAON || MESON == ALL_MESONS ) {
    // create the dirac operator
    createDirac(dSTRANGEPLUS, dSloppySTRANGEPLUS, dPreSTRANGEPLUS, *param, pc_solve);
  }

  Dirac &diracSTRANGEPLUS = *dSTRANGEPLUS;
  Dirac &diracSloppySTRANGEPLUS = *dSloppySTRANGEPLUS;
  Dirac &diracPreSTRANGEPLUS = *dPreSTRANGEPLUS;

  //ensure mu is strange minus flavor
  param->mu *= -1.0 * param->mu_s;
  
  Dirac *dSTRANGEMINUS = NULL;
  Dirac *dSloppySTRANGEMINUS = NULL;
  Dirac *dPreSTRANGEMINUS = NULL;
  if( MESON == KAON || MESON == ALL_MESONS ) {
    // create the dirac operator
    createDirac(dSTRANGEMINUS, dSloppySTRANGEMINUS, dPreSTRANGEMINUS, *param, pc_solve);
  }

  Dirac &diracSTRANGEMINUS = *dSTRANGEMINUS;
  Dirac &diracSloppySTRANGEMINUS = *dSloppySTRANGEMINUS;
  Dirac &diracPreSTRANGEMINUS = *dPreSTRANGEMINUS;

  //revert to up flavor
  param->mu = param->mu_l;
 
  profileInvert.TPSTART(QUDA_PROFILE_H2D);

  //QKXTM: DMH rewrite for spinor field memalloc
  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  const int *X = cudaGauge->X();

  void *input_vector = malloc(GK_localL[0]*
			      GK_localL[1]*
			      GK_localL[2]*
			      GK_localL[3]*spinorSiteSize*sizeof(double));
  
  void *output_vector = malloc(GK_localL[0]*
			       GK_localL[1]*
			       GK_localL[2]*
			       GK_localL[3]*spinorSiteSize*sizeof(double));
  
  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, pc_solution, 
			    param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);
  
  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x = new cudaColorSpinorField(cudaParam);

  profileInvert.TPSTOP(QUDA_PROFILE_H2D);


  // Create Operators
  DiracM mUP(diracUP), 
    mSloppyUP(diracSloppyUP), 
    mPreUP(diracPreUP);
  DiracM mDOWN(diracDOWN), 
    mSloppyDOWN(diracSloppyDOWN), 
    mPreDOWN(diracPreDOWN);
  DiracM mSTRANGEPLUS(diracSTRANGEPLUS), 
    mSloppySTRANGEPLUS(diracSloppySTRANGEPLUS), 
    mPreSTRANGEPLUS(diracPreSTRANGEPLUS);
  DiracM mSTRANGEMINUS(diracSTRANGEMINUS), 
    mSloppySTRANGEMINUS(diracSloppySTRANGEMINUS), 
    mPreSTRANGEMINUS(diracPreSTRANGEMINUS);
  DiracMdagM mdagmUP(diracUP), 
    mdagmSloppyUP(diracSloppyUP), 
    mdagmPreUP(diracPreUP);
  DiracMdagM mdagmDOWN(diracDOWN), 
    mdagmSloppyDOWN(diracSloppyDOWN), 
    mdagmPreDOWN(diracPreDOWN);
  DiracMdagM mdagmSTRANGEPLUS(diracSTRANGEPLUS), 
    mdagmSloppySTRANGEPLUS(diracSloppySTRANGEPLUS), 
    mdagmPreSTRANGEPLUS(diracPreSTRANGEPLUS);
  DiracMdagM mdagmSTRANGEMINUS(diracSTRANGEMINUS), 
    mdagmSloppySTRANGEMINUS(diracSloppySTRANGEMINUS), 
    mdagmPreSTRANGEMINUS(diracPreSTRANGEMINUS);
 
  // Create SolverParams

  param->mu = param->mu_l;
  param->preconditioner = param->preconditionerUP;
  SolverParam solverParamU(*param);

  param->mu = -1.0 * param->mu_l;
  param->preconditioner = param->preconditionerDOWN;
  SolverParam solverParamD(*param);

  param->mu = param->mu_s;
  param->preconditioner = param->preconditionerSTRANGEPLUS;
  SolverParam solverParamSP(*param);

  param->mu = -1.0 * param->mu_s;
  param->preconditioner = param->preconditionerSTRANGEMINUS;
  SolverParam solverParamSM(*param);
  
  //======================================================================//
  //================ P R O B L E M   E X E C U T I O N  ==================// 
  //======================================================================//
 
  //We loop over all source positions and spin-colour componenets

  for(int isource = 0 ; isource < info.Nsources ; isource++){
    t5 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d begin now ###\n\n",
	       isource,
	       info.sourcePosition[isource][0],
	       info.sourcePosition[isource][1],
	       info.sourcePosition[isource][2],
	       info.sourcePosition[isource][3]);
    
    if( CorrFileFormat==ASCII_FORM ){
      sprintf(filename_mesons,"%s.mesons.SS.%02d.%02d.%02d.%02d.dat",
	      filename_twop,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);      
    }
    else if( CorrFileFormat==HDF5_FORM ){
      char *str;
      if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
      else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");
      sprintf(filename_mesons ,"%s_mesons_%s_SS.%02d.%02d.%02d.%02d.h5" ,
	      filename_twop,str,
	      info.sourcePosition[isource][0],
	      info.sourcePosition[isource][1],
	      info.sourcePosition[isource][2],
	      info.sourcePosition[isource][3]);
    }
    
    if(info.check_files){
      bool checkMesons;
      checkMesons = exists_file(filename_mesons);
      if( checkMesons == true ) continue;
    }
    
    printfQuda("Forward Inversions:\n");
    t1 = MPI_Wtime();
    for(int isc = 0 ; isc < 12 ; isc++){

      t4 = MPI_Wtime();

      ///////////////////////////////
      // Forward prop for up quark //
      ///////////////////////////////
      memset(input_vector,0,
	     X[0]*X[1]*X[2]*X[3]*
	     spinorSiteSize*sizeof(double));

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = (info.sourcePosition[isource][i] - 
		     comm_coords(default_topo)[i] * X[i]);
      
      if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	  (my_src[1]>=0) && (my_src[1]<X[1]) && 
	  (my_src[2]>=0) && (my_src[2]<X[2]) && 
	  (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + 
	   my_src[3]*X[2]*X[1]*X[0]*24 + 
	   my_src[2]*X[1]*X[0]*24 + 
	   my_src[1]*X[0]*24 + 
	   my_src[0]*24 + 
	   isc*2 ) = 1.0;
      
      //Ensure mu is up flavor
      param->mu = param->mu_l;
      mapNormalToEvenOdd(input_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
      tx1 = MPI_Wtime();
      performWuppertalnStep(output_vector, input_vector, param, GK_nsmearGauss, GK_alphaGauss);
      tx2 = MPI_Wtime();
      summ_tx12 += tx2-tx1;
      mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
      K_guess->packVector((double*) output_vector);
      K_guess->loadVector();
      K_guess->uploadToCuda(b,flag_eo);
      blas::zero(*x);
      diracUP.prepare(in,out,*x,*b,param->solution_type);

      printfQuda(" up - %02d: \n",isc);
      tx3 = MPI_Wtime();

      
      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet

      // Change mu to up flavor
      param->mu = param->mu_l;
      
      if(param->inv_type == QUDA_GCR_INVERTER){
	Solver *solveU = Solver::create(solverParamU, mUP, mSloppyUP, 
					mPreUP, profileInvert);

	(*solveU)(*out,*in);

	delete solveU;
      }
      else if(param->inv_type == QUDA_CG_INVERTER){
	cudaColorSpinorField tmp(*in);
	diracUP.Mdag(*in, tmp);

	Solver *solveU = Solver::create(solverParamU, mdagmUP, mdagmSloppyUP, 
					mdagmPreUP, profileInvert);
	(*solveU)(*out,*in);

	delete solveU;
      }

      tx4 = MPI_Wtime();
      summ_tx34 += tx4-tx3;
      solverParamU.updateInvertParam(*param);
      diracUP.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }
      
      K_temp->castDoubleToFloat(*K_vector);
      K_prop_up->absorbVectorToDevice(*K_temp,isc/3,isc%3);

      
      t2 = MPI_Wtime();
      printfQuda("Inversion up = %d,  for source = %d finished in time %f sec\n",
		 isc,isource,t2-t4);

      if(isc == 0) saveTuneCache();
      
      /////////////////////////////////
      // Forward prop for down quark //
      /////////////////////////////////
      memset(input_vector,0,
	     X[0]*X[1]*X[2]*X[3]*
	     spinorSiteSize*sizeof(double));

      for(int i = 0 ; i < 4 ; i++)
	my_src[i] = (info.sourcePosition[isource][i] - 
		     comm_coords(default_topo)[i] * X[i]);

      if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	  (my_src[1]>=0) && (my_src[1]<X[1]) && 
	  (my_src[2]>=0) && (my_src[2]<X[2]) && 
	  (my_src[3]>=0) && (my_src[3]<X[3]))
	*( (double*)input_vector + 
	   my_src[3]*X[2]*X[1]*X[0]*24 + 
	   my_src[2]*X[1]*X[0]*24 + 
	   my_src[1]*X[0]*24 + 
	   my_src[0]*24 + 
	   isc*2 ) = 1.0;

      K_guess->uploadToCuda(b,flag_eo);
      blas::zero(*x);
      diracDOWN.prepare(in,out,*x,*b,param->solution_type);
      printfQuda(" dn - %02d: \n",isc);
      tx3 = MPI_Wtime();

      // Change mu to down flavor
      param->mu = -1.0 * param->mu_l;
      if(param->inv_type == QUDA_GCR_INVERTER){
	Solver *solveD = Solver::create(solverParamD, mDOWN, mSloppyDOWN, 
					mPreDOWN, profileInvert);
	(*solveD)(*out,*in);

	delete solveD;
      }
      else if(param->inv_type == QUDA_CG_INVERTER){
	cudaColorSpinorField tmp(*in);
	diracDOWN.Mdag(*in, tmp);

	Solver *solveD = Solver::create(solverParamD, mdagmDOWN, mdagmSloppyDOWN, 
					mdagmPreDOWN, profileInvert);
	(*solveD)(*out,*in);

	delete solveD;
      }

      tx4 = MPI_Wtime();
      summ_tx34 += tx4-tx3;
      solverParamD.updateInvertParam(*param);
      diracDOWN.reconstruct(*x,*b,param->solution_type);
      K_vector->downloadFromCuda(x,flag_eo);
      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	K_vector->scaleVector(2*param->kappa);
      }

      K_temp->castDoubleToFloat(*K_vector);
      K_prop_down->absorbVectorToDevice(*K_temp,isc/3,isc%3);

      t4 = MPI_Wtime();
      printfQuda("Inversion down = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);

      if( MESON == KAON || MESON == ALL_MESONS ) {

	/////////////////////////////////////////////
	// Forward prop for positive strange quark //
	/////////////////////////////////////////////
	memset(input_vector,0,
	       X[0]*X[1]*X[2]*X[3]*
	       spinorSiteSize*sizeof(double));

	for(int i = 0 ; i < 4 ; i++)
	  my_src[i] = (info.sourcePosition[isource][i] - 
		       comm_coords(default_topo)[i] * X[i]);

	if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	    (my_src[1]>=0) && (my_src[1]<X[1]) && 
	    (my_src[2]>=0) && (my_src[2]<X[2]) && 
	    (my_src[3]>=0) && (my_src[3]<X[3]))
	  *( (double*)input_vector + 
	     my_src[3]*X[2]*X[1]*X[0]*24 + 
	     my_src[2]*X[1]*X[0]*24 + 
	     my_src[1]*X[0]*24 + 
	     my_src[0]*24 + 
	     isc*2 ) = 1.0;

	K_guess->uploadToCuda(b,flag_eo);
	blas::zero(*x);
	diracSTRANGEPLUS.prepare(in,out,*x,*b,param->solution_type);
	printfQuda(" strange plus - %02d: \n",isc);
	tx3 = MPI_Wtime();

	// Change mu to strange flavor
	param->mu = param->mu_s;
	if(param->inv_type == QUDA_GCR_INVERTER){
	  Solver *solveSP = Solver::create(solverParamSP, mSTRANGEPLUS, mSloppySTRANGEPLUS, 
					   mPreSTRANGEPLUS, profileInvert);
	  (*solveSP)(*out,*in);

	  delete solveSP;
	}
	else if(param->inv_type == QUDA_CG_INVERTER){
	  cudaColorSpinorField tmp(*in);
	  diracSTRANGEPLUS.Mdag(*in, tmp);

	  Solver *solveSP = Solver::create(solverParamSP, mdagmSTRANGEPLUS, mdagmSloppySTRANGEPLUS, 
					   mdagmPreSTRANGEPLUS, profileInvert);
	  (*solveSP)(*out,*in);

	  delete solveSP;
	}

	tx4 = MPI_Wtime();
	summ_tx34 += tx4-tx3;
	solverParamSP.updateInvertParam(*param);
	diracSTRANGEPLUS.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	    param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}

	K_temp->castDoubleToFloat(*K_vector);
	K_prop_strangePlus->absorbVectorToDevice(*K_temp,isc/3,isc%3);

	t4 = MPI_Wtime();
	printfQuda("Inversion strange plus = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);

	tx4 = MPI_Wtime();
	summ_tx34 += tx4-tx3;
	solverParamSP.updateInvertParam(*param);
	diracSTRANGEPLUS.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	    param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}

	K_temp->castDoubleToFloat(*K_vector);
	K_prop_strangePlus->absorbVectorToDevice(*K_temp,isc/3,isc%3);

	t4 = MPI_Wtime();
	printfQuda("Inversion strange plus = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);

	/////////////////////////////////////////////
	// Forward prop for negative strange quark //
	/////////////////////////////////////////////
	memset(input_vector,0,
	       X[0]*X[1]*X[2]*X[3]*
	       spinorSiteSize*sizeof(double));

	for(int i = 0 ; i < 4 ; i++)
	  my_src[i] = (info.sourcePosition[isource][i] - 
		       comm_coords(default_topo)[i] * X[i]);

	if( (my_src[0]>=0) && (my_src[0]<X[0]) && 
	    (my_src[1]>=0) && (my_src[1]<X[1]) && 
	    (my_src[2]>=0) && (my_src[2]<X[2]) && 
	    (my_src[3]>=0) && (my_src[3]<X[3]))
	  *( (double*)input_vector + 
	     my_src[3]*X[2]*X[1]*X[0]*24 + 
	     my_src[2]*X[1]*X[0]*24 + 
	     my_src[1]*X[0]*24 + 
	     my_src[0]*24 + 
	     isc*2 ) = 1.0;

	K_guess->uploadToCuda(b,flag_eo);
	blas::zero(*x);
	diracSTRANGEMINUS.prepare(in,out,*x,*b,param->solution_type);
	printfQuda(" strange minus - %02d: \n",isc);
	tx3 = MPI_Wtime();

	// Change mu to strange flavor
	param->mu = param->mu_s;
	if(param->inv_type == QUDA_GCR_INVERTER){
	  Solver *solveSM = Solver::create(solverParamSM, mSTRANGEMINUS, mSloppySTRANGEMINUS, 
					   mPreSTRANGEMINUS, profileInvert);
	  (*solveSM)(*out,*in);

	  delete solveSM;
	}
	else if(param->inv_type == QUDA_CG_INVERTER){
	  cudaColorSpinorField tmp(*in);
	  diracSTRANGEMINUS.Mdag(*in, tmp);

	  Solver *solveSM = Solver::create(solverParamSM, mdagmSTRANGEMINUS, mdagmSloppySTRANGEMINUS, 
					   mdagmPreSTRANGEMINUS, profileInvert);
	  (*solveSM)(*out,*in);

	  delete solveSM;
	}

	tx4 = MPI_Wtime();
	summ_tx34 += tx4-tx3;
	solverParamSM.updateInvertParam(*param);
	diracSTRANGEMINUS.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	    param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}

	K_temp->castDoubleToFloat(*K_vector);
	K_prop_strangeMinus->absorbVectorToDevice(*K_temp,isc/3,isc%3);

	t4 = MPI_Wtime();
	printfQuda("Inversion strange minus = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);

	tx4 = MPI_Wtime();
	summ_tx34 += tx4-tx3;
	solverParamSM.updateInvertParam(*param);
	diracSTRANGEMINUS.reconstruct(*x,*b,param->solution_type);
	K_vector->downloadFromCuda(x,flag_eo);
	if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
	    param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	  K_vector->scaleVector(2*param->kappa);
	}

	K_temp->castDoubleToFloat(*K_vector);
	K_prop_strangeMinus->absorbVectorToDevice(*K_temp,isc/3,isc%3);

	t4 = MPI_Wtime();
	printfQuda("Inversion strange minus = %d,  for source = %d finished in time %f sec\n",isc,isource,t4-t2);
      }// End if kaon
    } 
    // Close loop over 12 spin-color
    
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Gaussian Smearing (For just the source point, all spin-color components): %f sec. \n",summ_tx12);
    summ_tx12=0.;
    printfQuda("TIME_REPORT - Just Inversions (24 in total): %f sec. \n", summ_tx34);
    summ_tx34=0.;
    printfQuda("TIME_REPORT - Forward Inversions (Total including smearing, inversions and others): %f sec.\n\n",t2-t1);
    
    ////////////////////////////////////
    // Smearing on the 3D propagators //
    ////////////////////////////////////
    
    if(info.run3pt_src[isource]){
      
      //-C.K: Loop over the number of sink-source separations
      int my_fixSinkTime;
      char *filename_threep_base;
      for(int its=0;its<info.Ntsink;its++){
	my_fixSinkTime = 
	  (info.tsinkSource[its] + info.sourcePosition[isource][3])%GK_totalL[3] 
	  - comm_coords(default_topo)[3] * X[3];
	
	t1 = MPI_Wtime();
	K_temp->zero_device();
	checkCudaError();
	if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ){
	  K_prop3D_up->absorbTimeSlice(*K_prop_up,my_fixSinkTime);
	  K_prop3D_down->absorbTimeSlice(*K_prop_down,my_fixSinkTime);
	  if( MESON == KAON || MESON == ALL_MESONS ) {
	    K_prop3D_strangePlus->absorbTimeSlice(*K_prop_strangePlus,my_fixSinkTime);
	    K_prop3D_strangeMinus->absorbTimeSlice(*K_prop_strangeMinus,my_fixSinkTime);
	  }
	}
	comm_barrier();

	for(int nu = 0 ; nu < 4 ; nu++)
	  for(int c2 = 0 ; c2 < 3 ; c2++){
	    ////////
	    // up //
	    ////////
	    K_temp->zero_device();
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_temp->copyPropagator3D(*K_prop3D_up,
				       my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_prop3D_up->absorbVectorTimeSlice(*K_temp,
						 my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_temp->zero_device();
	    //////////
	    // down //
	    //////////
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_temp->copyPropagator3D(*K_prop3D_down,
				       my_fixSinkTime,nu,c2);
	    comm_barrier();

	    K_vector->castFloatToDouble(*K_temp);
	    K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	    K_temp->castDoubleToFloat(*K_guess);
	    if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
	      K_prop3D_down->absorbVectorTimeSlice(*K_temp,
						   my_fixSinkTime,nu,c2);
	    comm_barrier();
	    
	    K_temp->zero_device();	
	    
	    if( MESON == KAON || MESON == ALL_MESONS ) {
	      //////////////////
	      // strange plus //
	      //////////////////
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_temp->copyPropagator3D(*K_prop3D_strangePlus,
					 my_fixSinkTime,nu,c2);
	      comm_barrier();

	      K_vector->castFloatToDouble(*K_temp);
	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_temp->castDoubleToFloat(*K_guess);
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_prop3D_strangePlus->absorbVectorTimeSlice(*K_temp,
							    my_fixSinkTime,nu,c2);
	      comm_barrier();
	    
	      K_temp->zero_device();	

	      ///////////////////
	      // strange minus //
	      ///////////////////
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_temp->copyPropagator3D(*K_prop3D_strangeMinus,
					 my_fixSinkTime,nu,c2);
	      comm_barrier();

	      K_vector->castFloatToDouble(*K_temp);
	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_temp->castDoubleToFloat(*K_guess);
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_prop3D_strangeMinus->absorbVectorTimeSlice(*K_temp,
							     my_fixSinkTime,nu,c2);
	      comm_barrier();
	    
	      K_temp->zero_device();	
	    } // End if Kaon
	  }

	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT - 3d Props preparation for sink-source[%d]=%d: %f sec\n",
		   its,info.tsinkSource[its],t2-t1);
	
	printfQuda("\n# Three-point function calculation for source-position = %d, sink-source = %d begins now\n",
		   isource,info.tsinkSource[its]);
	  
	if( CorrFileFormat==ASCII_FORM ){
	  asprintf(&filename_threep_base,"%s_tsink%d",
		   filename_threep,info.tsinkSource[its]);
	  printfQuda("The three-point function ASCII base name is: %s\n",
		     filename_threep_base);
	}

	if( MESON == PION || MESON == ALL_MESONS ) {

	  /////////////////////////////////////
	  // Sequential propagator for Pion+ //
	  /////////////////////////////////////

	  printfQuda("Sequential Inversions, %s:\n", "Pion+");
	
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();


	      //Ensure mu is down flavor:
	      param->mu *= -1.0 * param->mu_l;		
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		//-CJL: We use the up propagator for the pion+ so that we get back
		// anti-down after we conjugate during the contractions
		K_temp->copyPropagator3D(*K_prop3D_up,my_fixSinkTime,nu,c2);
	    
	      comm_barrier();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      //Scale up vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_guess->uploadToCuda(b,flag_eo);

	  
	      diracDOWN.prepare(in,out,*x,*b,param->solution_type);
	      
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      K_guess->uploadToCuda(out,flag_eo); 
	      // initial guess is ready
	      
	      printfQuda("%02d - \n",nu*3+c2);

	      if(param->inv_type == QUDA_GCR_INVERTER){
		Solver *solveD = Solver::create(solverParamD, mDOWN, mSloppyDOWN, 
						mPreDOWN, profileInvert);
		(*solveD)(*out,*in);

		delete solveD;
	      }
	      else if(param->inv_type == QUDA_CG_INVERTER){
		cudaColorSpinorField tmp(*in);
		diracDOWN.Mdag(*in, tmp);

		Solver *solveD = Solver::create(solverParamD, mdagmDOWN, mdagmSloppyDOWN, 
						mdagmPreDOWN, profileInvert);
		(*solveD)(*out,*in);

		delete solveD;
	      }

	      solverParamD.updateInvertParam(*param);
	      diracDOWN.reconstruct(*x,*b,param->solution_type);

	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }
	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	      
	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for %s seq prop = %d, source = %d at sink-source = %d, is: %f sec\n",
			 "pion+",nu*3+c2,isource,info.tsinkSource[its],t4-t3);
      
	    }

	  t2 = MPI_Wtime();

	  printfQuda("TIME_REPORT - Sequential Inversions, particle: %s, %f sec\n",
		     "pion+",t2-t1);

		  
	  ////////////////////////////
	  // Contractions for Pion+ //
	  ////////////////////////////


	  t1 = MPI_Wtime();

	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, 
				      *K_gaugeContractions,
				      corrThp_local, corrThp_noether, 
				      corrThp_oneD, MESON, 1, 
				      isource, CorrSpace);
	
	  t2 = MPI_Wtime();

	  printfQuda("TIME_REPORT - Three-point Contractions, %s: %f sec\n",
		     "pion+",t2-t1);
		  
	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, PION, 1, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();

	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],
		       "pion+",t2-t1);

	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd = 0;
	    int thrp_sign = 1;
	  
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_pion_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, 0, 
					   thrp_sign, THRP_LOCAL, CorrSpace,
					   HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_pion_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, 0, 
					   thrp_sign,THRP_NOETHER,CorrSpace,
					   HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_pion_HDF5[mu],
					     (void*)corrThp_oneD, mu, 
					     uOrd, its, info.Ntsink, 0, 
					     thrp_sign, THRP_ONED, CorrSpace,
					     HighMomForm);
	    t2 = MPI_Wtime();
	  
	    printfQuda("TIME_REPORT - 3-point function for %s copied to HDF5 write buffers in %f sec.\n","pion+",t2-t1);

	  }
	      
	  /////////////////////////////////////
	  // Sequential propagator for Pion- //
	  /////////////////////////////////////

	  printfQuda("Sequential Inversions, %s:\n",
		     "pion-");

	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();
	      //Ensure mu is up flavor
	      param->mu *= param->mu_l;
	      if( ( my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	      // We use the down propogator for the pion- so that we get back
	      // anti-up after we conjugate during the contractions

	      comm_barrier();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      // Scale vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_guess->uploadToCuda(b,flag_eo);
	      
	      diracUP.prepare(in,out,*x,*b,param->solution_type);
	    	
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      K_guess->uploadToCuda(out,flag_eo); 
	      // initial guess is ready
		
	      printfQuda("%02d - ",nu*3+c2);

	      if(param->inv_type == QUDA_GCR_INVERTER){
		Solver *solveU = Solver::create(solverParamU, mUP, mSloppyUP, 
						mPreUP, profileInvert);

		(*solveU)(*out,*in);

		delete solveU;
	      }
	      else if(param->inv_type == QUDA_CG_INVERTER){
		cudaColorSpinorField tmp(*in);
		diracUP.Mdag(*in, tmp);

		Solver *solveU = Solver::create(solverParamU, mdagmUP, mdagmSloppyUP, 
						mdagmPreUP, profileInvert);
		(*solveU)(*out,*in);

		delete solveU;
	      }

	      solverParamU.updateInvertParam(*param);
	      diracUP.reconstruct(*x,*b,param->solution_type);

	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }

	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	  
	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for %s seq prop = %d, source = %d at sink-source = %d, is: %f sec\n",
			 "pion-",nu*3+c2,isource,info.tsinkSource[its],t4-t3);
	    }

	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, particle %s: %f sec\n","pion-",t2-t1);
	
	  ////////////////////////////
	  // Contractions for Pion- //
	  ////////////////////////////

	  t1 = MPI_Wtime();
	
	  K_contract->contractFixSink(*K_seqProp, 
				      *K_prop_down, 
				      *K_gaugeContractions,
				      corrThp_local, corrThp_noether, 
				      corrThp_oneD, MESON, 2, 
				      isource, CorrSpace);

	  t2 = MPI_Wtime();
	
	  printfQuda("TIME_REPORT - Three-point Contractions, %s: %f sec\n", "pion-",t2-t1);

	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, PION, 2, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();

	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],"pion-",t2-t1);

	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd = 1;
	    
	    int thrp_sign = 1;

	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_pion_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, 
					   0, thrp_sign, THRP_LOCAL, 
					   CorrSpace, HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_pion_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, 
					   0, thrp_sign, THRP_NOETHER, 
					   CorrSpace, HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_pion_HDF5[mu], 
					     (void*)corrThp_oneD,mu, 
					     uOrd, its, info.Ntsink, 
					     0, thrp_sign, THRP_ONED, 
					     CorrSpace, HighMomForm);
	    t2 = MPI_Wtime();

	    printfQuda("TIME_REPORT - 3-point function for %s copied to HDF5 write buffers in %f sec.\n", 
		       "pion-",t2-t1);

	  }
	} // End if pion

	if( MESON == KAON || MESON == ALL_MESONS ) {

	  /////////////////////////////////////
	  // Sequential propagator for Kaon+ //
	  /////////////////////////////////////


	  printfQuda("Sequential Inversions, %s:\n", "Kaon+");
	
	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();

	      //Ensure mu is strange minus:
	      param->mu *= -1.0 * param->mu_s;		
	      if( (my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		//-CJL: We use the positive strange propagator for the kaon+ so that we get back
		// negative strange after we conjugate during the contractions
		K_temp->copyPropagator3D(*K_prop3D_strangePlus,my_fixSinkTime,nu,c2);
	    
	      comm_barrier();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      //Scale up vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_guess->uploadToCuda(b,flag_eo);

	  
	      diracDOWN.prepare(in,out,*x,*b,param->solution_type);
	      
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      K_guess->uploadToCuda(out,flag_eo); 
	      // initial guess is ready
	      
	      printfQuda("%02d - \n",nu*3+c2);

	      if(param->inv_type == QUDA_GCR_INVERTER){
		Solver *solveD = Solver::create(solverParamD, mDOWN, mSloppyDOWN, 
						mPreDOWN, profileInvert);
		(*solveD)(*out,*in);

		delete solveD;
	      }
	      else if(param->inv_type == QUDA_CG_INVERTER){
		cudaColorSpinorField tmp(*in);
		diracDOWN.Mdag(*in, tmp);

		Solver *solveD = Solver::create(solverParamD, mdagmDOWN, mdagmSloppyDOWN, 
						mdagmPreDOWN, profileInvert);
		(*solveD)(*out,*in);

		delete solveD;
	      }

	      solverParamD.updateInvertParam(*param);
	      diracDOWN.reconstruct(*x,*b,param->solution_type);

	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }
	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);
	      
	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for %s seq prop = %d, source = %d at sink-source = %d, is: %f sec\n",
			 "kaon+",nu*3+c2,isource,info.tsinkSource[its],t4-t3);
      
	    }

	  t2 = MPI_Wtime();

	  printfQuda("TIME_REPORT - Sequential Inversions, %s: %f sec\n",
		     "kaon+",t2-t1);

		  
	  ////////////////////////////
	  // Contractions for Kaon+ //
	  ////////////////////////////


	  t1 = MPI_Wtime();

	  K_contract->contractFixSink(*K_seqProp, *K_prop_up, 
				      *K_gaugeContractions,
				      corrThp_local, corrThp_noether, 
				      corrThp_oneD, MESON, 1, 
				      isource, CorrSpace);
	
	  t2 = MPI_Wtime();

	  printfQuda("TIME_REPORT - Three-point Contractions, %s: %f sec\n",
		     "kaon+",t2-t1);
		  
	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, KAON, 1, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();

	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],
		       "kaon+",t2-t1);

	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd = 0;
	    
	    int thrp_sign = 1;
	  
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_kaon_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, 0, 
					   thrp_sign, THRP_LOCAL, CorrSpace,
					   HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_kaon_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, 0, 
					   thrp_sign,THRP_NOETHER,CorrSpace,
					   HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_kaon_HDF5[mu],
					     (void*)corrThp_oneD, mu, 
					     uOrd, its, info.Ntsink, 0, 
					     thrp_sign, THRP_ONED, CorrSpace,
					     HighMomForm);
	    t2 = MPI_Wtime();
	  
	    printfQuda("TIME_REPORT - 3-point function for %s copied to HDF5 write buffers in %f sec.\n","kaon+",t2-t1);

	  }
	  
	  /////////////////////////////////////
	  // Sequential propagator for kaon- //
	  /////////////////////////////////////

	  printfQuda("Sequential Inversions, %s:\n",
		     "kaon-");

	  t1 = MPI_Wtime();
	  for(int nu = 0 ; nu < 4 ; nu++)
	    for(int c2 = 0 ; c2 < 3 ; c2++){
	      t3 = MPI_Wtime();
	      K_temp->zero_device();
	      //Ensure mu is up flavor
	      param->mu *= param->mu_l;
	      if( ( my_fixSinkTime >= 0) && ( my_fixSinkTime < X[3] ) ) 
		K_temp->copyPropagator3D(*K_prop3D_down,my_fixSinkTime,nu,c2);
	      // We use the down propogator for the kaon- so that we get back
	      // anti-up after we conjugate during the contractions

	      comm_barrier();
	      K_temp->apply_gamma5();
	      K_vector->castFloatToDouble(*K_temp);

	      // Scale vector to avoid MP errors
	      K_vector->scaleVector(1e+10);

	      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	      K_guess->uploadToCuda(b,flag_eo);
	      
	      diracUP.prepare(in,out,*x,*b,param->solution_type);
	    	
	      K_vector->downloadFromCuda(in,flag_eo);
	      K_vector->download();
	      K_guess->uploadToCuda(out,flag_eo); 
	      // initial guess is ready
		
	      printfQuda("%02d - ",nu*3+c2);

	      if(param->inv_type == QUDA_GCR_INVERTER){
		Solver *solveSP = Solver::create(solverParamSP, mSTRANGEPLUS, mSloppySTRANGEPLUS, 
						 mPreSTRANGEPLUS, profileInvert);

		(*solveSP)(*out,*in);

		delete solveSP;
	      }
	      else if(param->inv_type == QUDA_CG_INVERTER){
		cudaColorSpinorField tmp(*in);
		diracSTRANGEPLUS.Mdag(*in, tmp);

		Solver *solveSP = Solver::create(solverParamSP, mdagmSTRANGEPLUS, mdagmSloppySTRANGEPLUS, 
						 mdagmPreSTRANGEPLUS, profileInvert);
		(*solveSP)(*out,*in);

		delete solveSP;
	      }

	      solverParamSP.updateInvertParam(*param);
	      diracSTRANGEPLUS.reconstruct(*x,*b,param->solution_type);

	      K_vector->downloadFromCuda(x,flag_eo);
	      if (param->mass_normalization == QUDA_MASS_NORMALIZATION || 
		  param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
		K_vector->scaleVector(2*param->kappa);
	      }

	      // Rescale to normal
	      K_vector->scaleVector(1e-10);
	      
	      K_temp->castDoubleToFloat(*K_vector);
	      K_seqProp->absorbVectorToDevice(*K_temp,nu,c2);

	      t4 = MPI_Wtime();
	      
	      printfQuda("Inversion time for %s seq prop = %d, source = %d at sink-source = %d, is: %f sec\n",
			 "kaon-",nu*3+c2,isource,info.tsinkSource[its],t4-t3);
	    }
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Sequential Inversions, %s: %f sec\n","pion-",t2-t1);
	
	  ////////////////////////////
	  // Contractions for Kaon- //
	  ////////////////////////////

	  t1 = MPI_Wtime();
	
	  K_contract->contractFixSink(*K_seqProp, 
				      *K_prop_strangeMinus, 
				      *K_gaugeContractions,
				      corrThp_local, corrThp_noether, 
				      corrThp_oneD, MESON, 2, 
				      isource, CorrSpace);

	  t2 = MPI_Wtime();
	
	  printfQuda("TIME_REPORT - Three-point Contractions, partilce %s: %f sec\n", "kaon-",t2-t1);

	  t1 = MPI_Wtime();
	  if( CorrFileFormat==ASCII_FORM ){
	    K_contract->writeThrp_ASCII(corrThp_local, corrThp_noether, 
					corrThp_oneD, KAON, 2, 
					filename_threep_base, isource, 
					info.tsinkSource[its], CorrSpace);
	    t2 = MPI_Wtime();

	    printfQuda("TIME_REPORT - Done: 3-pt function for sp = %d, sink-source = %d, %s written ASCII format in %f sec.\n",
		       isource,info.tsinkSource[its],"kaon-",t2-t1);

	  }
	  else if( CorrFileFormat==HDF5_FORM ){
	    int uOrd = 1;
	    int thrp_sign = 1;

	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_local_kaon_HDF5, 
					   (void*)corrThp_local, 0, 
					   uOrd, its, info.Ntsink, 
					   0, thrp_sign, THRP_LOCAL, 
					   CorrSpace, HighMomForm);
	    K_contract->copyThrpToHDF5_Buf((void*)Thrp_noether_kaon_HDF5, 
					   (void*)corrThp_noether, 0, 
					   uOrd, its, info.Ntsink, 
					   0, thrp_sign, THRP_NOETHER, 
					   CorrSpace, HighMomForm);
	    for(int mu = 0;mu<4;mu++)
	      K_contract->copyThrpToHDF5_Buf((void*)Thrp_oneD_kaon_HDF5[mu], 
					     (void*)corrThp_oneD,mu, 
					     uOrd, its, info.Ntsink, 
					     0, thrp_sign, THRP_ONED, 
					     CorrSpace, HighMomForm);
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT - 3-point function for %s copied to HDF5 write buffers in %f sec.\n", 
		       "kaon-",t2-t1);

	  }
	} // End if kaon
      } // End loop over sink-source separations      
      
      //-C.K. Write the three-point function in HDF5 format
      if( CorrFileFormat==HDF5_FORM ){

	if( MESON == PION || MESON == ALL_MESONS ) {

	  char *str;
	  if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
	  else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");

	  t1 = MPI_Wtime();

	  asprintf(&filename_threep_base,"%s_%s_%s_SS.%02d.%02d.%02d.%02d.h5",
		   filename_threep, 
		   "pion", str, 
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3]);

	  printfQuda("\nThe three-point function HDF5 filename is: %s\n",
		     filename_threep_base);
	
	  K_contract->writeThrpHDF5((void*) Thrp_local_pion_HDF5, 
				    (void*) Thrp_noether_pion_HDF5, 
				    (void**)Thrp_oneD_pion_HDF5, 
				    filename_threep_base, info, 
				    isource, PION);
	
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d written in HDF5 format in %f sec.\n",isource,t2-t1);
	} // End if pion
	if( MESON == KAON || MESON == ALL_MESONS ) {

	  char *str;
	  if(CorrSpace==MOMENTUM_SPACE) asprintf(&str,"Qsq%d",info.Q_sq);
	  else if (CorrSpace==POSITION_SPACE) asprintf(&str,"PosSpace");

	  t1 = MPI_Wtime();

	  asprintf(&filename_threep_base,"%s_%s_%s_SS.%02d.%02d.%02d.%02d.h5",
		   filename_threep, 
		   "kaon", str, 
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3]);

	  printfQuda("\nThe three-point function HDF5 filename is: %s\n",
		     filename_threep_base);
	
	  K_contract->writeThrpHDF5((void*) Thrp_local_pion_HDF5, 
				    (void*) Thrp_noether_pion_HDF5, 
				    (void**)Thrp_oneD_pion_HDF5, 
				    filename_threep_base, info, 
				    isource, KAON);
	
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT - Done: Three-point function for source-position = %d written in HDF5 format in %f sec.\n",isource,t2-t1);
	} // End if kaon
      }
      printfQuda("\n");
    }
    // End loop if running for the specific isource
    
    ///////////////////////////////////
    // Smear the forward propagators //
    ///////////////////////////////////

    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	K_temp->copyPropagator(*K_prop_up,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_vector->download();
	mapNormalToEvenOdd((void*) K_vector->H_elem() , *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	performWuppertalnStep(output_vector, (void*) K_vector->H_elem(), param, GK_nsmearGauss, GK_alphaGauss);
	mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	K_guess->packVector((double*) output_vector);
	K_guess->loadVector();
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_vector->download();
	mapNormalToEvenOdd((void*) K_vector->H_elem() , *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	performWuppertalnStep(output_vector, (void*) K_vector->H_elem(), param, GK_nsmearGauss, GK_alphaGauss);
	mapEvenOddToNormal(output_vector, *param, GK_localL[0], GK_localL[1], GK_localL[2], GK_localL[3]);
	K_guess->packVector((double*) output_vector);
	K_guess->loadVector();
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_down->absorbVectorToDevice(*K_temp,nu,c2);
      }
    
    K_prop_up->rotateToPhysicalBase_device(+1);
    K_prop_down->rotateToPhysicalBase_device(-1);
    t1 = MPI_Wtime();
    K_contract->contractMesons (*K_prop_up,*K_prop_down, corrMesons, 
				isource, CorrSpace);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT - Two-point Contractions: %f sec\n",t2-t1);
    
    //======================================================================//
    //===================== W R I T E   D A T A  ===========================//
    //======================================================================//

    printfQuda("The mesons two-point function %s filename is: %s\n" ,
	       (CorrFileFormat==ASCII_FORM) ? "ASCII" : "HDF5",
	       filename_mesons);
    
    if( CorrFileFormat==ASCII_FORM ){
      t1 = MPI_Wtime();
      K_contract->writeTwopMesons_ASCII (corrMesons , filename_mesons , 
					 isource, CorrSpace);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Mesons for source-position = %d written in ASCII format in %f sec.\n",
		 isource,t2-t1);
    }    
    else if( CorrFileFormat==HDF5_FORM ){
      t1 = MPI_Wtime();
      K_contract->copyTwopMesonsToHDF5_Buf ((void*)Twop_mesons_HDF5 , 
					    (void*)corrMesons, CorrSpace,
					    HighMomForm);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Two-point function for mesons copied to HDF5 write buffers in %f sec.\n",t2-t1);
      
      t1 = MPI_Wtime();
      K_contract->writeTwopMesonsHDF5 ((void*) Twop_mesons_HDF5, 
				       filename_mesons , info, isource);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT - Done: Two-point function for Mesons for source-position = %d written in HDF5 format in %f sec.\n",
		 isource,t2-t1);
    }
    
    t6 = MPI_Wtime();
    printfQuda("\n ### Calculations for source-position %d - %02d.%02d.%02d.%02d Completed in %f sec. ###\n",isource, 
	       info.sourcePosition[isource][0],
	       info.sourcePosition[isource][1],
	       info.sourcePosition[isource][2],
	       info.sourcePosition[isource][3],
	       t6-t5);
  }
  // close loop over source positions

  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//  
  
  printfQuda("\nCleaning up...\n");
  free(corrThp_local);
  free(corrThp_noether);
  free(corrThp_oneD);
  free(corrMesons);

  if( CorrFileFormat==HDF5_FORM ){
    free(Thrp_local_pion_HDF5);
    free(Thrp_local_kaon_HDF5);
    free(Thrp_noether_pion_HDF5);
    free(Thrp_noether_kaon_HDF5);
    for(int mu=0;mu<4;mu++) {
      free(Thrp_oneD_pion_HDF5[mu]);
      free(Thrp_oneD_kaon_HDF5[mu]);    
    }
    free(Thrp_oneD_pion_HDF5);
    free(Thrp_oneD_kaon_HDF5);
    free(Twop_mesons_HDF5);
  }

  free(input_vector);
  free(output_vector);

  delete K_temp;
  delete K_contract;
  delete K_prop_up;
  delete K_prop_down;
  delete K_prop_strangePlus;
  delete K_prop_strangeMinus;
  delete dUP;
  delete dSloppyUP;
  delete dPreUP;
  delete dDOWN;
  delete dSloppyDOWN;
  delete dPreDOWN;
  delete dSTRANGEPLUS;
  delete dSTRANGEMINUS;
  delete dSloppySTRANGEPLUS;
  delete dSloppySTRANGEMINUS;
  delete dPreSTRANGEPLUS;
  delete dPreSTRANGEMINUS;
  delete K_guess;
  delete K_vector;
  delete K_gaugeSmeared;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete K_gaugeContractions;
  delete K_seqProp;
  delete K_prop3D_up;
  delete K_prop3D_down;
  delete K_prop3D_strangePlus;
  delete K_prop3D_strangeMinus;

  printfQuda("...Done\n");
  
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);

}

#endif

////////////////////////////////
// QKXTM Eigenvector routines //
////////////////////////////////

#ifdef HAVE_ARPACK


void calcLowModeProjection(QudaInvertParam *evInvParam, 
			   qudaQKXTM_arpackInfo arpackInfo){
  
  double t1,t2;
  char fname[256];
  sprintf(fname, "calcLowModeProjection");
  
  //======================================================================//
  //================= P A R A M E T E R   C H E C K S ====================//
  //======================================================================//

  if (!initialized) 
    errorQuda("%s: QUDA not initialized", fname);
  pushVerbosity(evInvParam->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(evInvParam);
  
  //-Checks for exact deflation part 
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if( (evInvParam->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && 
      (evInvParam->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) 
    errorQuda("Only asymmetric operators are supported in deflation\n");
  if( arpackInfo.isEven    && 
      (evInvParam->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ) 
    errorQuda("%s: Inconsistency between operator types!",fname);
  if( (!arpackInfo.isEven) && 
      (evInvParam->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) )   
    errorQuda("%s: Inconsistency between operator types!",fname);
  
  // QKXTM: DMH Here the low modes of the normal M^{\dagger}M 
  //        operator are constructed. We then apply gamma_5 to
  //        each eigenvector V, and then take the inner product:
  //        g5Prod_i = V^{\dagger}_i g5 V_i

  int NeV_Full = arpackInfo.nEv;  
  
  //Create object to store and calculate eigenpairs
  QKXTM_Deflation<double> *deflation = 
    new QKXTM_Deflation<double>(evInvParam,arpackInfo);
  deflation->printInfo();
  
  //- Calculate the eigenVectors
  t1 = MPI_Wtime(); 
  deflation->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("%s TIME REPORT:",fname);
  printfQuda("Full Operator EigenVector Calculation: %f sec\n",t2-t1);

  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//

  printfQuda("\nCleaning up...\n");
  
  delete deflation;

  printfQuda("...Done\n");
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}



//========================================================================//
//========= Function which calculates loops using exact Deflation=========//
//========================================================================//

void calcMG_loop_wOneD_wExact(void **gaugeToPlaquette, 
			      QudaInvertParam *EvInvParam, 
			      QudaInvertParam *param, 
			      QudaGaugeParam *gauge_param,
			      qudaQKXTM_arpackInfo arpackInfo, 
			      qudaQKXTM_loopInfo loopInfo, 
			      qudaQKXTMinfo info){

  double t1,t2,t3,t4,t5;
  char fname[256];
  sprintf(fname, "calcMG_loop_wOneD_wExact");
  
  //======================================================================//
  //================= P A R A M E T E R   C H E C K S ====================//
  //======================================================================//


  // parameters related to Probing and spinColorDil
  unsigned short int* Vc = NULL;
  int k_probing = loopInfo.k_probing;
  bool spinColorDil = loopInfo.spinColorDil;
  bool isProbing;
  bool isProbingMstep; // if this is enabled then probing will be done in steps

  int Nc; // number of Hadamard vectors, if not enabled then it is one
  int Nc_low; // from where to start doing hadamard vectors
  int Nc_high; // where to stop doing hadamard vectors
  int Nsc; // number of Spin-Color diluted vectors, if not enabled then it is one

  if(k_probing > 0){
    Nc = 2*pow(2,4*(k_probing-1));
    Vc = hch_coloring(k_probing,4); //4D hierarchical coloring
    isProbing=true;

    if(loopInfo.hadamLow < 0 || loopInfo.hadamHigh < 0) errorQuda("Error: You cannot give negative values for hadamLow or hadamHigh\n");
    if(loopInfo.hadamLow > loopInfo.hadamHigh) errorQuda("Error: hadamLow cannot be greater than hadamHigh\n");
    Nc_low = loopInfo.hadamLow;
    if(loopInfo.hadamHigh == 0) Nc_high = Nc;
    else Nc_high = loopInfo.hadamHigh;
    if(Nc_high > Nc) errorQuda("Error: You cannot choose hadamHigh to be greater than Nc\n");
    if(Nc_low > 0 || Nc_high < Nc) isProbingMstep=true;
  }
  else{
    Nc=1;
    Nc_low=0;
    Nc_high=Nc;
    isProbing=false;
    isProbingMstep=false;
  }

  if(spinColorDil)
    Nsc=12;
  else
    Nsc=1;


  if (!initialized) 
    errorQuda("%s: QUDA not initialized", fname);
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  
  printfQuda("\n### %s: Loop calculation begins now\n\n",fname);

  //-Checks for exact deflation part 
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  if( (EvInvParam->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) && 
      (EvInvParam->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) ) 
    errorQuda("Only asymmetric operators are supported in deflation\n");
  if( arpackInfo.isEven    && 
      (EvInvParam->matpc_type != QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ) 
    errorQuda("%s: Inconsistency between operator types!",fname);
  if( (!arpackInfo.isEven) && 
      (EvInvParam->matpc_type != QUDA_MATPC_ODD_ODD_ASYMMETRIC) )   
    errorQuda("%s: Inconsistency between operator types!",fname);

  //-Checks for stochastic approximation and generalities 
  if(param->inv_type != QUDA_GCR_INVERTER) 
    errorQuda("%s: This function works only with GCR method", fname);  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("%s: This function works only with ukqcd gamma basis\n",fname);
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("%s: This function works only with color-inside-spin\n",fname);
  
  //Stochastic, momentum, and data dump information.
  int Nstoch = loopInfo.Nstoch;
  unsigned long int seed = loopInfo.seed;
  int Ndump = loopInfo.Ndump;
  int Nprint = loopInfo.Nprint;
  loopInfo.Nmoms = GK_Nmoms;
  int Nmoms = GK_Nmoms;
  char filename_out[512];

  int deflSteps = loopInfo.nSteps_defl;
  int nDefl[deflSteps];
  for(int a=0; a<deflSteps; a++) nDefl[a] = loopInfo.deflStep[a];


  FILE_WRITE_FORMAT LoopFileFormat = loopInfo.FileFormat;

  char loop_exact_fname[512];
  char loop_stoch_fname[512];
  

  // std-ultra_local
  loopInfo.loop_type[0] = "Scalar"; 
  loopInfo.loop_oneD[0] = false;
  // gen-ultra_local
  loopInfo.loop_type[1] = "dOp";    
  loopInfo.loop_oneD[1] = false;   
  // std-one_derivative
  loopInfo.loop_type[2] = "Loops";  
  loopInfo.loop_oneD[2] = true;    
  // std-conserved current
  loopInfo.loop_type[3] = "LoopsCv";
  loopInfo.loop_oneD[3] = true;    
  // gen-one_derivative
  loopInfo.loop_type[4] = "LpsDw";  
  loopInfo.loop_oneD[4] = true;   
  // gen-conserved current 
  loopInfo.loop_type[5] = "LpsDwCv";
  loopInfo.loop_oneD[5] = true;   

  printfQuda("\nLoop Calculation Info\n");
  printfQuda("=====================\n");
  printfQuda(" The seed is: %ld\n",seed);
  printfQuda(" The conf trajectory is: %04d\n",loopInfo.traj);
  printfQuda(" Will produce the loop for %d Momentum Combinations\n",Nmoms);
  printfQuda(" The loop file format is %s\n", (LoopFileFormat == ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda(" Will write the loops in %s\n", loopInfo.HighMomForm ? "High-Momenta Form" : "Standard Form");
  printfQuda(" The loop base name is %s\n",loopInfo.loop_fname);
  
  if(isProbingMstep){
    warningQuda("You have chosen hierarchical probing with Msteps. Be very carefull use the same seed and keep partitioning the same to have the same noise vectors\n");
    printfQuda(" %d Stoch vectors, %d Hadamard vectors (using Mstep), %d spin-colour diluted : %04d inversions\n", Nstoch, Nc_high-Nc_low, Nsc, Nstoch*(Nc_high-Nc_low)*Nsc);
  }
  else
    printfQuda(" %d Stoch vectors, %d Hadamard vectors, %d spin-colour diluted : %04d inversions\n", Nstoch, Nc, Nsc, Nstoch*Nc*Nsc);

  printfQuda(" Will project\n");
  for (int a=0; a<deflSteps; a++) printfQuda(" Ndefl %d: %d\n", a, nDefl[a]);
  printfQuda(" exact eigenmodes fom the solutions\n");
  if(info.source_type==RANDOM) printfQuda(" Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda(" Will use UNITY stochastic sources\n");
  printfQuda("=====================\n\n");
  
  bool exact_part = true;
  bool stoch_part = false;

  bool LowPrecSum = true;
  bool HighPrecSum = false;

  //======================================================================//
  //================ M E M O R Y   A L L O C A T I O N ===================// 
  //======================================================================//


  //-------------- Allocate memory for accumulation buffers --------------//
  //======================================================================// 
  // These buffers will first be used to hold the data from the exaxt part
  // of the operator, then they will be reset to hold data from the 
  // stochastic part. 
 

  void *tmp_loop;  
  if((cudaHostAlloc(&tmp_loop, sizeof(double)*2*16*GK_localVolume, 
		    cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("%s: Error allocating memory tmp_loop\n",fname);
  cudaMemset(tmp_loop      , 0, sizeof(double)*2*16*GK_localVolume);
  
  
  //Ultra Local
  void *std_uloc[deflSteps];
  void *gen_uloc[deflSteps];

  //One Derivative
  void **std_oneD[deflSteps];
  void **gen_oneD[deflSteps];
  void **std_csvC[deflSteps];
  void **gen_csvC[deflSteps];  

  for(int step=0;step<deflSteps;step++){  
    //- ultra-local loops
    if((cudaHostAlloc(&(std_uloc[step]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory std_uloc[%d]\n",fname,step);
    if((cudaHostAlloc(&(gen_uloc[step]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory gen_uloc[%d]\n",fname,step);
    
    cudaMemset(std_uloc[step], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc[step], 0, sizeof(double)*2*16*GK_localVolume);

    cudaDeviceSynchronize();

    //- one-Derivative and conserved current loops
    std_oneD[step] = (void**) malloc(sizeof(double*)*4);
    gen_oneD[step] = (void**) malloc(sizeof(double*)*4);
    std_csvC[step] = (void**) malloc(sizeof(double*)*4);
    gen_csvC[step] = (void**) malloc(sizeof(double*)*4);
    
    if(std_oneD[step] == NULL) 
      errorQuda("%s: Error allocating memory std_oneD[%d]\n",fname,step);
    if(gen_oneD[step] == NULL) 
      errorQuda("%s: Error allocating memory gen_oneD[%d]\n",fname,step);
    if(std_csvC[step] == NULL) 
      errorQuda("%s: Error allocating memory std_csvC[%d]\n",fname,step);
    if(gen_csvC[step] == NULL) 
      errorQuda("%s: Error allocating memory gen_csvC[%d]\n",fname,step);
    cudaDeviceSynchronize();
    
    for(int mu = 0; mu < 4 ; mu++){
      if((cudaHostAlloc(&(std_oneD[step][mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory std_oneD[%d][%d]\n",
		  fname,step,mu);
      if((cudaHostAlloc(&(gen_oneD[step][mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory gen_oneD[%d][%d]\n",
		  fname,step,mu);
      if((cudaHostAlloc(&(std_csvC[step][mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory std_csvC[%d][%d]\n",
		  fname,step,mu);
      if((cudaHostAlloc(&(gen_csvC[step][mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory gen_csvC[%d][%d]\n",
		  fname,step,mu);
      
      cudaMemset(std_oneD[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
  }//-step (Nev)
  printfQuda("%s: Accumulation buffers memory allocated properly.\n",fname);
  //----------------------------------------------------------------------//
  
  //-Allocate memory for the write buffers
  int Nprt = Nprint;

  double *buf_std_uloc[deflSteps];
  double *buf_gen_uloc[deflSteps];
  double **buf_std_oneD[deflSteps];
  double **buf_gen_oneD[deflSteps];
  double **buf_std_csvC[deflSteps];
  double **buf_gen_csvC[deflSteps];

  for(int step=0;step<deflSteps;step++){

    buf_std_uloc[step] = 
      (double*)malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);
    buf_gen_uloc[step] = 
      (double*)malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);

    buf_std_oneD[step] = (double**) malloc(sizeof(double*)*4);
    buf_gen_oneD[step] = (double**) malloc(sizeof(double*)*4);  
    buf_std_csvC[step] = (double**) malloc(sizeof(double*)*4);
    buf_gen_csvC[step] = (double**) malloc(sizeof(double*)*4);
    
    if( buf_std_uloc[step] == NULL ) 
      errorQuda("Allocation of buffer buf_std_uloc[%d] failed.\n",step);
    if( buf_gen_uloc[step] == NULL ) 
      errorQuda("Allocation of buffer buf_gen_uloc[%d] failed.\n",step);
    
    if( buf_std_oneD[step] == NULL ) 
      errorQuda("Allocation of buffer buf_std_oneD[%d] failed.\n",step);
    if( buf_gen_oneD[step] == NULL ) 
      errorQuda("Allocation of buffer buf_gen_oneD[%d] failed.\n",step);
    if( buf_std_csvC[step] == NULL ) 
      errorQuda("Allocation of buffer buf_std_csvC[%d] failed.\n",step);
    if( buf_gen_csvC[step] == NULL ) 
      errorQuda("Allocation of buffer buf_gen_csvC[%d] failed.\n",step);
    
    for(int mu = 0; mu < 4 ; mu++){
      buf_std_oneD[step][mu] = 
	(double*) malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);
      buf_gen_oneD[step][mu] = 
	(double*) malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);
      buf_std_csvC[step][mu] = 
	(double*) malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);
      buf_gen_csvC[step][mu] = 
	(double*) malloc(sizeof(double)*Nprt*2*16*Nmoms*GK_localL[3]);
      
      if( buf_std_oneD[step][mu] == NULL ) 
	errorQuda("Allocation of buffer buf_std_oneD[%d][%d] failed.\n",
		  step,mu);
      if( buf_gen_oneD[step][mu] == NULL ) 
	errorQuda("Allocation of buffer buf_gen_oneD[%d][%d] failed.\n",
		  step,mu);
      if( buf_std_csvC[step][mu] == NULL ) 
	errorQuda("Allocation of buffer buf_std_csvC[%d][%d] failed.\n",
		  step,mu);
      if( buf_gen_csvC[step][mu] == NULL ) 
	errorQuda("Allocation of buffer buf_gen_csvC[%d][%d] failed.\n",
		  step,mu);
    }
  }


  printfQuda("%s: Write buffers memory allocated properly.\n",fname);
  //--------------------------------------
  //======================================================================//
  //========== E X A C T   P R O B L E M   C O N S T R U C T =============// 
  //======================================================================//

  // QKXTM: DMH Here the low modes of the normal M^{\dagger}M 
  //        operator are constructed. The correlation function
  //        values are calculated at every Nth deflation step:
  //        Nth = loopInfo.deflStep[s]
  
  printfQuda("\n ### Exact part calculation ###\n");

  int NeV_Full = arpackInfo.nEv;  
  
  //Create object to store and calculate eigenpairs
  QKXTM_Deflation<double> *deflation = 
    new QKXTM_Deflation<double>(EvInvParam,arpackInfo);
  deflation->printInfo();
  
  //- Calculate the eigenVectors
  t1 = MPI_Wtime(); 
  deflation->eigenSolver();
  t2 = MPI_Wtime();
  printfQuda("%s TIME REPORT:",fname);
  printfQuda("Full Operator EigenVector Calculation: %f sec\n",t2-t1);

  deflation->MapEvenOddToFull();

  // ==================== Prepare covariant derivative ======//
  DiracParam dWParam;
  dWParam.matpcType = QUDA_MATPC_EVEN_EVEN;
  dWParam.dagger    = QUDA_DAG_NO;
  dWParam.gauge     = gaugePrecise;
  dWParam.kappa     = param->kappa;
  dWParam.mass      = 1./(2.*param->kappa) - 4.;
  dWParam.m5        = 0.;
  dWParam.mu        = 0.;
  for(int i=0; i<4; i++)
    dWParam.commDim[i] = 1;

  if(param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH){
    dWParam.type = QUDA_CLOVER_DIRAC;
    dWParam.clover = cloverPrecise;
  } 
  else if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH){
    dWParam.type = QUDA_WILSON_DIRAC;
  }
  else{
    errorQuda("Operator not supported\n");
  }

  GaugeCovDev *cov = new GaugeCovDev(dWParam);
  //================================================//

  //- Calculate the exact part of the loop
  int iPrint = 0;
  int s;
  s = (loopInfo.deflStep[0] == 0) ? 1 : 0;
  
  for(int n=0;n<NeV_Full;n++){
    t1 = MPI_Wtime();
    deflation->Loop_w_One_Der_FullOp_Exact(n, EvInvParam,
					   gen_uloc[0], std_uloc[0], 
					   gen_oneD[0], std_oneD[0], 
					   gen_csvC[0], std_csvC[0], cov);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Exact part for EV %d done in: %f sec\n",
	       n+1,t2-t1);
    
    if( (n+1)==loopInfo.deflStep[s] ){     

      t1 = MPI_Wtime();
      performGPU_FT<double>(buf_std_uloc[0], std_uloc[0], iPrint);
      performGPU_FT<double>(buf_gen_uloc[0], gen_uloc[0], iPrint);
      for(int mu=0;mu<4;mu++){
	performGPU_FT<double>(buf_std_oneD[0][mu], std_oneD[0][mu], iPrint);
	performGPU_FT<double>(buf_std_csvC[0][mu], std_csvC[0][mu], iPrint);
	performGPU_FT<double>(buf_gen_oneD[0][mu], gen_oneD[0][mu], iPrint);
	performGPU_FT<double>(buf_gen_csvC[0][mu], gen_csvC[0][mu], iPrint);
      }
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: GPU FT in %f sec\n",t2-t1);

      //================================================================//
      //=============== D U M P   E X A C T   D A T A  =================// 
      //================================================================//

      //-Write the exact part of the loop
      sprintf(loop_exact_fname,"%s_exact_NeV%d",loopInfo.loop_fname,n+1);
      if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
	// Scalar
	writeLoops_ASCII(buf_std_uloc[0], loop_exact_fname, 
			 loopInfo,  0, 0, exact_part);
	// dOp
	writeLoops_ASCII(buf_gen_uloc[0], loop_exact_fname, 
			 loopInfo,  1, 0, exact_part);
	for(int mu = 0 ; mu < 4 ; mu++){
	  // Loops
	  writeLoops_ASCII(buf_std_oneD[0][mu], loop_exact_fname, 
			   loopInfo,  2, mu, exact_part);
	  // LoopsCv 
	  writeLoops_ASCII(buf_std_csvC[0][mu], loop_exact_fname, 
			   loopInfo,  3, mu, exact_part);
	  // LpsDw
	  writeLoops_ASCII(buf_gen_oneD[0][mu], loop_exact_fname, 
			   loopInfo,  4, mu, exact_part);
	  // LpsDwCv 
	  writeLoops_ASCII(buf_gen_csvC[0][mu], loop_exact_fname, 
			   loopInfo,  5, mu, exact_part); 
	}
      }
      else if(LoopFileFormat==HDF5_FORM){
	// Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc[0], buf_gen_uloc[0], 
			buf_std_oneD[0], buf_std_csvC[0], 
			buf_gen_oneD[0], buf_gen_csvC[0], 
			loop_exact_fname, loopInfo, 
			exact_part);
      }
      
      printfQuda("Writing the Exact part of the loops for NeV = %d completed.\n",n+1);
      s++;
    }//-if
  }//-for NeV_Full
  
  printfQuda("\n ### Exact part calculation Done ###\n");

  //=====================================================================//
  //======  S T O C H A S T I C   P R O B L E M   C O N S T R U C T =====//
  //=====================================================================//

  printfQuda("\n ### Stochastic part calculation ###\n\n");

  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge<double> *K_gauge = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  // QKXTM: DMH Calculation should default to these settings.
  printfQuda("%s: Will solve the stochastic part using Multigrid.\n",fname);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
    (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (param->solve_type == QUDA_NORMOP_PC_SOLVE) || (param->solve_type == QUDA_NORMERR_PC_SOLVE);
  bool mat_solution = (param->solution_type == QUDA_MAT_SOLUTION) ||
    (param->solution_type ==  QUDA_MATPC_SOLUTION);
  bool direct_solve = (param->solve_type == QUDA_DIRECT_SOLVE) ||
    (param->solve_type == QUDA_DIRECT_PC_SOLVE);

  // QKXTM: DMH This is a fairly arbitrary setting in terms of
  //        solver performance. true = Even-Even; false = Odd-Odd
  bool flag_eo = false;
  if(info.isEven){
    printfQuda("%s: Solving for the Even-Even operator\n", fname);
    flag_eo = true;
  } else {
    printfQuda("%s: Solving for the Odd-Odd operator\n", fname);
  }

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;
  profileInvert.TPSTART(QUDA_PROFILE_H2D);


  ColorSpinorField *b = NULL;
  ColorSpinorField *x = NULL;
  ColorSpinorField *in = NULL;
  ColorSpinorField *out = NULL;
  ColorSpinorField *tmp3 = NULL;
  ColorSpinorField *tmp4 = NULL;
  ColorSpinorField *sol  = NULL;

  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*
			      spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*
			       spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

  void *temp_input_vector  = NULL;

  if(isProbing || spinColorDil){
    temp_input_vector=malloc(GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
    memset(temp_input_vector ,0,GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]*spinorSiteSize*sizeof(double));
  }


  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(input_vector, *param, X, 
			    pc_solution, param->input_location);
  ColorSpinorField *h_b = ColorSpinorField::Create(cpuParam);

  cpuParam.v = output_vector;
  cpuParam.location = param->output_location;
  ColorSpinorField *h_x = ColorSpinorField::Create(cpuParam);

  //Zero out the spinors
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  b    = new cudaColorSpinorField(*h_b, cudaParam);
  x    = new cudaColorSpinorField(cudaParam);
  tmp3 = new cudaColorSpinorField(cudaParam);
  tmp4 = new cudaColorSpinorField(cudaParam);
  
  profileInvert.TPSTOP(QUDA_PROFILE_H2D);

  QKXTM_Vector<double> *K_vector = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<double> *K_vecdef = 
    new QKXTM_Vector<double>(BOTH,VECTOR);

  //Solver operators
  DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);

  //-Set Randon Number Generator
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  //-Define the accumulation-sum limits
  int Nrun = Nstoch;
  int Nd   = Ndump;
  char *msg_str;
  asprintf(&msg_str,"LOOPS");

  //- Prepare the accumulation buffers for the stochastic part
  cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);

  for(int step=0;step<deflSteps;step++){
    cudaMemset(std_uloc[step], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc[step], 0, sizeof(double)*2*16*GK_localVolume);
    
    for(int mu = 0; mu < 4 ; mu++){
      cudaMemset(std_oneD[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC[step][mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
  }


  //--------- Begin loop over stochastc inversions ----------//
  //=========================================================//
  
  iPrint = -1;
  //Loop over stochastic sources
  for(int is = 0 ; is < Nrun ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);

    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: %s %04d - Source creation: %f sec\n",
	       msg_str,is+1,t2-t1);

    //Loop over probing iterations
    for( int ih = Nc_low ; ih < Nc_high ; ih++) {
      //Loop over spin-colour dilution
      for(int sc = 0 ; sc < Nsc ; sc++){
	t3 = MPI_Wtime();
	
	if(spinColorDil){
	  if(isProbing) {
	    get_probing4D_spinColor_dilution<double>(temp_input_vector, input_vector, Vc, ih, sc);
	  }
	  else
	    get_spinColor_dilution<double>(temp_input_vector, input_vector, sc);
	}
	else{
	  if(isProbing)
	    get_probing4D_dilution<double>(temp_input_vector, input_vector, Vc, ih);
	  else
	    temp_input_vector = input_vector;
	}
	
	K_vector->packVector((double*) temp_input_vector);
	K_vector->loadVector();
	K_vector->uploadToCuda(b,flag_eo);
	
	double orig_tol = param->tol;
	long int orig_maxiter = param->maxiter;
		
	t5 = 0.0;
     
	  
	t1 = MPI_Wtime();
	    
	dirac.prepare(in,out,*x,*b,param->solution_type); 

	SolverParam solverParam(*param);
	Solver *solve = Solver::create(solverParam, m, mSloppy, 
				       mPre, profileInvert);
	    
	(*solve)(*out,*in);	    
	dirac.reconstruct(*x,*b,param->solution_type);
	sol = new cudaColorSpinorField(*x);
	if(is == 0 && ih == 0 && is == 0) saveTuneCache();
	delete solve;
	t2 = MPI_Wtime();
	    
	t5 += t2 - t1;
	    	    	
	printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, "
		   "Spin-colour = %02d "
		   "- Full Inversion Time: %f sec\n",
		   msg_str, is, ih, sc, t5);
	
	// Revert to the original, high-precision values
	param->tol = orig_tol;           
	param->maxiter = orig_maxiter;
	
		  
	// Loop over the number of deflation steps
	for(int dstep=0;dstep<deflSteps;dstep++){
	  int NeV_defl = nDefl[dstep];
	    
	  t1 = MPI_Wtime();	
	  K_vector->downloadFromCuda(sol,flag_eo);
	  K_vector->download();
	    
	  // Solution is projected and put into x, x <- (1-UU^dag) x
	  deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);
	  K_vecdef->uploadToCuda(x, flag_eo);              
	    
	  t2 = MPI_Wtime();
	  printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d, NeV = %04d, Solution projection: %f sec\n",
		     msg_str, is, ih, sc, NeV_defl, t2-t1);
	    
	    
	  //Index to point to correct part of accumulation array and 
	  //write buffer
	  int idx = dstep;
	    
	  t1 = MPI_Wtime();
	  oneEndTrick_w_One_Der<double>(*x, *tmp3, *tmp4, param, 
					gen_uloc[idx], std_uloc[idx], 
					gen_oneD[idx], std_oneD[idx], 
					gen_csvC[idx], std_csvC[idx],cov);
	  t2 = MPI_Wtime();
	    
	  printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d, NeV = %04d, oneEndTrick: %f sec\n",
		     msg_str,is, ih, sc, NeV_defl, t2-t1);
	    
	  //Condition to assert if we are dumping at this stochastic source
	  //and if we have completed a loop over Hadamard vectors. If true,
	  //dump the data.
	  if( ((is+1)%Nd == 0)&&(ih*Nsc+sc == Nc_high*Nsc-1)){
	    //iPrint increments the starting points in the write buffers.
	    if(idx==0) iPrint++;
	    t1 = MPI_Wtime();
	    performGPU_FT<double>(buf_std_uloc[idx], std_uloc[idx], iPrint);
	    performGPU_FT<double>(buf_gen_uloc[idx], gen_uloc[idx], iPrint);
		
	    for(int mu=0;mu<4;mu++){
	      performGPU_FT<double>(buf_std_oneD[idx][mu], std_oneD[idx][mu], iPrint);
	      performGPU_FT<double>(buf_std_csvC[idx][mu], std_csvC[idx][mu], iPrint);
	      performGPU_FT<double>(buf_gen_oneD[idx][mu], gen_oneD[idx][mu], iPrint);
	      performGPU_FT<double>(buf_gen_csvC[idx][mu], gen_csvC[idx][mu], iPrint);
	    }
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d, NeV = %04d, Loops FFT and copy %f sec\n",msg_str, is, ih, sc, NeV_defl,  t2-t1);
	  }// Dump conditonal
	}// Deflation steps

	delete sol;
	t5 = MPI_Wtime();
	printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d"
		   " - Total Processing Time %f sec\n",msg_str, is, ih, sc, t5-t3);
	
      }// Spin-color dilution
    }// Hadamard vectors
  }// Nstoch
  
  //----------- Dump data at Nth NeV---------------------//
  //======================================================================//
  
  //in the HDf5 write routines
  // Loop over the number of deflation steps
  for(int dstep=0;dstep<deflSteps;dstep++){
    int NeV_defl = nDefl[dstep];
      
    //Index to point to correct part of accumulation array.
    int idx = dstep;
      
    t1 = MPI_Wtime();
    sprintf(loop_stoch_fname,"%s_stoch_NeV%d",
	    loopInfo.loop_fname, NeV_defl);
      
    if(LoopFileFormat==ASCII_FORM){ 
      // Write the loops in ASCII format
      writeLoops_ASCII(buf_std_uloc[idx], loop_stoch_fname, 
		       loopInfo,  0, 0, stoch_part); // Scalar
      writeLoops_ASCII(buf_gen_uloc[idx], loop_stoch_fname, 
		       loopInfo,  1, 0, stoch_part); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	writeLoops_ASCII(buf_std_oneD[idx][mu], loop_stoch_fname, 
			 loopInfo,  2, mu, stoch_part); // Loops
	writeLoops_ASCII(buf_std_csvC[idx][mu], loop_stoch_fname, 
			 loopInfo,  3, mu, stoch_part); // LoopsCv
	writeLoops_ASCII(buf_gen_oneD[idx][mu], loop_stoch_fname, 
			 loopInfo,  4, mu, stoch_part); // LpsDw
	writeLoops_ASCII(buf_gen_csvC[idx][mu], loop_stoch_fname, 
			 loopInfo,  5, mu, stoch_part); // LpsDwCv
      }
    }
    else if(LoopFileFormat==HDF5_FORM){ 
      // Write the loops in HDF5 format
      writeLoops_HDF5(buf_std_uloc[idx], buf_gen_uloc[idx], 
		      buf_std_oneD[idx], buf_std_csvC[idx], 
		      buf_gen_oneD[idx], buf_gen_csvC[idx],
		      loop_stoch_fname, loopInfo,  
		      stoch_part);
    }
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Writing the Stochastic part of the loops for NeV = %d: completed in %f sec.\n",NeV_defl,t2-t1);
  }//N defl 
  
  
  gsl_rng_free(rNum);
  
  printfQuda("\n ### Stochastic part calculation Done ###\n");
  
  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//

  printfQuda("\nCleaning up...\n");  
  //---------------------------
  
  //-Free loop buffers
  cudaFreeHost(tmp_loop);
  for(int step=0;step<deflSteps;step++){
    //-accumulation buffers
    cudaFreeHost(std_uloc[step]);
    cudaFreeHost(gen_uloc[step]);
    for(int mu = 0 ; mu < 4 ; mu++){
      cudaFreeHost(std_oneD[step][mu]);
      cudaFreeHost(gen_oneD[step][mu]);
      cudaFreeHost(std_csvC[step][mu]);
      cudaFreeHost(gen_csvC[step][mu]);
    }
    free(std_oneD[step]);
    free(gen_oneD[step]);
    free(std_csvC[step]);
    free(gen_csvC[step]);   
    
    //-write buffers
    free(buf_std_uloc[step]);
    free(buf_gen_uloc[step]);
    for(int mu = 0 ; mu < 4 ; mu++){
      free(buf_std_oneD[step][mu]);
      free(buf_std_csvC[step][mu]);
      free(buf_gen_oneD[step][mu]);
      free(buf_gen_csvC[step][mu]);
    }
    free(buf_std_oneD[step]);
    free(buf_std_csvC[step]);
    free(buf_gen_oneD[step]);
    free(buf_gen_csvC[step]);
  }//-step
  //---------------------------
  

  free(input_vector);
  free(output_vector);

  if(isProbing || spinColorDil)
    free(temp_input_vector);
  if(isProbing)
    free(Vc);

  delete deflation;
  delete cov;
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_vecdef;
  delete K_vector;
  delete K_gauge;
  delete x;
  delete h_x;
  delete b;
  delete h_b;
  delete tmp3;
  delete tmp4;  
  
  printfQuda("...Done\n");
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

#endif


