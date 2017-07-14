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
  param->preconditioner = param->preconditionerDN;
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
      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      tx1 = MPI_Wtime();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      tx2 = MPI_Wtime();
      summ_tx12 += tx2-tx1;

      K_guess->uploadToCuda(b,flag_eo);
      diracUP.prepare(in,out,*x,*b,param->solution_type);

      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet
      
      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      K_guess->uploadToCuda(out,flag_eo); 
      // initial guess is ready
      
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
      
      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      tx1=MPI_Wtime();
      K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
      tx2=MPI_Wtime();
      summ_tx12 += tx2-tx1;
      K_guess->uploadToCuda(b,flag_eo);
      diracDN.prepare(in,out,*x,*b,param->solution_type);
      
      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet

      K_vector->downloadFromCuda(in,flag_eo);
      K_vector->download();
      K_guess->uploadToCuda(out,flag_eo); 
      // initial guess is ready
      
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
					corrThp_oneD, PID, NUCLEON, 1, 
					isource, CorrSpace);
	  if(NUCLEON == NEUTRON) 
	    K_contract->contractFixSink(*K_seqProp, *K_prop_down, 
					*K_gaugeContractions, 
					corrThp_local, corrThp_noether, 
					corrThp_oneD, PID, NUCLEON, 1, 
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
					corrThp_oneD, PID, NUCLEON, 2, 
					isource, CorrSpace);
	  if(NUCLEON == NEUTRON) 
	    K_contract->contractFixSink(*K_seqProp, *K_prop_up, 
					*K_gaugeContractions, 
					corrThp_local, corrThp_noether, 
					corrThp_oneD, PID, NUCLEON, 2, 
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
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
	K_temp->castDoubleToFloat(*K_guess);
	K_prop_up->absorbVectorToDevice(*K_temp,nu,c2);
	
	K_temp->copyPropagator(*K_prop_down,nu,c2);
	K_vector->castFloatToDouble(*K_temp);
	K_guess->gaussianSmearing(*K_vector,*K_gaugeSmeared);
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

#endif

////////////////////////////////
// QKXTM Eigenvector routines //
////////////////////////////////

#ifdef HAVE_ARPACK


void calcLowModeProjection(void **gaugeToPlaquette, 
			   QudaInvertParam *EvInvParam, 
			   QudaInvertParam *param, 
			   QudaGaugeParam *gauge_param,
			   qudaQKXTM_arpackInfo arpackInfo, 
			   qudaQKXTMinfo info){
  
  double t1,t2,t3,t4,t5;
  char fname[256];
  sprintf(fname, "calcLowModeProjection");
  
  //======================================================================//
  //================= P A R A M E T E R   C H E C K S ====================//
  //======================================================================//

  if (!initialized) 
    errorQuda("%s: QUDA not initialized", fname);
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);
  
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

  // QKXTM: DMH Here the low modes of the normal M^{\dagger}M 
  //        operator are constructed. We then apply gamma_5 to
  //        each eigenvector u, and then take the inner product:
  //        G5iProd_i = u^dag_i \gamma_5 u_i
  
  printfQuda("\n ### Exact part calculation ###\n");

  int NeV_Full = arpackInfo.nEv;  
  
  //Create object to store and calculate eigenpairs
  QKXTM_Deflation<double> *deflation = 
    new QKXTM_Deflation<double>(param,arpackInfo);
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

void calcMG_loop_wOneD_TSM_wExact(void **gaugeToPlaquette, 
				  QudaInvertParam *EvInvParam, 
				  QudaInvertParam *param, 
				  QudaGaugeParam *gauge_param,
				  qudaQKXTM_arpackInfo arpackInfo, 
				  qudaQKXTM_loopInfo loopInfo, 
				  qudaQKXTMinfo info){

  double t1,t2,t3,t4,t5;
  char fname[256];
  sprintf(fname, "calcMG_loop_wOneD_TSM_wExact");
  
  //======================================================================//
  //================= P A R A M E T E R   C H E C K S ====================//
  //======================================================================//


  // parameters related to Probing and spinColorDil
  unsigned short int* Vc = NULL;
  int k_probing = loopInfo.k_probing;
  bool spinColorDil = loopInfo.spinColorDil;
  bool isProbing;
  int Nc; // number of Hadamard vectors, if not enabled then it is one
  int Nsc; // number of Spin-Color diluted vectors, if not enabled then it is one

  if(k_probing > 0){
    Nc = 2*pow(2,4*(k_probing-1));
    Vc = hch_coloring(k_probing,4); //4D hierarchical coloring
    isProbing=true;
  }
  else{
    Nc=1;
    isProbing=false;
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
  bool loopCovDev = loopInfo.loopCovDev;

  int deflSteps = loopInfo.nSteps_defl;
  int nDefl[deflSteps];
  for(int a=0; a<deflSteps; a++) nDefl[a] = loopInfo.deflStep[a];


  FILE_WRITE_FORMAT LoopFileFormat = loopInfo.FileFormat;

  char loop_exact_fname[512];
  char loop_stoch_fname[512];

  //Fix to true for data write routines only.
  bool useTSM = true;

  //-C.K. Truncated solver method params
  int TSM_NprintLP = loopInfo.TSM_NprintLP;
  int TSM_NLP_iters = loopInfo.TSM_NLP_iters;
  double TSM_tol[TSM_NLP_iters];
  int TSM_maxiter[TSM_NLP_iters];
  for(int a=0; a<TSM_NLP_iters; a++) {
    TSM_tol[a] = loopInfo.TSM_tol[a];
    TSM_maxiter[a] = loopInfo.TSM_maxiter[a];
  }
  

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
  printfQuda(" Will %sperform covariant derivative calculations\n",loopCovDev ? "" : "not ");
  printfQuda(" %d Stoch vectors, %d Hadamard vectors, %d spin-colour diluted : %04d inversions per TSM criterion\n", Nstoch, Nc, Nsc, Nstoch*Nc*Nsc);
  printfQuda(" N_LP_iters = %d Low precision stopping criteria\n",TSM_NLP_iters);
  for(int a=0; a<TSM_NLP_iters; a++) {
    if (TSM_maxiter[0] == 0) printfQuda(" Solver stopping criterion %d is: tol = %e\n",a, TSM_tol[a]);
    else printfQuda(" Solver stopping criterion %d is: max-iter = %ld\n", a, TSM_maxiter[a]);
  }
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
  void *std_uloc[deflSteps*TSM_NLP_iters];
  void *gen_uloc[deflSteps*TSM_NLP_iters];

  //One Derivative
  void **std_oneD[deflSteps*TSM_NLP_iters];
  void **gen_oneD[deflSteps*TSM_NLP_iters];
  void **std_csvC[deflSteps*TSM_NLP_iters];
  void **gen_csvC[deflSteps*TSM_NLP_iters];  

  for(int step=0;step<deflSteps*TSM_NLP_iters;step++){  
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
  }//-step (Nev and TSM LP)
  printfQuda("%s: Accumulation buffers memory allocated properly.\n",fname);
  //----------------------------------------------------------------------//
  
  //-Allocate memory for the write buffers
  int Nprt = TSM_NprintLP*TSM_NLP_iters;

  double *buf_std_uloc[deflSteps*TSM_NLP_iters];
  double *buf_gen_uloc[deflSteps*TSM_NLP_iters];
  double **buf_std_oneD[deflSteps*TSM_NLP_iters];
  double **buf_gen_oneD[deflSteps*TSM_NLP_iters];
  double **buf_std_csvC[deflSteps*TSM_NLP_iters];
  double **buf_gen_csvC[deflSteps*TSM_NLP_iters];

  for(int step=0;step<deflSteps*TSM_NLP_iters;step++){

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
  
  //-Allocate the momenta
  int **mom,**momQsq;
  long int SplV = GK_totalL[0]*GK_totalL[1]*GK_totalL[2];
  mom =    (int**) malloc(sizeof(int*)*SplV);
  momQsq = (int**) malloc(sizeof(int*)*Nmoms);
  if(mom    == NULL) errorQuda("Error in allocating mom\n");
  if(momQsq == NULL) errorQuda("Error in allocating momQsq\n");
  
  for(int ip=0; ip<SplV; ip++) {
    mom[ip] = (int*) malloc(sizeof(int)*3);
    if(mom[ip] == NULL) errorQuda("Error in allocating mom[%d]\n",ip);
  }
  for(int ip=0; ip<Nmoms; ip++) {
    momQsq[ip] = (int *) malloc(sizeof(int)*3);
    if(momQsq[ip] == NULL) errorQuda("Error in allocating momQsq[%d]\n",ip);
  }
  createLoopMomenta(mom,momQsq,info.Q_sq,Nmoms);
  printfQuda("%s: Momenta created\n",fname);

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

  //- Calculate the exact part of the loop
  int iPrint = 0;
  int s = 0;
  for(int n=0;n<NeV_Full;n++){
    t1 = MPI_Wtime();
    deflation->Loop_w_One_Der_FullOp_Exact(n, EvInvParam, loopCovDev,
					   gen_uloc[0], std_uloc[0], 
					   gen_oneD[0], std_oneD[0], 
					   gen_csvC[0], std_csvC[0]);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Exact part for EV %d done in: %f sec\n",
	       n+1,t2-t1);
    
    if( (n+1)==loopInfo.deflStep[s] ){     
      if(GK_nProc[2]==1){      
	doCudaFFT_v2<double>(std_uloc[0], tmp_loop); // Scalar
	copyLoopToWriteBuf(buf_std_uloc[0], tmp_loop,
			   iPrint, info.Q_sq, Nmoms,mom);
	doCudaFFT_v2<double>(gen_uloc[0], tmp_loop); // dOp
	copyLoopToWriteBuf(buf_gen_uloc[0], tmp_loop, 
			   iPrint, info.Q_sq, Nmoms,mom);
	
	for(int mu = 0 ; mu < 4 ; mu++){
	  doCudaFFT_v2<double>(std_oneD[0][mu], tmp_loop); // Loops
	  copyLoopToWriteBuf(buf_std_oneD[0][mu], tmp_loop,
			     iPrint, info.Q_sq, Nmoms,mom);
	  doCudaFFT_v2<double>(std_csvC[0][mu], tmp_loop); // LoopsCv
	  copyLoopToWriteBuf(buf_std_csvC[0][mu], tmp_loop, 
			     iPrint, info.Q_sq, Nmoms, mom);
	  
	  doCudaFFT_v2<double>(gen_oneD[0][mu], tmp_loop); // LpsDw
	  copyLoopToWriteBuf(buf_gen_oneD[0][mu], tmp_loop, 
			     iPrint, info.Q_sq, Nmoms, mom);
	  doCudaFFT_v2<double>(gen_csvC[0][mu], tmp_loop); // LpsDwCv
	  copyLoopToWriteBuf(buf_gen_csvC[0][mu], tmp_loop, 
			     iPrint, info.Q_sq, Nmoms, mom);
	}
	printfQuda("Exact part of Loops for NeV = %d copied to write buffers\n",n+1);
      }
      else if(GK_nProc[2]>1){
	t1 = MPI_Wtime();
	performFFT<double>(buf_std_uloc[0], std_uloc[0], 
			   iPrint, Nmoms, momQsq);
	performFFT<double>(buf_gen_uloc[0], gen_uloc[0], 
			   iPrint, Nmoms, momQsq);
	
	for(int mu=0;mu<4;mu++){
	  performFFT<double>(buf_std_oneD[0][mu], std_oneD[0][mu], 
			     iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_std_csvC[0][mu], std_csvC[0][mu], 
			     iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_oneD[0][mu], gen_oneD[0][mu], 
			     iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_csvC[0][mu], gen_csvC[0][mu], 
			     iPrint, Nmoms, momQsq);
	}
	t2 = MPI_Wtime();
	printfQuda("TIME_REPORT: FFT and copying to Write Buffers is %f sec\n",t2-t1);
      }

      //================================================================//
      //=============== D U M P   E X A C T   D A T A  =================// 
      //================================================================//

      //-Write the exact part of the loop
      sprintf(loop_exact_fname,"%s_exact_NeV%d",loopInfo.loop_fname,n+1);
      if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
	// Scalar
	writeLoops_ASCII(buf_std_uloc[0], loop_exact_fname, 
			 loopInfo, momQsq, 0, 0, exact_part, false, false);
	// dOp
	writeLoops_ASCII(buf_gen_uloc[0], loop_exact_fname, 
			 loopInfo, momQsq, 1, 0, exact_part, false ,false);
	for(int mu = 0 ; mu < 4 ; mu++){
	  // Loops
	  writeLoops_ASCII(buf_std_oneD[0][mu], loop_exact_fname, 
			   loopInfo, momQsq, 2, mu, exact_part,false,false);
	  // LoopsCv 
	  writeLoops_ASCII(buf_std_csvC[0][mu], loop_exact_fname, 
			   loopInfo, momQsq, 3, mu, exact_part,false,false);
	  // LpsDw
	  writeLoops_ASCII(buf_gen_oneD[0][mu], loop_exact_fname, 
			   loopInfo, momQsq, 4, mu, exact_part,false,false);
	  // LpsDwCv 
	  writeLoops_ASCII(buf_gen_csvC[0][mu], loop_exact_fname, 
			   loopInfo, momQsq, 5, mu, exact_part,false,false); 
	}
      }
      else if(LoopFileFormat==HDF5_FORM){
	// Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc[0], buf_gen_uloc[0], 
			buf_std_oneD[0], buf_std_csvC[0], 
			buf_gen_oneD[0], buf_gen_csvC[0], 
			loop_exact_fname, loopInfo, 
			momQsq, exact_part, false, false);
      }
      
      printfQuda("Writing the Exact part of the loops for NeV = %d completed.\n",n+1);
      s++;
    }//-if
  }//-for NeV_Full
  
  printfQuda("\n ### Exact part calculation Done ###\n");

  //=====================================================================//
  //======  S T O C H A S T I C   P R O B L E M   C O N S T R U C T =====//
  //=====================================================================//

  //QKXTM: DMH Here we calculate the contribution to the All-to-All
  //       propagator from stochastic sources. In previous versions
  //       of this code, deflation was used to accelerate the inversions
  //       using either a deflation operator from the exact part with a 
  //       normal (M^+M \phi = M^+ \eta) solve type, or exact deflation 
  //       and an extra Even/Odd preconditioned deflation step on the 
  //       remainder.

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

  ColorSpinorField *x_LP[TSM_NLP_iters];
  ColorSpinorField *sol_LP[TSM_NLP_iters];
  for(int a=0; a<TSM_NLP_iters; a++) {
    x_LP[a]   = NULL;
    sol_LP[a] = NULL;
  }

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
  for(int a=0; a<TSM_NLP_iters; a++) 
    x_LP[a] = new cudaColorSpinorField(cudaParam);
  
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
  if(useTSM == true ? asprintf(&msg_str,"TSM") : asprintf(&msg_str,"NO_TSM") );

  //- Prepare the accumulation buffers for the stochastic part
  cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);

  for(int step=0;step<deflSteps*TSM_NLP_iters;step++){
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
    for( int ih = 0 ; ih < Nc ; ih++) {
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
	
	//If we are using the TSM, we need the LP
	//solves for bias estimation. We loop over
	//the LP stopping criteria, and store each 
	//solution as a guess for the next LP. 
	//If TSM_NLP_iters is set to 1, this loop,
	//runs once.
	
	t5 = 0.0;
	
	for(int LP_crit = 0; LP_crit<TSM_NLP_iters; LP_crit++) {
	  
	  t1 = MPI_Wtime();
	  // Set the low-precision criterion
	  if(TSM_maxiter[0]==0) param->tol = TSM_tol[LP_crit];
	  else if(TSM_tol[0]==0) param->maxiter = TSM_maxiter[LP_crit];  
	    
	  dirac.prepare(in,out,*x_LP[LP_crit],*b,param->solution_type); 

	  // Create the low-precision solver
	  if(LP_crit > 0) param->use_init_guess = QUDA_USE_INIT_GUESS_YES;
	    SolverParam solverParam_LP(*param);
	    Solver *solve_LP = Solver::create(solverParam_LP, m, mSloppy, 
					      mPre, profileInvert);
	    
	    //LP solve
	    (*solve_LP)(*out,*in);	    
	    dirac.reconstruct(*x_LP[LP_crit],*b,param->solution_type);
	    //Store each LP solution
	    sol_LP[LP_crit] = new cudaColorSpinorField(*x_LP[LP_crit]);
	    if(is == 0 && LP_crit == 0) saveTuneCache();
	    delete solve_LP;
	    t2 = MPI_Wtime();
	    
	    t5 += t2 - t1;
	    
	    //DMH: Experimental, attempt to reuse last result
	    if(LP_crit < TSM_NLP_iters - 1) blas::copy(*x_LP[LP_crit + 1], *x_LP[LP_crit]);
	    
	    printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, "
		       "Spin-colour = %02d, LP_crit = %02d "
		       "- Partial Inversion Time: %f sec\n",
		       msg_str, is, ih, sc, LP_crit, t2-t1);
	    
	}
	
	printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, "
		   "Spin-colour = %02d "
		   "- Full Inversion Time: %f sec\n",
		   msg_str, is, ih, sc, t5);
	
	// Revert to the original, high-precision values
	param->tol = orig_tol;           
	param->maxiter = orig_maxiter;
	
	
	//Loop over LP criteria slowest to preserve data structure
	//in the HDf5 write routines, with deflation steps running the 
	//fastest. If TSM_NLP_iters is set to 1, this 
	//loop runs once.
	for(int LP_crit=0; LP_crit<TSM_NLP_iters; LP_crit++){
	  
	  // Loop over the number of deflation steps
	  for(int dstep=0;dstep<deflSteps;dstep++){
	    int NeV_defl = nDefl[dstep];
	    
	    t1 = MPI_Wtime();	
	    K_vector->downloadFromCuda(sol_LP[LP_crit],flag_eo);
	    K_vector->download();
	    
	    // Solution is projected and put into x, x <- (1-UU^dag) x
	    deflation->projectVector(*K_vecdef,*K_vector,is+1,NeV_defl);
	    K_vecdef->uploadToCuda(x_LP[LP_crit], flag_eo);              
	    
	    t2 = MPI_Wtime();
	    printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, "
		       "Spin-colour = %02d, NeV = %04d, "
		       "LP crit = %02d - Solution projection: %f sec\n",
		       msg_str, is, ih, sc, NeV_defl, LP_crit, t2-t1);
	    
	    
	    //Index to point to correct part of accumulation array and 
	    //write buffer
	    int idx = LP_crit*deflSteps + dstep;
	    
	    t1 = MPI_Wtime();
	    oneEndTrick_w_One_Der<double>(*x_LP[LP_crit], *tmp3, *tmp4, param, loopCovDev,
					  gen_uloc[idx], std_uloc[idx], 
					  gen_oneD[idx], std_oneD[idx], 
					  gen_csvC[idx], std_csvC[idx]);
	    t2 = MPI_Wtime();
	    
	    printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, "
		       "Spin-colour = %02d, NeV = %04d, "
		       "LP crit = %02d - oneEndTrick: %f sec\n",
		       msg_str,is, ih, sc, NeV_defl, LP_crit, t2-t1);
	    
	    //Condition to assert if we are dumping at this stochastic source
	    //and if we have completed a loop over Hadamard vectors. If true,
	    //dump the data.
	    if( ((is+1)%Nd == 0)&&(ih*Nsc+sc == Nc*Nsc-1)){
	      //iPrint increments the starting points in the write buffers.
	      if(idx==0) iPrint++;
	      
	      t1 = MPI_Wtime();
	      if(GK_nProc[2]==1){      
		doCudaFFT_v2<double>(std_uloc[idx], tmp_loop); // Scalar
		copyLoopToWriteBuf(buf_std_uloc[idx], tmp_loop, 
				   iPrint, info.Q_sq, Nmoms, mom);
		doCudaFFT_v2<double>(gen_uloc[idx], tmp_loop); // dOp
		copyLoopToWriteBuf(buf_gen_uloc[idx], tmp_loop, 
				   iPrint, info.Q_sq, Nmoms, mom);
		
		for(int mu = 0 ; mu < 4 ; mu++){
		  doCudaFFT_v2<double>(std_oneD[idx][mu], tmp_loop); // Loops
		  copyLoopToWriteBuf(buf_std_oneD[idx][mu], tmp_loop, 
				     iPrint, info.Q_sq, Nmoms, mom);
		  doCudaFFT_v2<double>(std_csvC[idx][mu], tmp_loop); // LoopsCv
		  copyLoopToWriteBuf(buf_std_csvC[idx][mu], tmp_loop, 
				     iPrint, info.Q_sq, Nmoms, mom);	      
		  doCudaFFT_v2<double>(gen_oneD[idx][mu],tmp_loop); // LpsDw
		  copyLoopToWriteBuf(buf_gen_oneD[idx][mu], tmp_loop, 
				     iPrint, info.Q_sq, Nmoms, mom);
		  doCudaFFT_v2<double>(gen_csvC[idx][mu], tmp_loop); // LpsDwCv
		  copyLoopToWriteBuf(buf_gen_csvC[idx][mu], tmp_loop, 
				     iPrint, info.Q_sq, Nmoms, mom);
		}
	      }
	      else if(GK_nProc[2]>1){
		performFFT<double>(buf_std_uloc[idx], std_uloc[idx], 
				   iPrint, Nmoms, momQsq);
		performFFT<double>(buf_gen_uloc[idx], gen_uloc[idx], 
				   iPrint, Nmoms, momQsq);
		
		for(int mu=0;mu<4;mu++){
		  performFFT<double>(buf_std_oneD[idx][mu], std_oneD[idx][mu],
				     iPrint, Nmoms, momQsq);
		  performFFT<double>(buf_std_csvC[idx][mu], std_csvC[idx][mu],
				     iPrint, Nmoms, momQsq);
		  performFFT<double>(buf_gen_oneD[idx][mu], gen_oneD[idx][mu],
				     iPrint, Nmoms, momQsq);
		  performFFT<double>(buf_gen_csvC[idx][mu], gen_csvC[idx][mu],
				     iPrint, Nmoms, momQsq);
		}
	      }
	      t2 = MPI_Wtime();
	      printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d, NeV = %04d, LP crit = %02d"
			 " - Loops FFT and copy %f sec\n",msg_str, is, ih, sc, NeV_defl, LP_crit, t2-t1);
	    }// Dump conditonal
	  }// Deflation steps
	}// LP criteria
	for(int a = 0; a<TSM_NLP_iters; a++) delete sol_LP[a];
	t5 = MPI_Wtime();
	printfQuda("TIME_REPORT: %s Stoch = %02d, HadVec = %02d, Spin-colour = %02d"
		   " - Total Processing Time %f sec\n",msg_str, is, ih, sc, t5-t3);
	
      }// Spin-color dilution
    }// Hadamard vectors
  }// Nstoch
  
  //----------- Dump data at Nth NeV and LP criteria ---------------------//
  //======================================================================//
  
  //Loop over LP criteria first to preserve data structure
  //in the HDf5 write routines
  for(int LP_crit=0; LP_crit<TSM_NLP_iters; LP_crit++){
    
    // Loop over the number of deflation steps
    for(int dstep=0;dstep<deflSteps;dstep++){
      int NeV_defl = nDefl[dstep];
      
      //Index to point to correct part of accumulation array.
      int idx = LP_crit*deflSteps + dstep;
      
      t1 = MPI_Wtime();
      sprintf(loop_stoch_fname,"%s_stoch_TSM_LP-crit-%d_NeV%d",
	      loopInfo.loop_fname, LP_crit, NeV_defl);
      
      if(LoopFileFormat==ASCII_FORM){ 
	// Write the loops in ASCII format
	writeLoops_ASCII(buf_std_uloc[idx], loop_stoch_fname, 
			 loopInfo, momQsq, 0, 0, stoch_part, 
			 useTSM, LowPrecSum); // Scalar
	writeLoops_ASCII(buf_gen_uloc[idx], loop_stoch_fname, 
			 loopInfo, momQsq, 1, 0, stoch_part, 
			 useTSM, LowPrecSum); // dOp
	for(int mu = 0 ; mu < 4 ; mu++){
	  writeLoops_ASCII(buf_std_oneD[idx][mu], loop_stoch_fname, 
			   loopInfo, momQsq, 2, mu, stoch_part, 
			   useTSM, LowPrecSum); // Loops
	  writeLoops_ASCII(buf_std_csvC[idx][mu], loop_stoch_fname, 
			   loopInfo, momQsq, 3, mu, stoch_part, 
			   useTSM, LowPrecSum); // LoopsCv
	  writeLoops_ASCII(buf_gen_oneD[idx][mu], loop_stoch_fname, 
			   loopInfo, momQsq, 4, mu, stoch_part, 
			   useTSM, LowPrecSum); // LpsDw
	  writeLoops_ASCII(buf_gen_csvC[idx][mu], loop_stoch_fname, 
			   loopInfo, momQsq, 5, mu, stoch_part, 
			   useTSM, LowPrecSum); // LpsDwCv
	}
      }
      else if(LoopFileFormat==HDF5_FORM){ 
	// Write the loops in HDF5 format
	writeLoops_HDF5(buf_std_uloc[idx], buf_gen_uloc[idx], 
			buf_std_oneD[idx], buf_std_csvC[idx], 
			buf_gen_oneD[idx], buf_gen_csvC[idx],
			loop_stoch_fname, loopInfo, momQsq, 
			stoch_part, useTSM, LowPrecSum);
      }
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: Writing the Stochastic part of the loops for NeV = %d, LP crit %d: completed in %f sec.\n",NeV_defl,LP_crit,t2-t1);
    }//N defl 
  } //LP criteria
  
  
  gsl_rng_free(rNum);
  
  printfQuda("\n ### Stochastic part calculation Done ###\n");
  
  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//

  printfQuda("\nCleaning up...\n");
  
  //-Free the momentum matrices
  for(int ip=0; ip<SplV; ip++) free(mom[ip]);
  free(mom);
  for(int ip=0;ip<Nmoms;ip++) free(momQsq[ip]);
  free(momQsq);
  //---------------------------
  
  //-Free loop buffers
  cudaFreeHost(tmp_loop);
  for(int step=0;step<deflSteps*TSM_NLP_iters;step++){
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

  for(int a=0; a<loopInfo.TSM_NLP_iters; a++) {
    delete x_LP[a];
  }
  
  
  printfQuda("...Done\n");
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

#endif


//-===========================================================
//- A D D I T I O N A L  D E P R E C A T E D   R O U T I N E S
//-===========================================================

/*

void calcMG_loop_wOneD_TSM_EvenOdd(void **gaugeToPlaquette, 
				   QudaInvertParam *param, 
				   QudaGaugeParam *gauge_param, 
				   qudaQKXTM_loopInfo loopInfo, 
				   qudaQKXTMinfo info){
  
  bool flag_eo;
  double t1,t2,t3,t4;
  char fname[256];
  sprintf(fname, "calcMG_loop_wOneD_TSM_EvenOdd");
  
  //======================================================================//
  //================= P A R A M E T E R   C H E C K S ====================//
  //======================================================================//
  
  profileInvert.TPSTART(QUDA_PROFILE_TOTAL);
  
  if(param->inv_type != QUDA_GCR_INVERTER) 
    errorQuda("%s: This function works only with GCR method", fname);
  
  if(param->gamma_basis != QUDA_UKQCD_GAMMA_BASIS) 
    errorQuda("%s: This function works only with ukqcd gamma basis", fname);
  if(param->dirac_order != QUDA_DIRAC_ORDER) 
    errorQuda("%s: This function works only with color-inside-spin", fname);
  
  if( info.isEven && (param->matpc_type != QUDA_MATPC_EVEN_EVEN) ){
    errorQuda("%s: Inconsistency between operator types!", fname);
  }
  if( (!info.isEven) && (param->matpc_type != QUDA_MATPC_ODD_ODD) ){
    errorQuda("%s: Inconsistency between operator types!", fname);
  }
  
  if(info.isEven){
    printfQuda("%s: Solving for the Even-Even operator\n", fname);
    flag_eo = true;
  }
  else{
    printfQuda("%s: Solving for the Odd-Odd operator\n", fname);
    flag_eo = false;
  }

  if (!initialized) 
    errorQuda("%s: QUDA not initialized", fname);
  pushVerbosity(param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
    printQudaInvertParam(param);
  
  //Stochastic, momentum, and data dump information.
  int Nstoch = loopInfo.Nstoch;
  unsigned long int seed = loopInfo.seed;
  int Ndump = loopInfo.Ndump;
  int Nprint = loopInfo.Nprint;
  loopInfo.Nmoms = GK_Nmoms;
  int Nmoms = GK_Nmoms;
  char filename_out[512];

  FILE_WRITE_FORMAT LoopFileFormat = loopInfo.FileFormat;

  char loop_stoch_fname[512];

  //-C.K. Truncated solver method params
  bool useTSM = loopInfo.useTSM;
  int TSM_NHP = loopInfo.TSM_NHP;
  int TSM_NLP = loopInfo.TSM_NLP;
  int TSM_NdumpHP = loopInfo.TSM_NdumpHP;
  int TSM_NdumpLP = loopInfo.TSM_NdumpLP;
  int TSM_NprintHP = loopInfo.TSM_NprintHP;
  int TSM_NprintLP = loopInfo.TSM_NprintLP;
  long int TSM_maxiter = 0;
  double TSM_tol = 0.0;
  if( (loopInfo.TSM_tol == 0) && 
      (loopInfo.TSM_maxiter !=0 ) ) {
    // LP criterion fixed by iteration number
    TSM_maxiter = loopInfo.TSM_maxiter; 
  }
  else if( (loopInfo.TSM_tol != 0) && 
	   (loopInfo.TSM_maxiter == 0) ) {
    // LP criterion fixed by tolerance
    TSM_tol = loopInfo.TSM_tol;
  }
  else if( useTSM && 
	   (loopInfo.TSM_tol != 0) && 
	   (loopInfo.TSM_maxiter != 0) ){
    warningQuda("Both max-iter = %ld and tolerance = %lf defined as criterions for the TSM. Proceeding with max-iter = %ld criterion.\n",
		loopInfo.TSM_maxiter,
		loopInfo.TSM_tol,
		loopInfo.TSM_maxiter);
    // LP criterion fixed by iteration number
    TSM_maxiter = loopInfo.TSM_maxiter;
  }

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
  
  printfQuda("Loop Calculation Info\n");
  printfQuda("=====================\n");
  printfQuda("No. of noise vectors: %d\n",Nstoch);
  printfQuda("The seed is: %ld\n",seed);
  printfQuda("The conf trajectory is: %04d\n",loopInfo.traj);
  printfQuda("Will produce the loop for %d Momentum Combinations\n",Nmoms);
  printfQuda("Will dump every %d noise vectors, thus %d times\n",Ndump,Nprint);
  printfQuda("The loop file format is %s\n",(LoopFileFormat==ASCII_FORM) ? "ASCII" : "HDF5");
  printfQuda("The loop base name is %s\n",filename_out);
  if(info.source_type==RANDOM) printfQuda("Will use RANDOM stochastic sources\n");
  else if (info.source_type==UNITY) printfQuda("Will use UNITY stochastic sources\n");
  if(useTSM){
    printfQuda(" Will perform using TSM with the following parameters:\n");
    printfQuda("  -N_HP = %d\n",TSM_NHP);
    printfQuda("  -N_LP = %d\n",TSM_NLP);
    if (TSM_maxiter == 0) printfQuda("  -Stopping criterion is: tol = %e\n",TSM_tol);
    else printfQuda("  -Stopping criterion is: max-iter = %ld\n",TSM_maxiter);
    printfQuda("  -Will dump every %d HP vectors, thus %d times\n",TSM_NdumpHP,TSM_NprintHP);
    printfQuda("  -Will dump every %d LP vectors, thus %d times\n",TSM_NdumpLP,TSM_NprintLP);
    printfQuda("=====================\n");
  }
  else{
    printfQuda(" Will not perform the Truncated Solver method\n");
    printfQuda(" No. of noise vectors: %d\n",Nstoch);
    printfQuda(" Will dump every %d noise vectors, thus %d times\n",Ndump,Nprint);
  }
  
  bool stoch_part = false;

  bool LowPrecSum = true;
  bool HighPrecSum = false;

  //======================================================================//
  //================ M E M O R Y   A L L O C A T I O N ===================// 
  //======================================================================//

  //- Allocate memory for local loops
  void *std_uloc;
  void *gen_uloc;
  void *tmp_loop;

  if((cudaHostAlloc(&std_uloc, sizeof(double)*2*16*GK_localVolume, 
		    cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("%s: Error allocating memory std_uloc\n", fname);
  if((cudaHostAlloc(&gen_uloc, sizeof(double)*2*16*GK_localVolume, 
		    cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("%s: Error allocating memory gen_uloc\n", fname);
  if((cudaHostAlloc(&tmp_loop, sizeof(double)*2*16*GK_localVolume, 
		    cudaHostAllocMapped)) != cudaSuccess)
    errorQuda("%s: Error allocating memory tmp_loop\n", fname);
  
  cudaMemset(std_uloc, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset(gen_uloc, 0, sizeof(double)*2*16*GK_localVolume);
  cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);
  cudaDeviceSynchronize();

  //- Allocate memory for one-Derivative and conserved current loops
  void **std_oneD;
  void **gen_oneD;
  void **std_csvC;
  void **gen_csvC;

  std_oneD = (void**) malloc(4*sizeof(double*));
  gen_oneD = (void**) malloc(4*sizeof(double*));
  std_csvC = (void**) malloc(4*sizeof(double*));
  gen_csvC = (void**) malloc(4*sizeof(double*));

  if(gen_oneD == NULL)
    errorQuda("%s: Error allocating memory gen_oneD higher level\n", fname);
  if(std_oneD == NULL)
    errorQuda("%s: Error allocating memory std_oneD higher level\n", fname);
  if(std_csvC == NULL)
    errorQuda("%s: Error allocating memory std_csvC higher level\n", fname);
  if(gen_csvC == NULL)
    errorQuda("%s: Error allocating memory gen_csvC higher level\n", fname);
  cudaDeviceSynchronize();

  for(int mu = 0; mu < 4 ; mu++){
    if((cudaHostAlloc(&(std_oneD[mu]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory std_oneD\n", fname);
    if((cudaHostAlloc(&(gen_oneD[mu]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory gen_oneD\n", fname);
    if((cudaHostAlloc(&(std_csvC[mu]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory std_csvC\n", fname);
    if((cudaHostAlloc(&(gen_csvC[mu]), sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory gen_csvC\n", fname);

    cudaMemset(std_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(std_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
  }
  cudaDeviceSynchronize();
  //---------------------------------------------------------------------//

  //-Allocate memory for the write buffers
  double *buf_std_uloc = 
    (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));
  double *buf_gen_uloc = 
    (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));

  double **buf_std_oneD = (double**) malloc(4*sizeof(double*));
  double **buf_gen_oneD = (double**) malloc(4*sizeof(double*));
  double **buf_std_csvC = (double**) malloc(4*sizeof(double*));
  double **buf_gen_csvC = (double**) malloc(4*sizeof(double*));
  
  for(int mu = 0; mu < 4 ; mu++){
    buf_std_oneD[mu] = 
      (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));
    buf_gen_oneD[mu] = 
      (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));
    buf_std_csvC[mu] = 
      (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));
    buf_gen_csvC[mu] = 
      (double*) malloc(Nprint*2*16*Nmoms*GK_localL[3]*sizeof(double));
  }
  
  int Nprt = ( useTSM ? TSM_NprintLP : Nprint );

  //- Check allocations
  if( buf_std_uloc == NULL ) 
    errorQuda("%s: Allocation of buffer buf_std_uloc failed.\n", fname);
  if( buf_gen_uloc == NULL ) 
    errorQuda("%s: Allocation of buffer buf_gen_uloc failed.\n", fname);

  if( buf_std_oneD == NULL ) 
    errorQuda("%s: Allocation of buffer buf_std_oneD failed.\n", fname);
  if( buf_gen_oneD == NULL ) 
    errorQuda("%s: Allocation of buffer buf_gen_oneD failed.\n", fname);
  if( buf_std_csvC == NULL ) 
    errorQuda("%s: Allocation of buffer buf_std_csvC failed.\n", fname);
  if( buf_gen_csvC == NULL ) 
    errorQuda("%s: Allocation of buffer buf_gen_csvC failed.\n", fname);
  
  for(int mu = 0; mu < 4 ; mu++){
    if( buf_std_oneD[mu] == NULL ) 
      errorQuda("%s: Allocation of buffer buf_std_oneD[%d] failed.\n", fname, mu);
    if( buf_gen_oneD[mu] == NULL ) 
      errorQuda("%s: Allocation of buffer buf_gen_oneD[%d] failed.\n", fname, mu);
    if( buf_std_csvC[mu] == NULL ) 
      errorQuda("%s: Allocation of buffer buf_std_csvC[%d] failed.\n", fname, mu);
    if( buf_gen_csvC[mu] == NULL ) 
      errorQuda("%s: Allocation of buffer buf_gen_csvC[%d] failed.\n", fname, mu);
  }
  
  //- Allocate extra memory if using TSM
  void *std_uloc_LP, *gen_uloc_LP;
  void **std_oneD_LP, **gen_oneD_LP, **std_csvC_LP, **gen_csvC_LP;

  double *buf_std_uloc_LP,*buf_gen_uloc_LP;
  double *buf_std_uloc_HP,*buf_gen_uloc_HP;
  double **buf_std_oneD_LP, **buf_gen_oneD_LP, **buf_std_csvC_LP, 
    **buf_gen_csvC_LP;
  double **buf_std_oneD_HP, **buf_gen_oneD_HP, **buf_std_csvC_HP, 
    **buf_gen_csvC_HP;

  if(useTSM){  
    //- local 
    if((cudaHostAlloc(&std_uloc_LP, sizeof(double)*2*16*GK_localVolume, 
 		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory std_uloc_LP\n", fname);
    if((cudaHostAlloc(&gen_uloc_LP, sizeof(double)*2*16*GK_localVolume, 
		      cudaHostAllocMapped)) != cudaSuccess)
      errorQuda("%s: Error allocating memory gen_uloc_LP\n", fname);
    
    cudaMemset(std_uloc_LP, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc_LP, 0, sizeof(double)*2*16*GK_localVolume);
    cudaDeviceSynchronize();
  
    //- one-Derivative and conserved current loops
    std_oneD_LP = (void**) malloc(4*sizeof(double*));
    gen_oneD_LP = (void**) malloc(4*sizeof(double*));
    std_csvC_LP = (void**) malloc(4*sizeof(double*));
    gen_csvC_LP = (void**) malloc(4*sizeof(double*));
    
    if(gen_oneD_LP == NULL) 
      errorQuda("%s: Error allocating memory gen_oneD_LP higher level\n", fname);
    if(std_oneD_LP == NULL) 
      errorQuda("%s: Error allocating memory std_oneD_LP higher level\n", fname);
    if(gen_csvC_LP == NULL) 
      errorQuda("%s: Error allocating memory gen_csvC_LP higher level\n", fname);
    if(std_csvC_LP == NULL) 
      errorQuda("%s: Error allocating memory std_csvC_LP higher level\n", fname);
    cudaDeviceSynchronize();
  
    for(int mu = 0; mu < 4 ; mu++){
      if((cudaHostAlloc(&(std_oneD_LP[mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory std_oneD_LP\n", fname);
      if((cudaHostAlloc(&(gen_oneD_LP[mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory gen_oneD_LP\n", fname);
      if((cudaHostAlloc(&(std_csvC_LP[mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory std_csvC_LP\n", fname);
      if((cudaHostAlloc(&(gen_csvC_LP[mu]), sizeof(double)*2*16*GK_localVolume, 
			cudaHostAllocMapped)) != cudaSuccess)
	errorQuda("%s: Error allocating memory gen_csvC_LP\n", fname);    
    
      cudaMemset(std_oneD_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC_LP[mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
  
  //-write buffers for Low-precision loops
    if( (buf_std_uloc_LP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc_LP failed.\n");
    if( (buf_gen_uloc_LP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc_LP failed.\n");
    
    if( (buf_std_oneD_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_LP failed.\n");
    if( (buf_gen_oneD_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_LP failed.\n");
    if( (buf_std_csvC_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_LP failed.\n");
    if( (buf_gen_csvC_LP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_LP failed.\n");
    
    for(int mu = 0; mu < 4 ; mu++){
      if( (buf_std_oneD_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_LP[%d] failed.\n",mu);
      if( (buf_gen_oneD_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_LP[%d] failed.\n",mu);
      if( (buf_std_csvC_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_LP[%d] failed.\n",mu);
      if( (buf_gen_csvC_LP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_LP[%d] failed.\n",mu);
    }
    
    //-write buffers for High-precision loops
    if( (buf_std_uloc_HP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_uloc_HP failed.\n");
    if( (buf_gen_uloc_HP = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_uloc_HP failed.\n");
    
    if( (buf_std_oneD_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_HP failed.\n");
    if( (buf_gen_oneD_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_HP failed.\n");
    if( (buf_std_csvC_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_HP failed.\n");
    if( (buf_gen_csvC_HP = (double**) malloc(4*sizeof(double*)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_HP failed.\n");
    
    for(int mu = 0; mu < 4 ; mu++){
      if( (buf_std_oneD_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_oneD_HP[%d] failed.\n",mu);
      if( (buf_gen_oneD_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_oneD_HP[%d] failed.\n",mu);
      if( (buf_std_csvC_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_std_csvC_HP[%d] failed.\n",mu);
      if( (buf_gen_csvC_HP[mu] = (double*) malloc(TSM_NprintHP*2*16*Nmoms*GK_localL[3]*sizeof(double)))==NULL ) errorQuda("Allocation of buffer buf_gen_csvC_HP[%d] failed.\n",mu);
    }
  }//-if useTSM

  //-Allocate the momenta
  int **mom,**momQsq;
  long int SplV = GK_totalL[0]*GK_totalL[1]*GK_totalL[2];
  if((mom =    (int**) malloc(sizeof(int*)*SplV )) == NULL) 
    errorQuda("Error in allocating mom\n");
  if((momQsq = (int**) malloc(sizeof(int*)*Nmoms)) == NULL) 
    errorQuda("Error in allocating momQsq\n");
  for(int ip=0; ip<SplV; ip++)
    if((mom[ip] = (int*) malloc(sizeof(int)*3)) == NULL) 
      errorQuda("Error in allocating mom[%d]\n",ip);

  for(int ip=0; ip<Nmoms; ip++)
    if((momQsq[ip] = (int *) malloc(sizeof(int)*3)) == NULL) 
      errorQuda("Error in allocating momQsq[%d]\n",ip);
  
  createLoopMomenta(mom,momQsq,info.Q_sq,Nmoms);
  printfQuda("Momenta created\n");

  //======================================================================//
  //================ P R O B L E M   C O N S T R U C T ===================// 
  //======================================================================//
  
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  QKXTM_Gauge<double> *K_gauge = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
  K_gauge->packGauge(gaugeToPlaquette);
  K_gauge->loadGauge();
  K_gauge->calculatePlaq();

  bool pc_solution = false;
  bool pc_solve = true;
  bool mat_solution = 
    (param->solution_type == QUDA_MAT_SOLUTION) || 
    (param->solution_type == QUDA_MATPC_SOLUTION);
  bool direct_solve = true;
  
  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? 
		       sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == 
			 QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == 
			 QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
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
  ColorSpinorField *x_LP   = NULL;
  ColorSpinorField *out_LP = NULL;

  const int *X = cudaGauge->X();

  void *input_vector = malloc(X[0]*X[1]*X[2]*X[3]*
			      spinorSiteSize*sizeof(double));
  void *output_vector = malloc(X[0]*X[1]*X[2]*X[3]*
			       spinorSiteSize*sizeof(double));

  memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
  memset(output_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));

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
  if(useTSM) x_LP = new cudaColorSpinorField(cudaParam);
  profileInvert.TPSTOP(QUDA_PROFILE_H2D);
  
  QKXTM_Vector<double> *K_vector = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<double> *K_guess = 
    new QKXTM_Vector<double>(BOTH,VECTOR);

  //Solver operators
  DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);

  //Random number generation
  gsl_rng *rNum = gsl_rng_alloc(gsl_rng_ranlux);
  gsl_rng_set(rNum, seed + comm_rank()*seed);

  //Set calculation parameters
  int Nrun;
  int Nd;
  char *msg_str;
  if(useTSM){
    Nrun = TSM_NLP;
    Nd = TSM_NdumpLP;
    asprintf(&msg_str,"NLP");
  }
  else{
    Nrun = Nstoch;
    Nd = Ndump;
    asprintf(&msg_str,"Stoch.");
  }
  
  //======================================================================//
  //================ P R O B L E M   E X E C U T I O N  ==================// 
  //======================================================================//
  
  //We first perform a loop over the desired number of stochastic vectors for 
  //production data. These will be LP solves if TSM is enabled.

  int iPrint = 0;

  //Loop over stochastic source vectors.
  for(int is = 0 ; is < Nrun ; is++){
    t1 = MPI_Wtime();
    memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
    getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: %s %04d - Source creation: %f sec\n",
	       msg_str,is+1,t2-t1);

    K_vector->packVector((double*) input_vector);
    K_vector->loadVector();
    K_vector->uploadToCuda(b,flag_eo);
    dirac.prepare(in,out,*x,*b,param->solution_type);
    // in is reference to the b but for a parity singlet
    // out is reference to the x but for a parity singlet

    K_vector->downloadFromCuda(in,flag_eo);
    K_vector->download();
    K_guess->uploadToCuda(out,flag_eo); // initial guess is ready

    if(useTSM) {
      //LP solve
      double orig_tol = param->tol;
      long int orig_maxiter = param->maxiter;
      // Set the low-precision criterion
      if(TSM_maxiter==0) param->tol = TSM_tol;
      else if(TSM_tol==0) param->maxiter = TSM_maxiter;  
      
      // Create the low-precision solver
      SolverParam solverParam_LP(*param);
      Solver *solve_LP = Solver::create(solverParam_LP, m, mSloppy, 
					mPre, profileInvert);
      //LP solve
      (*solve_LP)(*out,*in);
      delete solve_LP;
      
      // Revert to the original, high-precision values
      if(TSM_maxiter==0) param->tol = orig_tol;           
      else if(TSM_tol==0) param->maxiter = orig_maxiter;
    }
    else {
      //HP solve
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, 
				     mPre, profileInvert);
      (*solve)(*out,*in);
      delete solve;
    }
    
    dirac.reconstruct(*x,*b,param->solution_type);
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: Inversion for Stoch %04d is %f sec\n",
	       is+1,t2-t1);
    
    t1 = MPI_Wtime();
    oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param, gen_uloc, std_uloc, 
				  gen_oneD, std_oneD, gen_csvC, std_csvC);
    
    t2 = MPI_Wtime();
    printfQuda("TIME_REPORT: One-end trick for Stoch %04d is %f sec\n",
	       is+1,t2-t1);
    
    //======================================================================//
    //================ D U M P   D A T A   A T   NthStoch ==================//
    //======================================================================//
    
    if( (is+1)%Nd == 0){
      t1 = MPI_Wtime();
      if(GK_nProc[2]==1){      
	doCudaFFT_v2<double>(std_uloc,tmp_loop); // Scalar
	copyLoopToWriteBuf(buf_std_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	doCudaFFT_v2<double>(gen_uloc,tmp_loop); // dOp
	copyLoopToWriteBuf(buf_gen_uloc,tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	
	for(int mu = 0 ; mu < 4 ; mu++){
	  doCudaFFT_v2<double>(std_oneD[mu],tmp_loop); // Loops
	  copyLoopToWriteBuf(buf_std_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(std_csvC[mu],tmp_loop); // LoopsCv
	  copyLoopToWriteBuf(buf_std_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  
	  doCudaFFT_v2<double>(gen_oneD[mu],tmp_loop); // LpsDw
	  copyLoopToWriteBuf(buf_gen_oneD[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(gen_csvC[mu],tmp_loop); // LpsDwCv
	  copyLoopToWriteBuf(buf_gen_csvC[mu],tmp_loop,iPrint,info.Q_sq,Nmoms,mom);
	}
      }
      else if(GK_nProc[2]>1){
	performFFT<double>(buf_std_uloc, std_uloc, iPrint, Nmoms, momQsq);
	performFFT<double>(buf_gen_uloc, gen_uloc, iPrint, Nmoms, momQsq);
	
	for(int mu=0;mu<4;mu++){
	  performFFT<double>(buf_std_oneD[mu], std_oneD[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_std_csvC[mu], std_csvC[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_oneD[mu], gen_oneD[mu], iPrint, Nmoms, momQsq);
	  performFFT<double>(buf_gen_csvC[mu], gen_csvC[mu], iPrint, Nmoms, momQsq);
	}
      }
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: FFT and copying to Write Buffers is %f sec\n",t2-t1);
      iPrint++;
    }//-if (is+1)
  }//-loop over stochastic noise vectors
  
  //======================================================================//
  //================ W R I T E   F U L L   D A T A  ======================// 
  //======================================================================//
  
  //-Write the stochastic part of the loops
  t1 = MPI_Wtime();
  sprintf(loop_stoch_fname,"%s_stoch%sMG",loopInfo.loop_fname, useTSM ? "_TSM_" : "_");
  if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
    writeLoops_ASCII(buf_std_uloc, loop_stoch_fname, loopInfo, momQsq, 
		     0, 0, stoch_part, useTSM, LowPrecSum); // Scalar
    writeLoops_ASCII(buf_gen_uloc, loop_stoch_fname, loopInfo, momQsq, 
		     1, 0, stoch_part, useTSM, LowPrecSum); // dOp
    for(int mu = 0 ; mu < 4 ; mu++){
      writeLoops_ASCII(buf_std_oneD[mu], loop_stoch_fname, loopInfo, 
		       momQsq, 2, mu, stoch_part, useTSM, LowPrecSum); // Loops
      writeLoops_ASCII(buf_std_csvC[mu], loop_stoch_fname, loopInfo, 
		       momQsq, 3, mu, stoch_part, useTSM, LowPrecSum); // LoopsCv
      writeLoops_ASCII(buf_gen_oneD[mu], loop_stoch_fname, loopInfo, 
		       momQsq, 4, mu, stoch_part, useTSM, LowPrecSum); // LpsDw
      writeLoops_ASCII(buf_gen_csvC[mu], loop_stoch_fname, loopInfo, 
		       momQsq, 5, mu, stoch_part, useTSM, LowPrecSum); // LpsDwCv
    }
  }
  else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
    writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, 
		    buf_std_csvC, buf_gen_oneD, buf_gen_csvC, 
		    loop_stoch_fname, loopInfo, momQsq, 
		    stoch_part, useTSM, LowPrecSum);
  }
  t2 = MPI_Wtime();
  printfQuda("Writing the Stochastic part of the loops for MG completed in %f sec.\n",t2-t1);
  
  //Next we perform a loop over the desired number of stochastic vectors for 
  //HP and LP solves to estimate the bias from using a truncated solver. 
  //This will only be perfomed if TSM is enabled.

  if(useTSM){
    printfQuda("\nWill Perform the HP and LP inversions\n\n");
    
    //- These one-end trick buffers are to be re-used for the high-precision vectors
    cudaMemset(std_uloc, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(gen_uloc, 0, sizeof(double)*2*16*GK_localVolume);
    cudaMemset(tmp_loop, 0, sizeof(double)*2*16*GK_localVolume);
    
    for(int mu = 0; mu < 4 ; mu++){
      cudaMemset(std_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_oneD[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(std_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
      cudaMemset(gen_csvC[mu], 0, sizeof(double)*2*16*GK_localVolume);
    }
    cudaDeviceSynchronize();
    //---------------------------
    
    Nrun = TSM_NHP;
    Nd = TSM_NdumpHP;
    iPrint = 0;
    for(int is = 0 ; is < Nrun ; is++){
      t3 = MPI_Wtime();
      t1 = MPI_Wtime();
      memset(input_vector,0,X[0]*X[1]*X[2]*X[3]*spinorSiteSize*sizeof(double));
      getStochasticRandomSource<double>(input_vector,rNum,info.source_type);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: %s %04d - Source creation: %f sec\n",msg_str,is+1,t2-t1);

      K_vector->packVector((double*) input_vector);
      K_vector->loadVector();
      K_vector->uploadToCuda(b,flag_eo);

      blas::zero(*out);
      blas::zero(*out_LP);

      dirac.prepare(in,out,*x,*b,param->solution_type);
      dirac.prepare(in,out_LP,*x_LP,*b,param->solution_type);
      // in is reference to the b but for a parity singlet
      // out is reference to the x but for a parity singlet

      //HP solve
      //-------------------------------------------------
      SolverParam solverParam(*param);
      Solver *solve = Solver::create(solverParam, m, mSloppy, mPre, 
				     profileInvert);
      (*solve)   (*out,*in);
      delete solve;
      dirac.reconstruct(*x,*b,param->solution_type);
      //-------------------------------------------------


      //LP solve
      //-------------------------------------------------
      double orig_tol = param->tol;
      long int orig_maxiter = param->maxiter;
      // Set the low-precision criterion
      if(TSM_maxiter==0) param->tol = TSM_tol;
      else if(TSM_tol==0) param->maxiter = TSM_maxiter;  
      
      // Create the low-precision solver
      SolverParam solverParam_LP(*param);
      Solver *solve_LP = Solver::create(solverParam_LP, m, mSloppy, 
					mPre, profileInvert);
      (*solve_LP)(*out_LP,*in);
      delete solve_LP;
      dirac.reconstruct(*x_LP,*b,param->solution_type);
      
      // Revert to the original, high-precision values
      if(TSM_maxiter==0) param->tol = orig_tol;           
      else if(TSM_tol==0) param->maxiter = orig_maxiter;      
      //-------------------------------------------------


      // Contractions
      //-------------------------------------------------
      //-high-precision
      t1 = MPI_Wtime();
      oneEndTrick_w_One_Der<double>(*x,*tmp3,*tmp4,param, 
				    gen_uloc, std_uloc, gen_oneD, 
				    std_oneD, gen_csvC, std_csvC);
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: NHP %04d - HP Contractions: %f sec\n",
		 is+1,t2-t1);

      //-low-precision
      t1 = MPI_Wtime();
      oneEndTrick_w_One_Der<double>(*x_LP,*tmp3,*tmp4,param, 
				    gen_uloc_LP, std_uloc_LP, gen_oneD_LP, 
				    std_oneD_LP, gen_csvC_LP, std_csvC_LP); 
      t2 = MPI_Wtime();
      printfQuda("TIME_REPORT: NHP %04d - LP Contractions: %f sec\n",
		 is+1,t2-t1);
      
      t4 = MPI_Wtime();
      printfQuda("### TIME_REPORT: NHP %04d - Finished in %f sec\n",
		 is+1,t4-t3);
      //-------------------------------------------------      

      
      // FFT and copy to write buffers
      //-------------------------------------------------      
      if( (is+1)%Nd == 0){
	t1 = MPI_Wtime();
	if(GK_nProc[2]==1){
	  // Scalar
	  doCudaFFT_v2<double>(std_uloc   ,tmp_loop);
	  copyLoopToWriteBuf(buf_std_uloc_HP,tmp_loop,iPrint,
			     info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(std_uloc_LP,tmp_loop);
	  copyLoopToWriteBuf(buf_std_uloc_LP,tmp_loop,iPrint,
			     info.Q_sq,Nmoms,mom);

	  // dOp
	  doCudaFFT_v2<double>(gen_uloc   ,tmp_loop);
	  copyLoopToWriteBuf(buf_gen_uloc_HP,tmp_loop,iPrint,
			     info.Q_sq,Nmoms,mom);
	  doCudaFFT_v2<double>(gen_uloc_LP,tmp_loop);
	  copyLoopToWriteBuf(buf_gen_uloc_LP,tmp_loop,iPrint,
			     info.Q_sq,Nmoms,mom);
	  
	  for(int mu = 0 ; mu < 4 ; mu++){
	    // Loops
	    doCudaFFT_v2<double>(std_oneD[mu]   ,tmp_loop);
	    copyLoopToWriteBuf(buf_std_oneD_HP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom);
	    doCudaFFT_v2<double>(std_oneD_LP[mu],tmp_loop);
	    copyLoopToWriteBuf(buf_std_oneD_LP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom);
	    
	    // LoopsCv
	    doCudaFFT_v2<double>(std_csvC[mu]   ,tmp_loop);
	    copyLoopToWriteBuf(buf_std_csvC_HP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom); 
	    doCudaFFT_v2<double>(std_csvC_LP[mu],tmp_loop);
	    copyLoopToWriteBuf(buf_std_csvC_LP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom);

	    // LpsDw
	    doCudaFFT_v2<double>(gen_oneD[mu]   ,tmp_loop);
	    copyLoopToWriteBuf(buf_gen_oneD_HP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom); 
	    doCudaFFT_v2<double>(gen_oneD_LP[mu],tmp_loop);
	    copyLoopToWriteBuf(buf_gen_oneD_LP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom);

	    // LpsDwCv
	    doCudaFFT_v2<double>(gen_csvC[mu]   ,tmp_loop);
	    copyLoopToWriteBuf(buf_gen_csvC_HP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom); 
	    doCudaFFT_v2<double>(gen_csvC_LP[mu],tmp_loop);
	    copyLoopToWriteBuf(buf_gen_csvC_LP[mu],tmp_loop,
			       iPrint,info.Q_sq,Nmoms,mom);
	  }
	}
	else if(GK_nProc[2]>1){
	  performFFT<double>(buf_std_uloc_HP, std_uloc, iPrint, Nmoms,momQsq);
	  performFFT<double>(buf_std_uloc_LP,std_uloc_LP,iPrint,Nmoms,momQsq);
	  performFFT<double>(buf_gen_uloc_HP, gen_uloc, iPrint, Nmoms,momQsq);
	  performFFT<double>(buf_gen_uloc_LP,gen_uloc_LP,iPrint,Nmoms,momQsq);
	  
	  for(int mu=0;mu<4;mu++){
	    performFFT<double>(buf_std_oneD_HP[mu], std_oneD[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_std_oneD_LP[mu], std_oneD_LP[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_std_csvC_HP[mu], std_csvC[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_std_csvC_LP[mu], std_csvC_LP[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_oneD_HP[mu], gen_oneD[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_oneD_LP[mu], gen_oneD_LP[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_csvC_HP[mu], gen_csvC[mu], 
			       iPrint, Nmoms, momQsq);
	    performFFT<double>(buf_gen_csvC_LP[mu], gen_csvC_LP[mu], 
			       iPrint, Nmoms, momQsq);
	  }
	}
	t2 = MPI_Wtime();
	printfQuda("Loops for NHP = %04d FFT'ed and copied to write buffers in %f sec\n",is+1,t2-t1);
	iPrint++;
      }//-if (is+1)
    } // close loop over noise vectors
    
    //-Write the high-precision part
    t1 = MPI_Wtime();
    sprintf(loop_stoch_fname,"%s_stoch_TSM_MG_HighPrec",loopInfo.loop_fname);
    if(LoopFileFormat==ASCII_FORM){ 
      // Write the loops in ASCII format
      // Scalar
      writeLoops_ASCII(buf_std_uloc_HP, loop_stoch_fname, loopInfo, momQsq, 
		       0, 0, stoch_part, useTSM, HighPrecSum);
      // dOp      
      writeLoops_ASCII(buf_gen_uloc_HP, loop_stoch_fname, loopInfo, momQsq, 
		       1, 0, stoch_part, useTSM, HighPrecSum);  
      for(int mu = 0 ; mu < 4 ; mu++){
	// Loops
	writeLoops_ASCII(buf_std_oneD_HP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 2, mu, stoch_part, useTSM, HighPrecSum);
	// LoopsCv 
	writeLoops_ASCII(buf_std_csvC_HP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 3, mu, stoch_part, useTSM, HighPrecSum); 
	// LpsDw
	writeLoops_ASCII(buf_gen_oneD_HP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 4, mu, stoch_part, useTSM, HighPrecSum); 
	// LpsDwCv
	writeLoops_ASCII(buf_gen_csvC_HP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 5, mu, stoch_part, useTSM, HighPrecSum); 
      }
    }
    else if(LoopFileFormat==HDF5_FORM){ 
      // Write the loops in HDF5 format
      writeLoops_HDF5(buf_std_uloc, buf_gen_uloc, buf_std_oneD, 
		      buf_std_csvC, buf_gen_oneD, buf_gen_csvC, 
		      loop_stoch_fname, loopInfo, momQsq, 
		      stoch_part, useTSM, HighPrecSum);
    }
    t2 = MPI_Wtime();
    printfQuda("Writing the high-precision loops for MG completed in %f sec.\n",t2-t1);
    
    //-Write the low-precision part
    t1 = MPI_Wtime();
    sprintf(loop_stoch_fname,"%s_stoch_TSM_MG_LowPrec",loopInfo.loop_fname);
    if(LoopFileFormat==ASCII_FORM){ // Write the loops in ASCII format
      writeLoops_ASCII(buf_std_uloc_LP, loop_stoch_fname, loopInfo, momQsq, 
		       0, 0, stoch_part, useTSM, HighPrecSum); // Scalar
      writeLoops_ASCII(buf_gen_uloc_LP, loop_stoch_fname, loopInfo, momQsq, 
		       1, 0, stoch_part, useTSM, HighPrecSum); // dOp
      for(int mu = 0 ; mu < 4 ; mu++){
	writeLoops_ASCII(buf_std_oneD_LP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 2, mu, stoch_part, useTSM, HighPrecSum); // Loops
	writeLoops_ASCII(buf_std_csvC_LP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 3, mu, stoch_part, useTSM, HighPrecSum); // LoopsCv
	writeLoops_ASCII(buf_gen_oneD_LP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 4, mu, stoch_part, useTSM, HighPrecSum); // LpsDw
	writeLoops_ASCII(buf_gen_csvC_LP[mu], loop_stoch_fname, loopInfo, 
			 momQsq, 5, mu, stoch_part, useTSM, HighPrecSum); // LpsDwCv
      }
    }
    else if(LoopFileFormat==HDF5_FORM){ // Write the loops in HDF5 format
      writeLoops_HDF5(buf_std_uloc_LP, buf_gen_uloc_LP, buf_std_oneD_LP, 
		      buf_std_csvC_LP, buf_gen_oneD_LP, buf_gen_csvC_LP, 
		      loop_stoch_fname, loopInfo, momQsq, 
		      stoch_part, useTSM, HighPrecSum);
    }
    t2 = MPI_Wtime();
    printfQuda("Writing the low-precision loops for MG completed in %f sec.\n",t2-t1);
  }//-useTSM  
  
  //======================================================================//
  //================ M E M O R Y   C L E A N - U P =======================// 
  //======================================================================//
  
  //-Free the Cuda loop buffers
  cudaFreeHost(std_uloc);
  cudaFreeHost(gen_uloc);
  cudaFreeHost(tmp_loop);
  for(int mu = 0 ; mu < 4 ; mu++){
    cudaFreeHost(std_oneD[mu]);
    cudaFreeHost(gen_oneD[mu]);
    cudaFreeHost(std_csvC[mu]);
    cudaFreeHost(gen_csvC[mu]);
  }
  free(std_oneD);
  free(gen_oneD);
  free(std_csvC);
  free(gen_csvC);
  //---------------------------

  //-Free loop write buffers
  free(buf_std_uloc);
  free(buf_gen_uloc);
  for(int mu = 0 ; mu < 4 ; mu++){
    free(buf_std_oneD[mu]);
    free(buf_std_csvC[mu]);
    free(buf_gen_oneD[mu]);
    free(buf_gen_csvC[mu]);
  }
  free(buf_std_oneD);
  free(buf_std_csvC);
  free(buf_gen_oneD);
  free(buf_gen_csvC);
  //---------------------------

  //-Free the extra buffers if using TSM
  if(useTSM){
    free(buf_std_uloc_LP); free(buf_std_uloc_HP);
    free(buf_gen_uloc_LP); free(buf_gen_uloc_HP);
    for(int mu = 0 ; mu < 4 ; mu++){
      free(buf_std_oneD_LP[mu]); free(buf_std_oneD_HP[mu]);
      free(buf_std_csvC_LP[mu]); free(buf_std_csvC_HP[mu]);
      free(buf_gen_oneD_LP[mu]); free(buf_gen_oneD_HP[mu]);
      free(buf_gen_csvC_LP[mu]); free(buf_gen_csvC_HP[mu]);
    }
    free(buf_std_oneD_LP); free(buf_std_oneD_HP);
    free(buf_std_csvC_LP); free(buf_std_csvC_HP);
    free(buf_gen_oneD_LP); free(buf_gen_oneD_HP);
    free(buf_gen_csvC_LP); free(buf_gen_csvC_HP);

    cudaFreeHost(std_uloc_LP);
    cudaFreeHost(gen_uloc_LP);
    for(int mu = 0 ; mu < 4 ; mu++){
      cudaFreeHost(std_oneD_LP[mu]);
      cudaFreeHost(gen_oneD_LP[mu]);
      cudaFreeHost(std_csvC_LP[mu]);
      cudaFreeHost(gen_csvC_LP[mu]);
    }
    free(std_oneD_LP);
    free(gen_oneD_LP);
    free(std_csvC_LP);
    free(gen_csvC_LP);
  }
  //------------------------------------


  //-Free the momentum matrices
  for(int ip=0; ip<SplV; ip++) free(mom[ip]);
  free(mom);
  for(int ip=0;ip<Nmoms;ip++) free(momQsq[ip]);
  free(momQsq);
  //---------------------------

  free(input_vector);
  free(output_vector);
  gsl_rng_free(rNum);
  delete d;
  delete dSloppy;
  delete dPre;
  delete K_guess;
  delete K_vector;
  delete K_gauge;
  delete h_x;
  delete h_b;
  delete x;
  delete b;
  delete tmp3;
  delete tmp4;
  if(useTSM){
    delete x_LP;
  }
  popVerbosity();
  saveTuneCache();
  profileInvert.TPSTOP(QUDA_PROFILE_TOTAL);
}

*/
