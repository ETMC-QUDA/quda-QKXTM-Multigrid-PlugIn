#include <cufft.h>
#include <gsl/gsl_rng.h>
#include <contractQuda.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM_Kepler.h>
#include <qudaQKXTM_Kepler_utils.h>
#include <dirac_quda.h>
#include <errno.h>
#include <mpi.h>
#include <limits>
//#include <mkl.h> //QXKTM: FIXME
#include <cblas.h>
#include <common.h>
#include <omp.h>
#include <hdf5.h>
 
#define PI 3.141592653589793
 
//using namespace quda;
extern Topology *default_topo;
 
/* Block for global variables */
extern float GK_deviceMemory;
extern int GK_nColor;
extern int GK_nSpin;
extern int GK_nDim;
extern int GK_strideFull;
extern double GK_alphaAPE;
extern double GK_alphaGauss;
extern int GK_localVolume;
extern int GK_totalVolume;
extern int GK_nsmearAPE;
extern int GK_nsmearGauss;
extern bool GK_dimBreak[QUDAQKXTM_DIM];
extern int GK_localL[QUDAQKXTM_DIM];
extern int GK_totalL[QUDAQKXTM_DIM];
extern int GK_nProc[QUDAQKXTM_DIM];
extern int GK_plusGhost[QUDAQKXTM_DIM];
extern int GK_minusGhost[QUDAQKXTM_DIM];
extern int GK_surface3D[QUDAQKXTM_DIM];
extern bool GK_init_qudaQKXTM_Kepler_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;


#include <sys/stat.h>
#include <unistd.h>
#define TIMING_REPORT

static bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}

//Calculates the average palquette trace of the gauge field,
//passed as 4 (mu) pointers to pointers for each 
//spacetime dimension.
void testPlaquette(void **gauge){
  QKXTM_Gauge_Kepler<float> *gauge_object = 
    new QKXTM_Gauge_Kepler<float>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();
  delete gauge_object;

  QKXTM_Gauge_Kepler<double> *gauge_object_2 = 
    new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object_2->printInfo();
  gauge_object_2->packGauge(gauge);
  gauge_object_2->loadGauge();
  gauge_object_2->calculatePlaq();
  delete gauge_object_2;
}

//Performs Gaussian smearing on a point source located
//at (0,0,0,0) using links from the gauge field,
//passed as 4 (mu) pointers to pointers for each 
//spacetime dimension. Output is printed to 
//stdout.
void testGaussSmearing(void **gauge){
  QKXTM_Gauge_Kepler<double> *gauge_object = 
    new QKXTM_Gauge_Kepler<double>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();

  QKXTM_Vector_Kepler<double> *vecIn = 
    new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);
  QKXTM_Vector_Kepler<double> *vecOut = 
    new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

  void *input_vector = malloc(GK_localVolume*4*3*2*sizeof(double));
  *((double*) input_vector) = 1.;
  vecIn->packVector((double*) input_vector);
  vecIn->loadVector();
  vecOut->gaussianSmearing(*vecIn,*gauge_object);
  vecOut->download();
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      printf("%+e %+e\n",
	     vecOut->H_elem()[mu*3*2+c1*2+0],
	     vecOut->H_elem()[mu*3*2+c1*2+1]);

  delete vecOut;
  delete gauge_object;
}

//SOURCE_T RANDOM: Constructs a Z_4 random source using 
//                 gsl as RNG.
//SOURCE_T UNITY:  Constructs a momentum source with p=0.
template <typename Float>
void getStochasticRandomSource(void *spinorIn, gsl_rng *rNum, 
			       SOURCE_T source_type){

  memset(spinorIn,0,GK_localVolume*12*2*sizeof(Float));
  for(int i = 0; i<GK_localVolume*12; i++){
    int randomNumber = gsl_rng_uniform_int(rNum, 4);

    if(source_type==UNITY){
      ((Float*) spinorIn)[i*2] = 1.0;
      ((Float*) spinorIn)[i*2+1] = 0.0;
    }
    else if(source_type==RANDOM){
      switch  (randomNumber){
      case 0:
	((Float*) spinorIn)[i*2] = 1.;
	break;
      case 1:
	((Float*) spinorIn)[i*2] = -1.;
	break;
      case 2:
	((Float*) spinorIn)[i*2+1] = 1.;
	break;
      case 3:
	((Float*) spinorIn)[i*2+1] = -1.;
	break;
      }
    }
    else{
      errorQuda("Source type not set correctly!! Aborting.\n");
    }
  }
}

static int** allocateMomMatrix(int Q_sq){
  int **mom = (int **) malloc(sizeof(int*)*GK_localL[0]*GK_localL[1]*GK_localL[2]);
  if(mom == NULL) errorQuda("Error allocate memory for momenta\n");
  for(int ip=0; ip<GK_localL[0]*GK_localL[1]*GK_localL[2]; ip++) {
    mom[ip] = (int *) malloc(sizeof(int)*3);
    if(mom[ip] == NULL) errorQuda("Error allocate memory for momenta\n");
  }
  int momIdx       = 0;
  int totMom       = 0;
  
  for(int pz = 0; pz < GK_localL[2]; pz++)
    for(int py = 0; py < GK_localL[1]; py++)
      for(int px = 0; px < GK_localL[0]; px++){
	if      (px < GK_localL[0]/2)
	  mom[momIdx][0]   = px;
	else
	  mom[momIdx][0]   = px - GK_localL[0];

	if      (py < GK_localL[1]/2)
	  mom[momIdx][1]   = py;
	else
	  mom[momIdx][1]   = py - GK_localL[1];

	if      (pz < GK_localL[2]/2)
	  mom[momIdx][2]   = pz;
	else
	  mom[momIdx][2]   = pz - GK_localL[2];

	if((mom[momIdx][0]*mom[momIdx][0]+
	    mom[momIdx][1]*mom[momIdx][1]+
	    mom[momIdx][2]*mom[momIdx][2]) <= Q_sq) totMom++;
	momIdx++;
      }
  return mom;
}



template <typename Float>
void doCudaFFT_v2(void *cnIn, void *cnOut){
  static cufftHandle fftPlan;
  static int init = 0;
  int nRank[3] = {GK_localL[0], GK_localL[1], GK_localL[2]};
  const int Vol = GK_localL[0]*GK_localL[1]*GK_localL[2];
  static cudaStream_t     streamCuFFT;
  cudaStreamCreate(&streamCuFFT);

  if(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 
		   1, Vol, CUFFT_Z2Z, 16*GK_localL[3]) != CUFFT_SUCCESS) 
    errorQuda("Error in the FFT!!!\n");

  cufftSetCompatibilityMode(fftPlan, CUFFT_COMPATIBILITY_FFTW_PADDING);
  cufftSetStream           (fftPlan, streamCuFFT);
  checkCudaError();
  void* ctrnS;
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*Vol*GK_localL[3])) == 
     cudaErrorMemoryAllocation) errorQuda("Error with memory allocation\n");

  cudaMemcpy(ctrnS, cnIn, sizeof(Float)*32*Vol*GK_localL[3], 
	     cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnOut, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);

  cudaFree(ctrnS);
  cufftDestroy            (fftPlan);
  cudaStreamDestroy       (streamCuFFT);
  checkCudaError();
}


//-C.K. Added this function for convenience, when writing the 
//loops in the new ASCII format of the HDF5 format
void createLoopMomenta(int **mom, int **momQsq, int Q_sq, int Nmoms){

  int momIdx = 0;
  int totMom = 0;

  for(int pz = 0; pz < GK_totalL[2]; pz++)
    for(int py = 0; py < GK_totalL[1]; py++)
      for(int px = 0; px < GK_totalL[0]; px++){
	if(px < GK_totalL[0]/2)
	  mom[momIdx][0]   = px;
	else
	  mom[momIdx][0]   = px - GK_totalL[0];

	if(py < GK_totalL[1]/2)
	  mom[momIdx][1]   = py;
	else
	  mom[momIdx][1]   = py - GK_totalL[1];

	if(pz < GK_totalL[2]/2)
	  mom[momIdx][2]   = pz;
	else
	  mom[momIdx][2]   = pz - GK_totalL[2];

	if((mom[momIdx][0]*mom[momIdx][0]+
	    mom[momIdx][1]*mom[momIdx][1]+
	    mom[momIdx][2]*mom[momIdx][2])<=Q_sq){
	  if(totMom>=Nmoms) 
	    errorQuda("Inconsistency in Number of Momenta Requested\n");
	  for(int i=0;i<3;i++) momQsq[totMom][i] = mom[momIdx][i];
	  printfQuda("Mom %d: %+d %+d %+d\n", totMom,
		     momQsq[totMom][0],
		     momQsq[totMom][1],
		     momQsq[totMom][2]);
	  totMom++;
	}

	momIdx++;
      }

  if(totMom<=Nmoms-1) 
    warningQuda("Created momenta (%d) less than Requested (%d)!!\n",
		totMom,Nmoms);
  
}

//-C.K. Function which performs the Fourier Transform
template<typename Float>
void performFFT(Float *outBuf, void *inBuf, int iPrint, 
		int Nmoms, int **momQsq){
  
  int lx=GK_localL[0];
  int ly=GK_localL[1];
  int lz=GK_localL[2];
  int lt=GK_localL[3];
  int LX=GK_totalL[0];
  int LY=GK_totalL[1];
  int LZ=GK_totalL[2];
  long int SplV = lx*ly*lz;

  double two_pi = 4.0*asin(1.0);
  int z_coord = comm_coord(2);

  Float *sum = (Float*) malloc(2*16*Nmoms*lt*sizeof(Float));
  if(sum == NULL) errorQuda("performManFFT: Allocation of sum buffer failed.\n");
  memset(sum,0,2*16*Nmoms*lt*sizeof(Float));

  for(int ip=0;ip<Nmoms;ip++){
    int px = momQsq[ip][0];
    int py = momQsq[ip][1];
    int pz = momQsq[ip][2];

    int v = 0;
    for(int z=0;z<lz;z++){     //-For z-direction we must have lz
      int zg = z+z_coord*lz;
      for(int y=0;y<ly;y++){   //-Here either ly or LY is the same because we don't split in y (for now)
	for(int x=0;x<lx;x++){ //-The same here
	  Float expn = two_pi*( px*x/(Float)LX + py*y/(Float)LY + pz*zg/(Float)LZ );
	  Float phase[2];
	  phase[0] =  cos(expn);
	  phase[1] = -sin(expn);

	  for(int t=0;t<lt;t++){
	    for(int gm=0;gm<16;gm++){
	      sum[0 + 2*ip + 2*Nmoms*t + 2*Nmoms*lt*gm] += 
		((Float*)inBuf)[0+2*v+2*SplV*t+2*SplV*lt*gm]*phase[0] - 
		((Float*)inBuf)[1+2*v+2*SplV*t+2*SplV*lt*gm]*phase[1];
	      sum[1 + 2*ip + 2*Nmoms*t + 2*Nmoms*lt*gm] += 
		((Float*)inBuf)[0+2*v+2*SplV*t+2*SplV*lt*gm]*phase[1] + 
		((Float*)inBuf)[1+2*v+2*SplV*t+2*SplV*lt*gm]*phase[0];
	    }//-gm
	  }//-t
	  
	  v++;
	}//-x
      }//-y
    }//-z
  }//-ip

  if(typeid(Float)==typeid(float))  MPI_Reduce(sum, &(outBuf[2*Nmoms*lt*16*iPrint]), 2*Nmoms*lt*16, MPI_FLOAT , MPI_SUM, 0, GK_spaceComm);
  if(typeid(Float)==typeid(double)) MPI_Reduce(sum, &(outBuf[2*Nmoms*lt*16*iPrint]), 2*Nmoms*lt*16, MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);

  free(sum);
}

template<typename Float>
void copyLoopToWriteBuf(Float *writeBuf, void *tmpBuf, int iPrint, 
			int Q_sq, int Nmoms, int **mom){

  if(GK_nProc[2]==1){
    long int SplV = GK_localL[0]*GK_localL[1]*GK_localL[2];
    int imom = 0;
    
    for(int ip=0; ip < SplV; ip++){
      if ((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= Q_sq){
	for(int lt=0; lt < GK_localL[3]; lt++){
	  for(int gm=0; gm<16; gm++){
	    writeBuf[0+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint] = ((Float*)tmpBuf)[0+2*ip+2*SplV*lt+2*SplV*GK_localL[3]*gm];
	    writeBuf[1+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint] = ((Float*)tmpBuf)[1+2*ip+2*SplV*lt+2*SplV*GK_localL[3]*gm];
	  }//-gm
	}//-lt
	imom++;
      }//-if
    }//-ip
  }
  else errorQuda("copyLoopToWriteBuf: This function does not support more than 1 GPU in the z-direction\n");

}

/* Quarantined code
template <typename Float>
void getStochasticRandomSource(void *spinorIn, gsl_rng *rNum){
  memset(spinorIn,0,GK_localVolume*12*2*sizeof(Float));

  for(int i = 0; i<GK_localVolume*12; i++){

    //- Unity sources
    //    ((Float*) spinorIn)[i*2] = 1.0;
    //    ((Float*) spinorIn)[i*2+1] = 0.0;

    //- Random sources
    int randomNumber = gsl_rng_uniform_int(rNum, 4);
    switch  (randomNumber)
      {
      case 0:
	((Float*) spinorIn)[i*2] = 1.;
	break;
      case 1:
	((Float*) spinorIn)[i*2] = -1.;
	break;
      case 2:
	((Float*) spinorIn)[i*2+1] = 1.;
	break;
      case 3:
	((Float*) spinorIn)[i*2+1] = -1.;
	break;
      }

  }//-for

}
*/

/* Quarantined Code
template <typename Float>
void doCudaFFT(void *cnRes_gv, void *cnRes_vv, void *cnResTmp_gv,void *cnResTmp_vv){
  static cufftHandle      fftPlan;
  static int              init = 0;
  int                     nRank[3]         = {GK_localL[0], GK_localL[1], GK_localL[2]};
  const int               Vol              = GK_localL[0]*GK_localL[1]*GK_localL[2];
  static cudaStream_t     streamCuFFT;
  cudaStreamCreate(&streamCuFFT);

  if(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*GK_localL[3]) != CUFFT_SUCCESS) errorQuda("Error in the FFT!!!\n");
  cufftSetCompatibilityMode       (fftPlan, CUFFT_COMPATIBILITY_FFTW_PADDING);
  cufftSetStream                  (fftPlan, streamCuFFT);
  checkCudaError();
  void* ctrnS;
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*Vol*GK_localL[3])) == cudaErrorMemoryAllocation) errorQuda("Error with memory allocation\n");

  cudaMemcpy(ctrnS, cnRes_vv, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnResTmp_vv, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);

  cudaMemcpy(ctrnS, cnRes_gv, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyHostToDevice);
  if(typeid(Float) == typeid(double))if(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  if(typeid(Float) == typeid(float))if(cufftExecC2C(fftPlan, (float2 *) ctrnS, (float2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS) errorQuda("Error run cudafft\n");
  cudaMemcpy(cnResTmp_gv, ctrnS, sizeof(Float)*32*Vol*GK_localL[3], cudaMemcpyDeviceToHost);


  cudaFree(ctrnS);
  cufftDestroy            (fftPlan);
  cudaStreamDestroy       (streamCuFFT);
  checkCudaError();
}
*/

