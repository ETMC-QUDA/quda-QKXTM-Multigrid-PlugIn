#include <cufft.h>
#include <gsl/gsl_rng.h>
#include <contractQuda.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM.h>
#include <qudaQKXTM_utils.h>
#include <dirac_quda.h>
#include <errno.h>
#include <mpi.h>
#include <limits>

#ifdef HAVE_MKL
#include <mkl.h>
#endif

#ifdef HAVE_OPENBLAS
#include <cblas.h>
#include <common.h>
#endif

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
extern bool GK_init_qudaQKXTM_flag;
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

//Though only some template forward declarations
//are needed presently, future development may require the
//others, and are therefore placed here for convenience
// template  class QKXTM_Field<double>;
// template  class QKXTM_Gauge<double>;
// template  class QKXTM_Vector<double>;
// template  class QKXTM_Propagator<double>;
// template  class QKXTM_Propagator3D<double>;
// template  class QKXTM_Vector3D<double>;

// template  class QKXTM_Field<float>;
// template  class QKXTM_Gauge<float>;
// template  class QKXTM_Vector<float>;
// template  class QKXTM_Propagator<float>;
// template  class QKXTM_Propagator3D<float>;
// template  class QKXTM_Vector3D<float>;

static bool exists_file (const char* name) {
  return ( access( name, F_OK ) != -1 );
}

//Calculates the average palquette trace of the gauge field,
//passed as 4 (mu) pointers to pointers for each 
//spacetime dimension.
void testPlaquette(void **gauge){
  QKXTM_Gauge<float> *gauge_object = 
    new QKXTM_Gauge<float>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();
  delete gauge_object;

  QKXTM_Gauge<double> *gauge_object_2 = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
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
  QKXTM_Gauge<double> *gauge_object = 
    new QKXTM_Gauge<double>(BOTH,GAUGE);
  gauge_object->printInfo();
  gauge_object->packGauge(gauge);
  gauge_object->loadGauge();
  gauge_object->calculatePlaq();

  QKXTM_Vector<double> *vecIn = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  QKXTM_Vector<double> *vecOut = 
    new QKXTM_Vector<double>(BOTH,VECTOR);

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


/*
# There is an unsigned integer "k" running from 1 unitl ...
# From this integer we can specify several important quantities regarding the coloring
# The total number of colors is given by N_{hc} = 2 * 2^{d(k-1)} where d is the number of dimensions
# The distance seperating neighbors carrying the same color is D=2^k
# The size of the elementary coloring is given L_u=2^{k-1}
# A condition must be fulfilled in order to be able to do the coloring for a specific k
# The condition must be that the number of blocks in each direction must be even
# And that Ls%(2Lu)=0 and Lt%(2Lu)=0
 */


/*
Description: This function computes the colors for the elementary color block
Inputs:
lc: Pointer to the array where we want to store the colors
Nc: The number of colors we want to put in the block
Lu: The extent of the color block
d: Dimension, either 2 or 3
 */

void fcb(unsigned short int lc[][2], const int Nc,const  int Lu, const int d){
  if(d==3)
    for(int i = 0 ; i < Lu; i++)
      for(int j = 0 ; j < Lu ; j++)
	for(int k = 0 ; k < Lu ; k++)
	  for(int eo = 0 ; eo < 2 ; eo++)
	    lc[i*Lu*Lu+j*Lu+k][eo] = (i*Lu*Lu+j*Lu+k)*2+eo;
  if(d==4)
    for(int i = 0 ; i < Lu; i++)
      for(int j = 0 ; j < Lu ; j++)
	for(int k = 0 ; k < Lu ; k++)
	  for(int l = 0 ; l < Lu ; l++)
	    for(int eo = 0 ; eo < 2 ; eo++)
	      lc[i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l][eo] = (i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l)*2+eo;
}

/*
  Description: This function gets a lexicographic index and return a position vector
x: The pointer to the vector
idx: the index
L: The spatial total extent (temporal dimension is the slowest in memory)
 */
void get_ind2Vec(int *x, const long int idx, const long int *L, const int d){
  
  if(d == 3){
    x[2]=idx/(L[0]*L[1]);
    x[1]=idx/L[0]-x[2]*L[1];
    x[0]=idx - x[2]*L[0]*L[1] - x[1]*L[0];
    }
  if(d==4){
    x[3]=idx/(L[0]*L[1]*L[2]);
    x[2]=idx/(L[0]*L[1]) - x[3]*L[2];
    x[1]=idx/L[0] - x[3]*L[1]*L[2] - x[2]*L[1];
    x[0]=idx-(x[3]*L[0]*L[1]*L[2]+x[2]*L[0]*L[1]+x[1]*L[0]);
  }
}

/*
  Description: Does the opposite of get_ind2Vec
 */

inline long int get_vec2Idx(const int *x, const long int *L, const int d){
  if (d==3)
    return x[0]+x[1]*L[0]+x[2]*L[0]*L[1];
  else
    return x[0]+x[1]*L[0]+x[2]*L[0]*L[1]+x[3]*L[0]*L[1]*L[2]; 
}

void create_hch_coloring(unsigned short int *Vc, long int lenVc, int Nc, int Lu, int d){
  unsigned short int (*lc)[2] = (unsigned short int(*)[2]) malloc(Nc*sizeof(unsigned short int));
  fcb(lc,Nc,Lu,d);

  int *x = (int*) malloc(d*sizeof(int));
  long int *GL = (long int *) malloc(d*sizeof(long int));
  long int *lu = (long int*) malloc(d*sizeof(long int));
  int *bx = (int*) malloc(d*sizeof(int));
  int *lx = (int*) malloc(d*sizeof(int));

  for(int i = 0 ; i < d ; i++) GL[i] = GK_localL[i];
  for(int i = 0 ; i < d ; i++) lu[i] = Lu;
  int eo;
  for(long int i=0; i < lenVc; i++){
    get_ind2Vec(x,i,GL,d);
    for(int j = 0 ; j < d ; j++)
      bx[j] =  x[j]/Lu; // find the position of each block
    eo=0;
    for(int j = 0 ; j < d ; j++) eo += bx[j]; // find if the block is even or odd
    eo=eo & 1;
    for(int j = 0 ; j < d ; j++)
      lx[j] = x[j] - Lu*bx[j]; // find the position inside block
    Vc[i] = lc[get_vec2Idx(lx,lu,d)][eo];
  }
  
  free(x);
  free(GL);
  free(lu);
  free(bx);
  free(lx);
}


//#define CHECK_COLORING

#ifdef CHECK_COLORING

inline static int brt(int x, int L){
  int y=x;
  if (y >= L)
    y=y%L;
  if (y < 0)
    y=y+L;
  return y;
}

void check_coloring(unsigned short int *Vc, int D, int d){
  long int *GL = (long int *) malloc(d*sizeof(long int));
  for(int i = 0 ; i < d ; i++) GL[i] = GK_localL[i];
  int *xx = (int*) malloc(d*sizeof(int));

  if(d==3){
    for(int z = 0 ; z < GK_totalL[2] ; z++)
      for(int y = 0 ; y < GK_totalL[1] ; y++)
	for(int x = 0 ; x < GK_totalL[0] ; x++){
	  xx[0] = x; xx[1] = y; xx[2] = z;
	  int c1 = Vc[get_vec2Idx(xx, GL,d)];
	  for(int dx = -D+1 ; dx < D ; dx++)
	    for(int dy = -D+1 ; dy < D ; dy++)
	      for(int dz = -D+1 ; dz < D ; dz++){
		int ds = abs(dx) + abs(dy) + abs(dz);
		if ((ds<D) && (ds != 0)){
		  int xn = x + dx;
		  xn = brt(xn,GK_totalL[0]);
		  int yn = y + dy;
		  yn = brt(yn,GK_totalL[1]);
		  int zn = z + dz;
		  zn = brt(zn,GK_totalL[2]);
		  xx[0] = xn; xx[1] = yn; xx[2] = zn;
		  int c2 = Vc[get_vec2Idx(xx, GL,d)];
		  if(c1 == c2){
		    errorQuda("Mistake found in the coloring with (%d,%d,%d) and (%d,%d,%d)",x,y,z,xn,yn,zn);
		  }
		}
	      }
	}
  }
  else{
    for(int t = 0 ; t < GK_totalL[3] ; t++)
      for(int z = 0 ; z < GK_totalL[2] ; z++)
	for(int y = 0 ; y < GK_totalL[1] ; y++)
	  for(int x = 0 ; x < GK_totalL[0] ; x++){
	    xx[0] = x; xx[1] = y; xx[2] = z; xx[3] = t;
	    int c1 = Vc[get_vec2Idx(xx, GL,d)];
	    for(int dx = -D+1 ; dx < D ; dx++)
	      for(int dy = -D+1 ; dy < D ; dy++)
		for(int dz = -D+1 ; dz < D ; dz++)
		  for(int dt = -D+1 ; dt < D ; dt++){
		    int ds = abs(dx) + abs(dy) + abs(dz) + abs(dt);
		    if ((ds<D) && (ds != 0)){
		      int xn = x + dx;
		      xn = brt(xn,GK_totalL[0]);
		      int yn = y + dy;
		      yn = brt(yn,GK_totalL[1]);
		      int zn = z + dz;
		      zn = brt(zn,GK_totalL[2]);
		      int tn = t + dt;
		      tn = brt(tn,GK_totalL[3]);
		      xx[0] = xn; xx[1] = yn; xx[2] = zn; xx[3] = tn;
		      int c2 = Vc[get_vec2Idx(xx, GL,d)];
		      if(c1 == c2){
			printfQuda("Colors (%d,%d)\n",c1,c2);
			errorQuda("Mistake found in the coloring with (%d,%d,%d,%d) and (%d,%d,%d,%d)",x,y,z,t,xn,yn,zn,tn);
		      }
		    }
		  }
	  }
  }

  free(xx);
  free(GL);
  printfQuda("Check for coloring passed!!!!");
}

#endif


// k is the integer number related with the distance
// d is the number of dimensions 2 or 3
unsigned short int* hch_coloring(int k, int d){
  if ((d != 3) && (d != 4) )
    errorQuda("Only 3 and 4 dimensions of coloring are allowed");
  if( k < 1)
    errorQuda("k must be greater than 1");
  int Nc = 2*pow(2,d*(k-1));
  int D = pow(2,k);
  int Lu = pow(2,k-1);
  printfQuda("Number of colors for hierarchical probing is %d\n",Nc);
  printfQuda("Distance of neigbors is %d\n",D);
  printfQuda("The extent of the symmetric color unit block is %d\n",Lu);
  for(int i = 0 ; i < d ; i++)
    if( (GK_localL[i] % (2*Lu)) != 0 )
      errorQuda("2*Lu cannot fit in the local lattice extent");
  if (Nc > 65536)
    errorQuda("Exceeded maximum number of colors");
  unsigned short int *Vc;
  
  long int lenVc;
  if(d==3)
    lenVc=GK_localL[0]*GK_localL[1]*GK_localL[2];
  else
    lenVc=GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  Vc = (unsigned short int *) malloc(sizeof(unsigned short int)*lenVc);
  create_hch_coloring(Vc, lenVc, Nc, Lu, d);
#ifdef CHECK_COLORING
  // check for colors is implemented only for 1 process
  for(int i = 0 ; i < d ; i++)
    if(GK_localL[i] != GK_totalL[i])
      errorQuda("Test coloring is available for only 1 MPI task");

  check_coloring(Vc,D,d);
#endif
  return Vc;
}


int HadamardElements(int i, int j){
  int sum=0;
    for(int k = 0 ; k < 32 ; k++){
      sum += (i%2)*(j%2);
      i=i>>1;
      j=j>>1;
    }
  if( (sum%2) ==0 )
    return 1;
  else
    return -1;
}


template <typename Float>
void get_probing4D_spinColor_dilution(void *temp_input_vector, void *input_vector, unsigned short int *Vc, int ih, int sc){
  memset(temp_input_vector,0,GK_localVolume*12*2*sizeof(Float));
  int c;
  int signProbing;
  for(int i = 0 ; i < GK_localVolume ; i++){
    c = Vc[i];
    int signProbing = HadamardElements(c,ih);
    for(int ri = 0 ; ri < 2 ; ri++)
      ((Float*)temp_input_vector)[i*12*2+sc*2+ri] = signProbing * ((Float*)input_vector)[i*12*2+sc*2+ri];
  }
}

template <typename Float>
void get_spinColor_dilution(void *temp_input_vector, void *input_vector, int sc){
  memset(temp_input_vector,0,GK_localVolume*12*2*sizeof(Float));
  for(int i = 0 ; i < GK_localVolume ; i++){
    for(int ri = 0 ; ri < 2 ; ri++)
      ((Float*)temp_input_vector)[i*12*2+sc*2+ri] = ((Float*)input_vector)[i*12*2+sc*2+ri];
  }
}

template <typename Float>
void get_probing4D_dilution(void *temp_input_vector, void *input_vector, unsigned short int *Vc, int ih){
  memset(temp_input_vector,0,GK_localVolume*12*2*sizeof(Float));
  int c;
  int signProbing;
  for(int i = 0 ; i < GK_localVolume ; i++){
    c = Vc[i];
    int signProbing = HadamardElements(c,ih);
    for(int sc = 0 ; sc < 12 ; sc++)
      for(int ri = 0 ; ri < 2 ; ri++)
	((Float*)temp_input_vector)[i*12*2+sc*2+ri] = signProbing * ((Float*)input_vector)[i*12*2+sc*2+ri];
  }
}


/* Quarantined code

template <typename Float>
void get_probing4D_spinColor_temporal_dilution(void *temp_input_vector, void *input_vector, unsigned short int *Vc, int ih, int sc, int it, int tDil){
  
  //Zero out temp vector
  memset(temp_input_vector,0,GK_localVolume*12*2*sizeof(Float));

  // GK_timeRank is the value of the MPI process in the T dimension
  // GK_timeSize is the total  MPI processes in the T dimension
  // GK_totalL[3] is the temporal extent of the lattice
  // GK_localL[3] is the temporal extent of on the MPI node
  // tDil is the nth lattice timeslice to be populated.
  // it is the current temporal block to be populated.

  // Must determine if the local i index is part of a timeslice
  // to be populated. Using integer division, the local index i
  // divided by the local spatial volume gives a floored integer
  // equal to the local temporal index: eg
  // for a local x,y,z,t of {4,4,4,8} and index of, say 361
  // t_loc = floor[361/(4**3)] = 5, so the local time index is 5.
  // The lattice time index t is therefore 
  // t = GK_timeRank * GK_localL[3] + t_loc
  // If (t-it)%(tDil) == 0, populate the array.
  
  int c;
  int signProbing;
  int t;
  int localSpaceVol = GK_localVolume/GK_localL[3];
  for(int i = 0 ; i < GK_localVolume ; i++){
    //get local t index
    t = i/localSpaceVol;
    //get global t index
    t += GK_timeRank*GK_localL[3];
    //Apply temporal blocking condition
    if((t-it)%(tDil) == 0) {
      c = Vc[i];
      int signProbing = HadamardElements(c,ih);
      for(int ri = 0 ; ri < 2 ; ri++)
	((Float*)temp_input_vector)[i*12*2+sc*2+ri] = signProbing * ((Float*)input_vector)[i*12*2+sc*2+ri];
    }
  }
}



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


/* Moved to qudaQKXTM_Loops.cpp 
//-C.K. This is a new function to print all the loops in ASCII format
template<typename Float>
void writeLoops_ASCII(Float *writeBuf, const char *Pref, 
		      qudaQKXTM_loopInfo loopInfo, 
		      int **momQsq, int type, 
		      int mu, bool exact_loop, 
		      bool useTSM, bool LowPrec){
  
  if(exact_loop && useTSM) errorQuda("writeLoops_ASCII: Got conflicting options - exact_loop AND useTSM.\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    FILE *ptr;
    char file_name[512];
    char *lpart,*ptrVal;
    int Nprint,Ndump;
    int Nmoms = loopInfo.Nmoms;

    if(exact_loop) Nprint = 1;
    else{
      if(useTSM){
	if(LowPrec){
	  Nprint = loopInfo.TSM_NprintLP;
	  Ndump  = loopInfo.TSM_NdumpLP; 
	}
	else{
	  Nprint = loopInfo.TSM_NprintHP;
	  Ndump  = loopInfo.TSM_NdumpHP; 
	}
      }
      else{
	Nprint = loopInfo.Nprint;
	Ndump  = loopInfo.Ndump;
      }
    }

    for(int iPrint=0;iPrint<Nprint;iPrint++){
      if(exact_loop || useTSM) asprintf(&ptrVal,"%d_%d", GK_nProc[3], GK_timeRank);
      else asprintf(&ptrVal,"%04d.%d_%d",(iPrint+1)*Ndump, GK_nProc[3], GK_timeRank);

      if(useTSM) sprintf(file_name, "%s_%s%04d_%s.loop.%s", Pref, LowPrec ? "NLP" : "NHP", (iPrint+1)*Ndump, loopInfo.loop_type[type], ptrVal);
      else sprintf(file_name, "%s_%s.loop.%s",Pref,loopInfo.loop_type[type],ptrVal);

      if(loopInfo.loop_oneD[type] && mu!=0) ptr = fopen(file_name,"a");
      else ptr = fopen(file_name,"w");
      if(ptr == NULL) errorQuda("Cannot open %s to write the loop\n",file_name);

      if(loopInfo.loop_oneD[type]){
	for(int ip=0; ip < Nmoms; ip++){
	  for(int lt=0; lt < GK_localL[3]; lt++){
	    int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	    for(int gm=0; gm<16; gm++){
	      fprintf(ptr, "%02d %02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, mu, momQsq[ip][0], momQsq[ip][1], momQsq[ip][2],
		      0.25*writeBuf[0+2*ip+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint],
		      0.25*writeBuf[1+2*ip+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint]);
	    }
	  }//-lt
	}//-ip
      }
      else{
	for(int ip=0; ip < Nmoms; ip++){
	  for(int lt=0; lt < GK_localL[3]; lt++){
	    int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	    for(int gm=0; gm<16; gm++){
	      fprintf(ptr, "%02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, momQsq[ip][0], momQsq[ip][1], momQsq[ip][2],
		      writeBuf[0+2*ip+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint],
		      writeBuf[1+2*ip+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint]);
	    }
	  }//-lt
	}//-ip
      }
    
      fclose(ptr);
    }//-iPrint
  }//-if GK_timeRank

}
*/

/* Moved to qudaQKXTM_Loops.cpp
//-C.K: Copy the HDF5 dataset chunk into writeBuf
template<typename Float>
void getLoopWriteBuf(Float *writeBuf, Float *loopBuf, int iPrint, int Nmoms, int imom, bool oneD){

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    if(oneD){
      for(int lt=0;lt<GK_localL[3];lt++){
	for(int gm=0;gm<16;gm++){
	  writeBuf[0+2*gm+2*16*lt] = 0.25*loopBuf[0+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
	  writeBuf[1+2*gm+2*16*lt] = 0.25*loopBuf[1+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
	}
      }
    }
    else{
      for(int lt=0;lt<GK_localL[3];lt++){
	for(int gm=0;gm<16;gm++){
	  writeBuf[0+2*gm+2*16*lt] = loopBuf[0+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
	  writeBuf[1+2*gm+2*16*lt] = loopBuf[1+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
	}
      }
    }
  }//-if GK_timeRank

}
*/

/* Moved to qudaQKXTM_Loops.cpp
//-C.K: Funtion to write the loops in HDF5 format
template<typename Float>
void writeLoops_HDF5(Float *buf_std_uloc, Float *buf_gen_uloc, 
		     Float **buf_std_oneD, Float **buf_std_csvC, 
		     Float **buf_gen_oneD, Float **buf_gen_csvC, 
		     char *file_pref, 
		     qudaQKXTM_loopInfo loopInfo, int **momQsq,
		     bool exact_loop, bool useTSM, bool LowPrec){

  if(exact_loop && useTSM) errorQuda("writeLoops_HDF5: Got conflicting options - exact_loop AND useTSM.\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    char fname[512];
    int Nprint,Ndump;

    if(exact_loop){
      Nprint = 1;
      sprintf(fname,"%s_Qsq%d.h5",file_pref,loopInfo.Qsq);
    }
    else{
      if(useTSM){
	if(LowPrec){
	  Nprint = loopInfo.TSM_NprintLP;
	  Ndump  = loopInfo.TSM_NdumpLP;
	  sprintf(fname,"%s_NLP%04d_step%04d_Qsq%d.h5",file_pref,loopInfo.TSM_NLP,Ndump,loopInfo.Qsq);
	}
	else{
	  Nprint = loopInfo.TSM_NprintHP;
	  Ndump  = loopInfo.TSM_NdumpHP;
	  sprintf(fname,"%s_NHP%04d_step%04d_Qsq%d.h5",file_pref,loopInfo.TSM_NHP,Ndump,loopInfo.Qsq);
	}	
      }
      else{
	Nprint = loopInfo.Nprint;
	Ndump  = loopInfo.Ndump;
	sprintf(fname,"%s_Ns%04d_step%04d_Qsq%d.h5",file_pref,loopInfo.Nstoch,Ndump,loopInfo.Qsq);
      }
    }

    double *loopBuf = NULL;
    double *writeBuf = (double*) malloc(GK_localL[3]*16*2*sizeof(double));

    hsize_t start[3]  = {GK_timeRank*GK_localL[3], 0, 0};

    // Dimensions of the dataspace
    hsize_t dims[3]  = {GK_totalL[3], 16, 2}; // Global
    hsize_t ldims[3] = {GK_localL[3], 16, 2}; // Local

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    if(file_id<0) errorQuda("Cannot open %s. Check that directory exists!\n",fname);

    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",loopInfo.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t group2_id;
    hid_t group3_id;
    hid_t group4_id;
    hid_t group5_id;

    for(int iPrint=0;iPrint<Nprint;iPrint++){

      if(!exact_loop){
	char *group2_tag;
	if(useTSM){
	  if(LowPrec) asprintf(&group2_tag,"NLP_%04d",(iPrint+1)*Ndump);
	  else        asprintf(&group2_tag,"NHP_%04d",(iPrint+1)*Ndump);
	}
	else asprintf(&group2_tag,"Nstoch_%04d",(iPrint+1)*Ndump);
	group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }

      for(int it=0;it<6;it++){
	char *group3_tag;
	asprintf(&group3_tag,"%s",loopInfo.loop_type[it]);

	if(exact_loop) group3_id = H5Gcreate(group1_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else           group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	for(int imom=0;imom<loopInfo.Nmoms;imom++){
	  char *group4_tag;
	  asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",momQsq[imom][0],momQsq[imom][1],momQsq[imom][2]);
	  group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	  if(loopInfo.loop_oneD[it]){
	    for(int mu=0;mu<4;mu++){
	      if(strcmp(loopInfo.loop_type[it],"Loops")==0)   loopBuf = buf_std_oneD[mu];
	      if(strcmp(loopInfo.loop_type[it],"LoopsCv")==0) loopBuf = buf_std_csvC[mu];
	      if(strcmp(loopInfo.loop_type[it],"LpsDw")==0)   loopBuf = buf_gen_oneD[mu];
	      if(strcmp(loopInfo.loop_type[it],"LpsDwCv")==0) loopBuf = buf_gen_csvC[mu];

	      char *group5_tag;
	      asprintf(&group5_tag,"dir_%02d",mu);
	      group5_id = H5Gcreate(group4_id, group5_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	      hid_t filespace  = H5Screate_simple(3, dims, NULL);
	      hid_t dataset_id = H5Dcreate(group5_id, "loop", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	      hid_t subspace   = H5Screate_simple(3, ldims, NULL);
	      filespace = H5Dget_space(dataset_id);
	      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
	      hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	      getLoopWriteBuf(writeBuf,loopBuf,iPrint,loopInfo.Nmoms,imom, loopInfo.loop_oneD[it]);

	      herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, subspace, filespace, plist_id, writeBuf);

	      H5Sclose(subspace);
	      H5Dclose(dataset_id);
	      H5Sclose(filespace);
	      H5Pclose(plist_id);

	      H5Gclose(group5_id);
	    }//-mu
	  }//-if
	  else{
	    if(strcmp(loopInfo.loop_type[it],"Scalar")==0) loopBuf = buf_std_uloc;
	    if(strcmp(loopInfo.loop_type[it],"dOp")==0)    loopBuf = buf_gen_uloc;

	    hid_t filespace  = H5Screate_simple(3, dims, NULL);
	    hid_t dataset_id = H5Dcreate(group4_id, "loop", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    hid_t subspace   = H5Screate_simple(3, ldims, NULL);
	    filespace = H5Dget_space(dataset_id);
	    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
	    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	    getLoopWriteBuf(writeBuf,loopBuf,iPrint,loopInfo.Nmoms,imom, loopInfo.loop_oneD[it]);

	    herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, subspace, filespace, plist_id, writeBuf);

	    H5Sclose(subspace);
	    H5Dclose(dataset_id);
	    H5Sclose(filespace);
	    H5Pclose(plist_id);
	  }
	  H5Gclose(group4_id);
	}//-imom
	H5Gclose(group3_id);
      }//-it

      if(!exact_loop) H5Gclose(group2_id);
    }//-iPrint
    H5Gclose(group1_id);
    H5Fclose(file_id);
  
    free(writeBuf);
  }
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

