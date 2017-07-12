//-C.K. Interface for performing the loops and the correlation function 
//contractions, including the exact deflation using ARPACK
//#include <qudaQKXTM.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM.h>
#include <qudaQKXTM_utils.h>
#include <errno.h>
#include <mpi.h>
#include <limits>
#ifdef QKXTM_MKL
#include <mkl.h>
#endif
#ifdef QKXTM_OPENBLAS
#include <cblas.h>
#include <common.h>
#include <omp.h>
#endif

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

/*
// function for writing
extern "C"{
#include <lime.h>
}

static void qcd_swap_4(float *Rd, size_t N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
    {
      j=i; k=j+3;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
    }
}


static void qcd_swap_8(double *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R = (char*) Rd;

  max = R+(N<<3);
  for(i=R;i<max;i+=8)
    {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
    }
}

static int qcd_isBigEndian()
{
  union{
    char C[4];
    int  R   ;
  }word;
  word.R=1;
  if(word.C[3]==1) return 1;
  if(word.C[0]==1) return 0;

  return -1;
}

static char* qcd_getParam(char token[],char* params,int len)
{
  int i,token_len=strlen(token);

  for(i=0;i<len-token_len;i++)
    {
      if(memcmp(token,params+i,token_len)==0)
	{
	  i+=token_len;
	  *(strchr(params+i,'<'))='\0';
	  break;
	}
    }
  return params+i;
}
*/

/////////////////////////////
// QKXTM Class Definifions //
/////////////////////////////

/*
#define CC QKXTM_Field<Float>
#define DEVICE_MEMORY_REPORT
#define CMPLX_FLOAT std::complex<Float>

//--------------------------//
// class QKXTM_Field //
//--------------------------//

// This is is a class which allocates memory on either the
// the device, or host, or both for the structures:
// Field: one complex number per spacetime point.
// Gauge: one SU(3) link variable per spacetime point X spacetime dimension.
// Vector: 12 complex numbers (3colour x 4spin) per spacetime point.
// Vector3D: as above, but only defined on a single timeslice.
// Propagator: 12x12 complex matrix per sink position 
// (usually a whole spacetime volume.)
// Propagtor3D: as above, but with sinks only at one timeslice.

template<typename Float>
QKXTM_Field<Float>::QKXTM_Field(ALLOCATION_FLAG alloc_flag, 
					      CLASS_ENUM classT):
  h_elem(NULL) , d_elem(NULL) , h_ext_ghost(NULL) , h_elem_backup(NULL) , 
  isAllocHost(false) , isAllocDevice(false), isAllocHostBackup(false)
{
  if(GK_init_qudaQKXTM_flag == false) 
    errorQuda("You must initialize init_qudaQKXTM first");

  switch(classT){
  case FIELD:
    field_length = 1;
    total_length = GK_localVolume;
    break;
  case GAUGE:
    field_length = GK_nDim * GK_nColor * GK_nSpin;
    total_length = GK_localVolume;
    break;
  case VECTOR:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume;
    break;
  case PROPAGATOR3D:
    field_length = GK_nSpin * GK_nColor * GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  case VECTOR3D:
    field_length = GK_nSpin * GK_nColor;
    total_length = GK_localVolume/GK_localL[3];
    break;
  }

  ghost_length = 0;

  for(int i = 0 ; i < GK_nDim ; i++)
    ghost_length += 2*GK_surface3D[i];

  total_plus_ghost_length = total_length + ghost_length;

  bytes_total_length = total_length*field_length*2*sizeof(Float);
  bytes_ghost_length = ghost_length*field_length*2*sizeof(Float);
  bytes_total_plus_ghost_length = total_plus_ghost_length*field_length*2*sizeof(Float);

  if( alloc_flag == BOTH ){
    create_host();
    create_device();
  }
  else if (alloc_flag == HOST){
    create_host();
  }
  else if (alloc_flag == DEVICE){
    create_device();
  }
  else if (alloc_flag == BOTH_EXTRA){
    create_host();
    create_host_backup();
    create_device();    
  }

}

//Destructor
template<typename Float>
QKXTM_Field<Float>::~QKXTM_Field(){
  if(h_elem != NULL) destroy_host();
  if(h_elem_backup != NULL) destroy_host_backup();
  if(d_elem != NULL) destroy_device();
}

template<typename Float>
void QKXTM_Field<Float>::create_host(){
  h_elem = (Float*) malloc(bytes_total_plus_ghost_length);
  h_ext_ghost = (Float*) malloc(bytes_ghost_length);
  if(h_elem == NULL || 
     h_ext_ghost == NULL) errorQuda("Error with allocation host memory");
  isAllocHost = true;
  zero_host();
}

template<typename Float>
void QKXTM_Field<Float>::create_host_backup(){
  h_elem_backup = (Float*) malloc(bytes_total_plus_ghost_length);
  if(h_elem_backup == NULL) errorQuda("Error with allocation host memory");
  isAllocHostBackup = true;
  zero_host_backup();
}

template<typename Float>
void QKXTM_Field<Float>::create_device(){
  cudaMalloc((void**)&d_elem,bytes_total_plus_ghost_length);
  checkCudaError();
#ifdef DEVICE_MEMORY_REPORT
  // device memory in MB
  GK_deviceMemory += bytes_total_length/(1024.*1024.);          
  printfQuda("Device memory in use is %f MB A QKXTM \n",GK_deviceMemory);
#endif
  isAllocDevice = true;
  zero_device();
}

template<typename Float>
void QKXTM_Field<Float>::destroy_host(){
  free(h_elem);
  free(h_ext_ghost);
  h_elem=NULL;
  h_ext_ghost = NULL;
}

template<typename Float>
void QKXTM_Field<Float>::destroy_host_backup(){
  free(h_elem_backup);
  h_elem=NULL;
}

template<typename Float>
void QKXTM_Field<Float>::destroy_device(){
  cudaFree(d_elem);
  checkCudaError();
  d_elem = NULL;
#ifdef DEVICE_MEMORY_REPORT
  GK_deviceMemory -= bytes_total_length/(1024.*1024.);
  printfQuda("Device memory in used is %f MB D \n",GK_deviceMemory);
#endif
}

template<typename Float>
void QKXTM_Field<Float>::zero_host(){
  memset(h_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field<Float>::zero_host_backup(){
  memset(h_elem_backup,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field<Float>::zero_device(){
  cudaMemset(d_elem,0,bytes_total_plus_ghost_length);
}

template<typename Float>
void QKXTM_Field<Float>::createTexObject(cudaTextureObject_t *tex){
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = CC::Precision();
  if(precision == 4) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == 4){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == 8){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = bytes_total_plus_ghost_length;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  checkCudaError();
}

template<typename Float>
void QKXTM_Field<Float>::destroyTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
}

template<typename Float>
void QKXTM_Field<Float>::printInfo(){
  printfQuda("This object has precision %d\n",Precision());
  printfQuda("This object needs %f Mb\n",
	     bytes_total_plus_ghost_length/(1024.*1024.));
  printfQuda("The flag for the host allocation is %d\n",(int) isAllocHost);
  printfQuda("The flag for the device allocation is %d\n",(int) isAllocDevice);
}
*/

//--------------------------//
// class QKXTM_Gauge //
//--------------------------//

/*
template<typename Float>
QKXTM_Gauge<Float>::QKXTM_Gauge(ALLOCATION_FLAG alloc_flag, 
					      CLASS_ENUM classT): 
  QKXTM_Field<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Gauge<Float>::packGauge(void **gauge){

  double **p_gauge = (double**) gauge;
  
  for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++)
	  for(int part = 0 ; part < 2 ; part++){
	    CC::h_elem[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       iv*2 + part] = (Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
							   c1*GK_nColor*2 + c2*2 + part];
	  }
}

template<typename Float>
void QKXTM_Gauge<Float>::packGaugeToBackup(void **gauge){
  double **p_gauge = (double**) gauge;
  if(CC::h_elem_backup != NULL){
    for(int dir = 0 ; dir < GK_nDim ; dir++)
    for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++)
    for(int part = 0 ; part < 2 ; part++){
      CC::h_elem_backup[dir*GK_nColor*GK_nColor*GK_localVolume*2 + 
			c1*GK_nColor*GK_localVolume*2 + 
			c2*GK_localVolume*2 + 
			iv*2 + part] = 
	(Float) p_gauge[dir][iv*GK_nColor*GK_nColor*2 + 
			     c1*GK_nColor*2 + 
			     c2*2 + part];
    }
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }

}

template<typename Float>
void QKXTM_Gauge<Float>::justDownloadGauge(){
  cudaMemcpy(CC::h_elem,CC::d_elem,CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge<Float>::loadGauge(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge<Float>::loadGaugeFromBackup(){
  if(CC::h_elem_backup != NULL){
    cudaMemcpy(CC::d_elem,CC::h_elem_backup, CC::bytes_total_length, 
	       cudaMemcpyHostToDevice );
    checkCudaError();
  }
  else{
    errorQuda("Error you can call this method only if you allocate memory for h_elem_backup");
  }
}

// gpu collect ghost and send it to host
template<typename Float>
void QKXTM_Gauge<Float>::ghostToHost(){   

  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3];
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    position = GK_localL[0]-1;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[0]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
			   c1*GK_nColor*GK_surface3D[0]*2 + 
			   c2*GK_surface3D[0]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction y 
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    // number of blocks that we need
    int height = GK_localL[2] * GK_localL[3];
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[1]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
			   c1*GK_nColor*GK_surface3D[1]*2 + 
			   c2*GK_surface3D[1]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){

    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_minusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
    // set minus points to plus area
    position = 0;
    for(int i = 0 ; i < GK_nDim ; i++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int c2 = 0 ; c2 < GK_nColor ; c2++){
	  d_elem_offset = (CC::d_elem + 
			   i*GK_nColor*GK_nColor*GK_localVolume*2 + 
			   c1*GK_nColor*GK_localVolume*2 + 
			   c2*GK_localVolume*2 + 
			   position*2);  
	  h_elem_offset = (CC::h_elem + 
			   GK_plusGhost[2]*GK_nDim*GK_nColor*GK_nColor*2 + 
			   i*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
			   c1*GK_nColor*GK_surface3D[2]*2 + 
			   c2*GK_surface3D[2]*2);
	  cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		       spitch,width,height,cudaMemcpyDeviceToHost);
	}
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nDim*GK_nColor*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nDim*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
  }
  checkCudaError();
}

template<typename Float>
void QKXTM_Gauge<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = 
	  GK_surface3D[idim]*GK_nColor*GK_nColor*GK_nDim*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nColor*GK_nColor*GK_nDim*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nColor*GK_nColor*GK_nDim*2;
	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }

    for(int idim = 0 ; idim < GK_nDim ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void QKXTM_Gauge<Float>::ghostToDevice(){
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem+GK_localVolume*GK_nColor*GK_nColor*GK_nDim*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Gauge<Float>::calculatePlaq(){
  cudaTextureObject_t tex;

  ghostToHost();
  cpuExchangeGhost();
  ghostToDevice();
  CC::createTexObject(&tex);
  run_calculatePlaq_kernel(tex, sizeof(Float));
  CC::destroyTexObject(tex);

}

*/

//---------------------------//
// class QKXTM_Vector //
//---------------------------//

/*
template<typename Float>
QKXTM_Vector<Float>::QKXTM_Vector(ALLOCATION_FLAG alloc_flag, 
						CLASS_ENUM classT): 
  QKXTM_Field<Float>(alloc_flag, classT){ ; }

template<typename Float>
void QKXTM_Vector<Float>::packVector(Float *vector){
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++)  // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + iv*2 + part] = 
	    vector[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2 + c1*2 + part];
	}
}
 

template<typename Float>
void QKXTM_Vector<Float>::unpackVector(){

  Float *vector_tmp = (Float*) malloc( CC::bytes_total_length );
  if(vector_tmp == NULL)
    errorQuda("Error in allocate memory of tmp vector in unpackVector\n");
  
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  vector_tmp[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		       c1*GK_localVolume*2 + iv*2 + part];
	}
  
  memcpy(CC::h_elem,vector_tmp, CC::bytes_total_length);
  
  free(vector_tmp);
}

template<typename Float>
void QKXTM_Vector<Float>::unpackVector(Float *vector){
  
  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  CC::h_elem[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    vector[mu*GK_nColor*GK_localVolume*2 + 
		   c1*GK_localVolume*2 + iv*2 + part];
	}
}


template<typename Float>
void QKXTM_Vector<Float>::loadVector(){
  cudaMemcpy(CC::d_elem,CC::h_elem,CC::bytes_total_length, 
	     cudaMemcpyHostToDevice );
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector<Float>::unloadVector(){
  cudaMemcpy(CC::h_elem, CC::d_elem, CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
}


template<typename Float>
void QKXTM_Vector<Float>::download(){

  cudaMemcpy(CC::h_elem, CC::d_elem, CC::bytes_total_length, 
	     cudaMemcpyDeviceToHost);
  checkCudaError();

  Float *vector_tmp = (Float*) malloc( CC::bytes_total_length );
  if(vector_tmp == NULL) errorQuda("Error in allocate memory of tmp vector");

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int mu = 0 ; mu < GK_nSpin ; mu++) // always work with format colors inside spins
      for(int c1 = 0 ; c1 < GK_nColor ; c1++)
	for(int part = 0 ; part < 2 ; part++){
	  vector_tmp[iv*GK_nSpin*GK_nColor*2 + mu*GK_nColor*2+c1*2+part] = 
	    CC::h_elem[mu*GK_nColor*GK_localVolume*2 + 
		       c1*GK_localVolume*2 + iv*2 + part];
	}
  
  memcpy(CC::h_elem, vector_tmp, CC::bytes_total_length);

  free(vector_tmp);
}


template<typename Float>
void QKXTM_Vector<Float>::castDoubleToFloat(QKXTM_Vector<double> &vecIn){
  if(typeid(Float) != typeid(float) )errorQuda("This method works only to convert double to single precision\n");
  run_castDoubleToFloat((void*)CC::d_elem, (void*)vecIn.D_elem());
}

template<typename Float>
void QKXTM_Vector<Float>::castFloatToDouble(QKXTM_Vector<float> &vecIn){
  if(typeid(Float) != typeid(double) )errorQuda("This method works only to convert single to double precision\n");
  run_castFloatToDouble((void*)CC::d_elem, (void*)vecIn.D_elem());
}

template<typename Float>
void QKXTM_Vector<Float>::ghostToHost(){
  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3];
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = (GK_localL[0]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[0]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[0]*2 + 
			 c1*GK_surface3D[0]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[0]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[0]*2 + 
			 c1*GK_surface3D[0]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
  }
  // direction y 
  if( GK_localL[1] < GK_totalL[1]){
    int position;
    // number of blocks that we need
    int height = GK_localL[2] * GK_localL[3];
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[1]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[1]*2 + 
			 c1*GK_surface3D[1]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[1]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[1]*2 + 
			 c1*GK_surface3D[1]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
  }
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){
    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_minusGhost[2]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[2]*2 + 
			 c1*GK_surface3D[2]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
      for(int c1 = 0 ; c1 < GK_nColor ; c1++){
	d_elem_offset = (CC::d_elem + 
			 mu*GK_nColor*GK_localVolume*2 + 
			 c1*GK_localVolume*2 + 
			 position*2);  
	h_elem_offset = (CC::h_elem + 
			 GK_plusGhost[2]*GK_nSpin*GK_nColor*2 + 
			 mu*GK_nColor*GK_surface3D[2]*2 + 
			 c1*GK_surface3D[2]*2);
	cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		     spitch,width,height,cudaMemcpyDeviceToHost);
      }
  }
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nSpin*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,
		 cudaMemcpyDeviceToHost);
    // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nSpin*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,width,height,
		 cudaMemcpyDeviceToHost);
  }
}


template<typename Float>
void QKXTM_Vector<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes=GK_surface3D[idim]*GK_nSpin*GK_nColor*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nSpin*GK_nColor*2;

	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nSpin*GK_nColor*2;

	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }
    for(int idim = 0 ; idim < GK_nDim ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void QKXTM_Vector<Float>::ghostToDevice(){ 
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem + GK_localVolume*GK_nSpin*GK_nColor*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}


template<typename Float>
void QKXTM_Vector<Float>::gaussianSmearing(QKXTM_Vector<Float> &vecIn,QKXTM_Gauge<Float> &gaugeAPE){
  gaugeAPE.ghostToHost();
  gaugeAPE.cpuExchangeGhost();
  gaugeAPE.ghostToDevice();

  vecIn.ghostToHost();
  vecIn.cpuExchangeGhost();
  vecIn.ghostToDevice();

  cudaTextureObject_t texVecIn,texVecOut,texGauge;
  this->createTexObject(&texVecOut);
  vecIn.createTexObject(&texVecIn);
  gaugeAPE.createTexObject(&texGauge);
  
  for(int i = 0 ; i < GK_nsmearGauss ; i++){
    if( (i%2) == 0){
      run_GaussianSmearing((void*)this->D_elem(),texVecIn,texGauge, sizeof(Float));
      this->ghostToHost();
      this->cpuExchangeGhost();
      this->ghostToDevice();
    }
    else{
      run_GaussianSmearing((void*)vecIn.D_elem(),texVecOut,texGauge, sizeof(Float));
      vecIn.ghostToHost();
      vecIn.cpuExchangeGhost();
      vecIn.ghostToDevice();
    }
  }

  if( (GK_nsmearGauss%2) == 0) cudaMemcpy(this->D_elem(),vecIn.D_elem(),CC::bytes_total_length,cudaMemcpyDeviceToDevice);
  
  this->destroyTexObject(texVecOut);
  vecIn.destroyTexObject(texVecIn);
  gaugeAPE.destroyTexObject(texGauge);
  checkCudaError();
}

template<typename Float>
void QKXTM_Vector<Float>::uploadToCuda(ColorSpinorField *qudaVector, 
					      bool isEv){
  run_UploadToCuda((void*) CC::d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void QKXTM_Vector<Float>::downloadFromCuda(ColorSpinorField *qudaVector, bool isEv){
  run_DownloadFromCuda((void*) CC::d_elem, *qudaVector, sizeof(Float), isEv);
}

template<typename Float>
void  QKXTM_Vector<Float>::scaleVector(double a){
  run_ScaleVector(a,(void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Vector<Float>::conjugate(){
  run_conjugate_vector((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Vector<Float>::apply_gamma5(){
  run_apply_gamma5_vector((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void QKXTM_Vector<Float>::norm2Host(){
  Float res = 0.;
  Float globalRes;

  for(int i = 0 ; i < GK_nSpin*GK_nColor*GK_localVolume ; i++){
    res += CC::h_elem[i*2 + 0]*CC::h_elem[i*2 + 0] + CC::h_elem[i*2 + 1]*CC::h_elem[i*2 + 1];
  }

  int rc = MPI_Allreduce(&res , &globalRes , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD);
  if( rc != MPI_SUCCESS ) errorQuda("Error in MPI reduction for plaquette");
  printfQuda("Vector norm2 is %e\n",globalRes);
}

template<typename Float>
void QKXTM_Vector<Float>::copyPropagator3D(QKXTM_Propagator3D<Float> &prop, int timeslice, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  int V3 = GK_localVolume/GK_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + 
		     mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + 
		     timeslice*V3*2);
      pointer_src = (prop.D_elem() + 
		     mu*4*3*3*V3*2 + 
		     nu*3*3*V3*2 + 
		     c1*3*V3*2 + 
		     c2*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), 
		 cudaMemcpyDeviceToDevice);
    }

  pointer_src = NULL;
  pointer_dst = NULL;
  checkCudaError();

}

template<typename Float>
void QKXTM_Vector<Float>::copyPropagator(QKXTM_Propagator<Float> &prop, int nu , int c2){
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + 
		     mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2);
      pointer_src = (prop.D_elem() + 
		     mu*4*3*3*GK_localVolume*2 + 
		     nu*3*3*GK_localVolume*2 + 
		     c1*3*GK_localVolume*2 + 
		     c2*GK_localVolume*2);
      cudaMemcpy(pointer_dst, pointer_src, GK_localVolume*2 * sizeof(Float), cudaMemcpyDeviceToDevice);
    }
  
  pointer_src = NULL;
  pointer_dst = NULL;
  checkCudaError();

}

template<typename Float>
void QKXTM_Vector<Float>::write(char *filename){
  FILE *fid;
  int error_in_header=0;
  LimeWriter *limewriter;
  LimeRecordHeader *limeheader = NULL;
  int ME_flag=0, MB_flag=0, limeStatus;
  u_int64_t message_length;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray  
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  long int i;
  int chunksize,mu,c1;
  char *buffer;
  int x,y,z,t;
  char tmp_string[2048];

  if(comm_rank() == 0){ // master will write the lime header
    fid = fopen(filename,"w");
    if(fid == NULL){
      fprintf(stderr,"Error open file to write propagator in %s \n",__func__);
      comm_abort(-1);
    }
    else{
      limewriter = limeCreateWriter(fid);
      if(limewriter == (LimeWriter*)NULL) {
	fprintf(stderr, "Error in %s. LIME error in file for writing!\n", __func__);
	error_in_header=1;
	comm_abort(-1);
      }
      else
	{
	  sprintf(tmp_string, "DiracFermion_Sink");
	  message_length=(long int) strlen(tmp_string);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "propagator-type", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }

	  if( typeid(Float) == typeid(double) )
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>64</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);
	  else
	    sprintf(tmp_string, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n\t<field>diracFermion</field>\n\t<precision>32</precision>\n\t<flavours>1</flavours>\n\t<lx>%d</lx>\n\t<ly>%d</ly>\n\t<lz>%d</lz>\n\t<lt>%d</lt>\n\t<spin>4</spin>\n\t<colour>3</colour>\n</etmcFormat>", GK_totalL[0], GK_totalL[1], GK_totalL[2], GK_totalL[3]);

	  message_length=(long int) strlen(tmp_string); 
	  MB_flag=1; ME_flag=1;

	  limeheader = limeCreateHeader(MB_flag, ME_flag, "quda-propagator-format", message_length);
	  if(limeheader == (LimeRecordHeader*)NULL)
	    {
	      fprintf(stderr, "Error in %s. LIME create header error.\n", __func__);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeStatus = limeWriteRecordHeader(limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  limeDestroyHeader(limeheader);
	  limeStatus = limeWriteRecordData(tmp_string, &message_length, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	      comm_abort(-1);
	    }
	  
	  message_length = GK_totalVolume*4*3*2*sizeof(Float);
	  MB_flag=1; ME_flag=1;
	  limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", message_length);
	  limeStatus = limeWriteRecordHeader( limeheader, limewriter);
	  if(limeStatus < 0 )
	    {
	      fprintf(stderr, "Error in %s. LIME write header error %d\n", __func__, limeStatus);
	      error_in_header=1;
	    }
	  limeDestroyHeader( limeheader );
	}
      message_length=1;
      limeWriteRecordData(tmp_string, &message_length, limewriter);
      limeDestroyWriter(limewriter);
      offset = ftell(fid)-1;
      fclose(fid);
    }
  }

  MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
  
  sizes[0]=GK_totalL[3];
  sizes[1]=GK_totalL[2];
  sizes[2]=GK_totalL[1];
  sizes[3]=GK_totalL[0];
  sizes[4]=4*3*2;
  lsizes[0]=GK_localL[3];
  lsizes[1]=GK_localL[2];
  lsizes[2]=GK_localL[1];
  lsizes[3]=GK_localL[0];
  lsizes[4]=sizes[4];
  starts[0]=comm_coords(default_topo)[3]*GK_localL[3];
  starts[1]=comm_coords(default_topo)[2]*GK_localL[2];
  starts[2]=comm_coords(default_topo)[1]*GK_localL[1];
  starts[3]=comm_coords(default_topo)[0]*GK_localL[0];
  starts[4]=0;  

  if( typeid(Float) == typeid(double) )
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_DOUBLE,&subblock);
  else
    MPI_Type_create_subarray(5,sizes,lsizes,starts,
			     MPI_ORDER_C,MPI_FLOAT,&subblock);

  MPI_Type_commit(&subblock);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY, 
		MPI_INFO_NULL, &mpifid);
  MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, 
		    "native", MPI_INFO_NULL);

  chunksize=4*3*2*sizeof(Float);
  buffer = (char*) malloc(chunksize*GK_localVolume);

  if(buffer==NULL)  
    {
      fprintf(stderr,"Error in %s! Out of memory\n", __func__);
      comm_abort(-1);
    }

  i=0;
                        
  for(t=0; t<GK_localL[3];t++)
  for(z=0; z<GK_localL[2];z++)
  for(y=0; y<GK_localL[1];y++)
  for(x=0; x<GK_localL[0];x++)
  for(mu=0; mu<4; mu++)
  for(c1=0; c1<3; c1++) 
    // works only for QUDA_DIRAC_ORDER (color inside spin)
    {
      ((Float *)buffer)[i] = 
	(CC::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 0]);
      
      ((Float *)buffer)[i+1] = 
	(CC::h_elem[t*GK_localL[2]*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    z*GK_localL[1]*GK_localL[0]*4*3*2 + 
		    y*GK_localL[0]*4*3*2 + 
		    x*4*3*2 + mu*3*2 + c1*2 + 1]);
      i+=2;
    }
  if(!qcd_isBigEndian()){
    if( typeid(Float) == typeid(double) ) 
      qcd_swap_8((double*) buffer,2*4*3*GK_localVolume);
    else qcd_swap_4((float*) buffer,2*4*3*GK_localVolume);
  }
  if( typeid(Float) == typeid(double) )
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_DOUBLE, &status);
  else
    MPI_File_write_all(mpifid, buffer, 4*3*2*GK_localVolume, 
		       MPI_FLOAT, &status);

  free(buffer);
  MPI_File_close(&mpifid);
  MPI_Type_free(&subblock);
}
*/

//-------------------------------//
// class QKXTM_Propagator //
//-------------------------------//
/*
template<typename Float>
QKXTM_Propagator<Float>::
QKXTM_Propagator(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT): 
  QKXTM_Field<Float>(alloc_flag, classT){;}

template <typename Float>
void QKXTM_Propagator<Float>::
absorbVectorToHost(QKXTM_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_host;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_host = (CC::h_elem + 
			mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
			nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
			c1*GK_nColor*GK_localVolume*2 + 
			c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_host,pointVec_dev,GK_localVolume*2*sizeof(Float),cudaMemcpyDeviceToHost); 
    }
  checkCudaError();
}
 
template <typename Float>
void QKXTM_Propagator<Float>::absorbVectorToDevice(QKXTM_Vector<Float> &vec, int nu, int c2){
  Float *pointProp_dev;
  Float *pointVec_dev;
  for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++){
      pointProp_dev = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2);
      pointVec_dev = vec.D_elem() + mu*GK_nColor*GK_localVolume*2 + c1*GK_localVolume*2;
      cudaMemcpy(pointProp_dev,pointVec_dev,GK_localVolume*2*sizeof(Float),
		 cudaMemcpyDeviceToDevice); 
    }
  checkCudaError();
}

template<typename Float>
void QKXTM_Propagator<Float>::rotateToPhysicalBase_device(int sign){
  if( (sign != +1) && (sign != -1) ) errorQuda("The sign can be only +-1\n");
  run_rotateToPhysicalBase((void*) CC::d_elem, sign , sizeof(Float));
}

//QKXTM: DMH Rewrote some parts of this function to conform to new
// QUDA standards. Eg, assigning vaules to complex variable:
// var.real() = 1.0; is changed to var.real(1.0);
template <typename Float>
void QKXTM_Propagator<Float>::rotateToPhysicalBase_host(int sign_int){
  if( (sign_int != +1) && (sign_int != -1) ) 
    errorQuda("The sign can be only +-1\n");
  
  std::complex<Float> sign;
  sign.real(1.0*sign_int);
  sign.imag(0.0);

  std::complex<Float> coeff;
  coeff.real(0.5);
  coeff.imag(0.0);
  
  std::complex<Float> P[4][4];
  std::complex<Float> PT[4][4];
  std::complex<Float> imag_unit;
  //imag_unit.real() = 0.;
  //imag_unit.imag() = 1.;
  imag_unit.real(0.0);
  imag_unit.imag(1.0);

  for(int iv = 0 ; iv < GK_localVolume ; iv++)
    for(int c1 = 0 ; c1 < 3 ; c1++)
      for(int c2 = 0 ; c2 < 3 ; c2++){
	      
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    //P[mu][nu].real() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 0];
	    //P[mu][nu].imag() = CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + nu*GK_nColor*GK_nColor*GK_localVolume + c1*GK_nColor*GK_localVolume + c2*GK_localVolume + iv)*2 + 1]
	    P[mu][nu].real(CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
				       nu*GK_nColor*GK_nColor*GK_localVolume + 
				       c1*GK_nColor*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 0]);
	    P[mu][nu].imag(CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
				       nu*GK_nColor*GK_nColor*GK_localVolume + 
				       c1*GK_nColor*GK_localVolume + 
				       c2*GK_localVolume + iv)*2 + 1]);
	  }
	
	PT[0][0] = coeff * (P[0][0] + sign * ( imag_unit * P[0][2] ) + sign * ( imag_unit * P[2][0] ) - P[2][2]);
	PT[0][1] = coeff * (P[0][1] + sign * ( imag_unit * P[0][3] ) + sign * ( imag_unit * P[2][1] ) - P[2][3]);
	PT[0][2] = coeff * (sign * ( imag_unit * P[0][0] ) + P[0][2] - P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[0][3] = coeff * (sign * ( imag_unit * P[0][1] ) + P[0][3] - P[2][1] + sign * ( imag_unit * P[2][3] ));
	
	PT[1][0] = coeff * (P[1][0] + sign * ( imag_unit * P[1][2] ) + sign * ( imag_unit * P[3][0] ) - P[3][2]);
	PT[1][1] = coeff * (P[1][1] + sign * ( imag_unit * P[1][3] ) + sign * ( imag_unit * P[3][1] ) - P[3][3]);
	PT[1][2] = coeff * (sign * ( imag_unit * P[1][0] ) + P[1][2] - P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[1][3] = coeff * (sign * ( imag_unit * P[1][1] ) + P[1][3] - P[3][1] + sign * ( imag_unit * P[3][3] ));
	
	PT[2][0] = coeff * (sign * ( imag_unit * P[0][0] ) - P[0][2] + P[2][0] + sign * ( imag_unit * P[2][2] ));
	PT[2][1] = coeff * (sign * ( imag_unit * P[0][1] ) - P[0][3] + P[2][1] + sign * ( imag_unit * P[2][3] ));
	PT[2][2] = coeff * (sign * ( imag_unit * P[0][2] ) - P[0][0] + sign * ( imag_unit * P[2][0] ) + P[2][2]);
	PT[2][3] = coeff * (sign * ( imag_unit * P[0][3] ) - P[0][1] + sign * ( imag_unit * P[2][1] ) + P[2][3]);

	PT[3][0] = coeff * (sign * ( imag_unit * P[1][0] ) - P[1][2] + P[3][0] + sign * ( imag_unit * P[3][2] ));
	PT[3][1] = coeff * (sign * ( imag_unit * P[1][1] ) - P[1][3] + P[3][1] + sign * ( imag_unit * P[3][3] ));
	PT[3][2] = coeff * (sign * ( imag_unit * P[1][2] ) - P[1][0] + sign * ( imag_unit * P[3][0] ) + P[3][2]);
	PT[3][3] = coeff * (sign * ( imag_unit * P[1][3] ) - P[1][1] + sign * ( imag_unit * P[3][1] ) + P[3][3]);

	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++){
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
			nu*GK_nColor*GK_nColor*GK_localVolume + 
			c1*GK_nColor*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 0] = PT[mu][nu].real();
	    CC::h_elem[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
			nu*GK_nColor*GK_nColor*GK_localVolume + 
			c1*GK_nColor*GK_localVolume + 
			c2*GK_localVolume + iv)*2 + 1] = PT[mu][nu].imag();
	  }
      }
}

// gpu collect ghost and send it to host
template<typename Float>
void QKXTM_Propagator<Float>::ghostToHost(){   
  // direction x 
  if( GK_localL[0] < GK_totalL[0]){
    int position;
    // number of blocks that we need
    int height = GK_localL[1] * GK_localL[2] * GK_localL[3]; 
    size_t width = 2*sizeof(Float);
    size_t spitch = GK_localL[0]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;
    // set plus points to minus area
    position = (GK_localL[0]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[0]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       c1*GK_nColor*GK_surface3D[0]*2 + 
		       c2*GK_surface3D[0]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
    // set minus points to plus area
    position = 0;

    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[0]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[0]*2 + 
		       c1*GK_nColor*GK_surface3D[0]*2 + 
		       c2*GK_surface3D[0]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
  }
  
  // direction y   
  if( GK_localL[1] < GK_totalL[1]){
    
    int position;
    int height = GK_localL[2] * GK_localL[3]; // number of blocks that we need
    size_t width = GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[1]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    // set plus points to minus area
    position = GK_localL[0]*(GK_localL[1]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[1]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       c1*GK_nColor*GK_surface3D[1]*2 + 
		       c2*GK_surface3D[1]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
    
    // set minus points to plus area
    position = 0;
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[1]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[1]*2 + 
		       c1*GK_nColor*GK_surface3D[1]*2 + 
		       c2*GK_surface3D[1]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }    
  }
  
  // direction z 
  if( GK_localL[2] < GK_totalL[2]){
    int position;
    // number of blocks that we need
    int height = GK_localL[3]; 
    size_t width = GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[2]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    // set plus points to minus area
    // position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1)*GK_localL[3];
    position = GK_localL[0]*GK_localL[1]*(GK_localL[2]-1);
    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_minusGhost[2]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       c1*GK_nColor*GK_surface3D[2]*2 + 
		       c2*GK_surface3D[2]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }

    // set minus points to plus area
    position = 0;

    for(int mu = 0 ; mu < GK_nSpin ; mu++)
    for(int nu = 0 ; nu < GK_nSpin ; nu++)
    for(int c1 = 0 ; c1 < GK_nColor ; c1++)
    for(int c2 = 0 ; c2 < GK_nColor ; c2++){
      d_elem_offset = (CC::d_elem + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       nu*GK_nColor*GK_nColor*GK_localVolume*2 + 
		       c1*GK_nColor*GK_localVolume*2 + 
		       c2*GK_localVolume*2 + 
		       position*2);  
      h_elem_offset = (CC::h_elem + 
		       GK_plusGhost[2]*GK_nSpin*GK_nSpin*
		       GK_nColor*GK_nColor*2 + 
		       mu*GK_nSpin*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       nu*GK_nColor*GK_nColor*GK_surface3D[2]*2 + 
		       c1*GK_nColor*GK_surface3D[2]*2 + 
		       c2*GK_surface3D[2]*2);
      cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,
		   spitch,width,height,cudaMemcpyDeviceToHost);
    }
  }
  
  // direction t 
  if( GK_localL[3] < GK_totalL[3]){
    int position;
    int height = GK_nSpin*GK_nSpin*GK_nColor*GK_nColor;
    size_t width = GK_localL[2]*GK_localL[1]*GK_localL[0]*2*sizeof(Float);
    size_t spitch = GK_localL[3]*width;
    size_t dpitch = width;
    Float *h_elem_offset = NULL;
    Float *d_elem_offset = NULL;

    // set plus points to minus area
    position = GK_localL[0]*GK_localL[1]*GK_localL[2]*(GK_localL[3]-1);
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_minusGhost[3]*GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);

    // set minus points to plus area
    position = 0;
    d_elem_offset = CC::d_elem + position*2;
    h_elem_offset = CC::h_elem + GK_plusGhost[3]*GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*2;
    cudaMemcpy2D(h_elem_offset,dpitch,d_elem_offset,spitch,
		 width,height,cudaMemcpyDeviceToHost);
    
    checkCudaError();
  }
}

template<typename Float>
void QKXTM_Propagator<Float>::cpuExchangeGhost(){
  if( comm_size() > 1 ){
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];

    Float *pointer_receive = NULL;
    Float *pointer_send = NULL;

    for(int idim = 0 ; idim < GK_nDim; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	size_t nbytes = GK_surface3D[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2*sizeof(Float);
	// send to plus
	pointer_receive = CC::h_ext_ghost + (GK_minusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_minusGhost[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;

	mh_from_back[idim] = comm_declare_receive_relative(pointer_receive,idim,-1,nbytes);
	mh_send_fwd[idim] = comm_declare_send_relative(pointer_send,idim,1,nbytes);
	comm_start(mh_from_back[idim]);
	comm_start(mh_send_fwd[idim]);
	comm_wait(mh_send_fwd[idim]);
	comm_wait(mh_from_back[idim]);
		
	// send to minus
	pointer_receive = CC::h_ext_ghost + (GK_plusGhost[idim]-GK_localVolume)*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
	pointer_send = CC::h_elem + GK_plusGhost[idim]*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;

	mh_from_fwd[idim] = comm_declare_receive_relative(pointer_receive,idim,1,nbytes);
	mh_send_back[idim] = comm_declare_send_relative(pointer_send,idim,-1,nbytes);
	comm_start(mh_from_fwd[idim]);
	comm_start(mh_send_back[idim]);
	comm_wait(mh_send_back[idim]);
	comm_wait(mh_from_fwd[idim]);
		
	pointer_receive = NULL;
	pointer_send = NULL;

      }
    }
    for(int idim = 0 ; idim < GK_nDim ; idim++){
      if(GK_localL[idim] < GK_totalL[idim]){
	comm_free(mh_send_fwd[idim]);
	comm_free(mh_from_fwd[idim]);
	comm_free(mh_send_back[idim]);
	comm_free(mh_from_back[idim]);
      }
    }
    
  }
}

template<typename Float>
void QKXTM_Propagator<Float>::ghostToDevice(){ 
  if(comm_size() > 1){
    Float *host = CC::h_ext_ghost;
    Float *device = CC::d_elem + GK_localVolume*GK_nSpin*GK_nColor*GK_nSpin*GK_nColor*2;
    cudaMemcpy(device,host,CC::bytes_ghost_length,cudaMemcpyHostToDevice);
    checkCudaError();
  }
}

template<typename Float>
void  QKXTM_Propagator<Float>::conjugate(){
  run_conjugate_propagator((void*)CC::d_elem,sizeof(Float));
}

template<typename Float>
void  QKXTM_Propagator<Float>::apply_gamma5(){
  run_apply_gamma5_propagator((void*)CC::d_elem,sizeof(Float));
}
*/

//--------------------------------//
// class QKXTM_Contraction //
//--------------------------------//

/*
#define N_MESONS 10
template<typename Float>
void QKXTM_Contraction<Float>::
contractMesons(QKXTM_Propagator<Float> &prop1,
	       QKXTM_Propagator<Float> &prop2, 
	       char *filename_out, int isource){
  
  errorQuda("contractMesons: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");

  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractMesons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractMesons: Will perform in double precision\n");
  
  Float (*corr_mom_local)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_MESONS] =(Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  Float (*corr_mom)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_totalL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
  
  if( corr_mom_local == NULL || 
      corr_mom_local_reduced == NULL || 
      corr_mom == NULL )
    errorQuda("Error problem to allocate memory");
  
  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractMesons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float),MOMENTUM_SPACE);
  }
  
  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local, corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_MESONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %+e %+e \t %+e %+e\n",
		  ip,it,
		  GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip], 
		  corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip]);
	}
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

#define N_BARYONS 10
template<typename Float>
void QKXTM_Contraction<Float>::
contractBaryons(QKXTM_Propagator<Float> &prop1,
		QKXTM_Propagator<Float> &prop2, 
		char *filename_out, int isource){
  
  errorQuda("contractBaryons: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");
  
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractBaryons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractBaryons: Will perform in double precision\n");

  Float (*corr_mom_local)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4])calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom_local_reduced)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  Float (*corr_mom)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_totalL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  
  if( corr_mom_local == NULL || 
      corr_mom_local_reduced == NULL || 
      corr_mom == NULL )
    errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++){
    run_contractBaryons(texProp1,texProp2,(void*) corr_mom_local,it,isource,sizeof(Float),MOMENTUM_SPACE);
  }

  int error;

  if( typeid(Float) == typeid(float) ){
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_FLOAT,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corr_mom_local,corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,MPI_SUM,0, GK_spaceComm);
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corr_mom_local_reduced,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,corr_mom,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,MPI_DOUBLE,0,GK_timeComm);
      if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
    }
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_BARYONS ; ip++)
    for(int it = 0 ; it < GK_totalL[3] ; it++)
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
    for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gammap = 0 ; gammap < 4 ; gammap++){
      int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
      int sign = (it+GK_sourcePosition[isource][3]) >= GK_totalL[3] ? -1 : +1;
      fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %d %d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],gamma,gammap,
	      sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][0][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+0][1][ip][gamma][gammap], sign*corr_mom[it_shift*GK_Nmoms*2+imom*2+1][1][ip][gamma][gammap]);
    }
    fclose(ptr_out);
  }

  free(corr_mom_local);
  free(corr_mom_local_reduced);
  free(corr_mom);
  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopBaryonsHDF5(void *twopBaryons, 
		     char *filename, 
		     qudaQKXTMinfo info, 
		     int isource){

  if(info.CorrSpace==MOMENTUM_SPACE)      
    writeTwopBaryonsHDF5_MomSpace((void*)twopBaryons,filename,info,isource);
  else if(info.CorrSpace==POSITION_SPACE) 
    writeTwopBaryonsHDF5_PosSpace((void*)twopBaryons,filename,info,isource);
  else errorQuda("writeTwopBaryonsHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

//-C.K. - New function to write the baryons two-point function in 
// HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopBaryonsHDF5_PosSpace(void *twopBaryons, 
			      char *filename, 
			      qudaQKXTMinfo info, 
			      int isource){

  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeTwopBaryonsHDF5_PosSpace: Support for writing the Baryon two-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeTwopBaryonsHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeTwopBaryonsHDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeTwopBuf;

  int Sdim = 7;
  int Sel = 16;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;
  // Size of the dataspace -> #Baryons, volume, spin, re-im
  hsize_t dims[7]  = {2,tL[3],tL[2],tL[1],tL[0],Sel,2};
  // Dimensions of the "local" dataspace, for each rank
  hsize_t ldims[7] = {2,lL[3],lL[2],lL[1],lL[0],Sel,2}; 
  // start position for each rank
  hsize_t start[7] = {0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0,0};


  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  // Attribute writing 
  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", 
			     type_id, attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);

  //- Index identification-ordering, precision
  char *corr_info;
  asprintf(&corr_info,"Position-space baryon 2pt-correlator\nIndex Order: [flav, t, z, y, x, spin, real/imag]\nSpin-index order: Row-major\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", 
			       type_id_2, attrdat_id_2, 
			       H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  for(int bar=0;bar<N_BARYONS;bar++){
    char *group3_tag;
    asprintf(&group3_tag,"%s",info.baryon_type[bar]);
    hid_t group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
    hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
    hid_t dataset_id = H5Dcreate(group3_id, "twop-baryon", 
				 DATATYPE_H5, filespace, H5P_DEFAULT, 
				 H5P_DEFAULT, H5P_DEFAULT);
    filespace = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    writeTwopBuf = &(((Float*)twopBaryons)[2*Sel*lV*2*bar]);

    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
			     filespace, plist_id, writeTwopBuf);
    if(status<0) errorQuda("writeTwopBaryonsHDF5_PosSpace: Unsuccessful writing of the dataset. Exiting\n");

    H5Dclose(dataset_id);
    H5Pclose(plist_id);
    H5Sclose(subspace);
    H5Sclose(filespace);
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);

}

//-C.K. - New function to write the baryons two-point function in 
//HDF5 format, momentum-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopBaryonsHDF5_MomSpace(void *twopBaryons, 
			      char *filename, 
			      qudaQKXTMinfo info, 
			      int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopBaryonsHDF5_MomSpace: Support for writing the Baryon two-point function only in momentum-space!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopBaryonsHDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;

    hsize_t dims[3] = {T,16,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[3];
    ldims[1] = dims[1];
    ldims[2] = dims[2];
    if(GK_timeRank==src_rank) ldims[0] = h;
    else ldims[0] = Lt;

    //-Determine the start position for each rank
    hsize_t start[3];
    // if src_rank = sink_rank then this is the same
    if(GK_timeRank==src_rank) start[0] = 0; 
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
	if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }
    start[1] = 0; //
    start[2] = 0; //-These are common among all ranks

    for(int bar=0;bar<N_BARYONS;bar++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.baryon_type[bar]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int imom=0;imom<GK_Nmoms;imom++){
	char *group4_tag;
	asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",
		 GK_moms[imom][0],
		 GK_moms[imom][1],
		 GK_moms[imom][2]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
	
	hid_t filespace  = H5Screate_simple(3, dims, NULL);
	hid_t subspace   = H5Screate_simple(3, ldims, NULL);

	for(int ip=0;ip<2;ip++){
	  char *dset_tag;
	  asprintf(&dset_tag,"twop_baryon_%d",ip+1);

	  hid_t dataset_id = H5Dcreate(group4_id, dset_tag, DATATYPE_H5, 
				       filespace, H5P_DEFAULT, H5P_DEFAULT, 
				       H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
			      NULL, ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopBaryons)[2*16*tail + 2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);
	  else writeTwopBuf = &(((Float*)twopBaryons)[2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);
	
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace,
				   filespace, plist_id, writeTwopBuf);
	  
	  H5Dclose(dataset_id);
	  H5Pclose(plist_id);
	}//-ip
	H5Sclose(subspace);
	H5Sclose(filespace);
	H5Gclose(group4_id);
      }//-imom
      H5Gclose(group3_id);
    }//-bar

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = 16;
      ldims[2] = 2;
      start[0] = T-tail;
      start[1] = 0;
      start[2] = 0;

      for(int bar=0;bar<N_BARYONS;bar++){
	for(int imom=0;imom<GK_Nmoms;imom++){
	  char *group_tag;
	  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s/mom_xyz_%+d_%+d_%+d",
		   info.traj,
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3],
		   info.baryon_type[bar],
		   GK_moms[imom][0],
		   GK_moms[imom][1],
		   GK_moms[imom][2]);  
	  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
	  
	  for(int ip=0;ip<2;ip++){
	    char *dset_tag;
	    asprintf(&dset_tag,"twop_baryon_%d",ip+1);

	    hid_t dset_id  = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
	    hid_t mspace_id  = H5Screate_simple(3, ldims, NULL);
	    hid_t dspace_id = H5Dget_space(dset_id);

	    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, 
				NULL, ldims, NULL);
	  
	    tailBuf = &(((Float*)twopBaryons)[2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip]);

	    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);

	    H5Dclose(dset_id);
	    H5Sclose(mspace_id);
	    H5Sclose(dspace_id);
	  }
	  H5Gclose(group_id);
	}//-imom
      }//-bar

      H5Fclose(file_idt);
    }//-tail!=0
  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}

//-C.K. - New function to copy the baryon two-point functions into write 
// Buffers for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction<Float>::
copyTwopBaryonsToHDF5_Buf(void *Twop_baryons_HDF5, 
			  void *corrBaryons, 
			  int isource, 
			  CORR_SPACE CorrSpace){

  int Lt = GK_localL[3];
  int SpVol = GK_localVolume/Lt;
  int t_src = GK_sourcePosition[isource][3];

  if(CorrSpace==MOMENTUM_SPACE){
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){      

    for(int ip=0;ip<2;ip++){
      for(int bar=0;bar<N_BARYONS;bar++){
	for(int imom=0;imom<GK_Nmoms;imom++){
	  for(int it=0;it<Lt;it++){
	    int t_glob = GK_timeRank*Lt+it;
	    int sign = t_glob < t_src ? -1 : +1;
	    for(int ga=0;ga<4;ga++){
	      for(int gap=0;gap<4;gap++){
		int im=gap+4*ga;
		((Float*)Twop_baryons_HDF5)[0 + 2*im + 2*16*it + 2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[0 + 2*imom + 2*GK_Nmoms*it][ip][bar][ga][gap];
		((Float*)Twop_baryons_HDF5)[1 + 2*im + 2*16*it + 2*16*Lt*imom + 2*16*Lt*GK_Nmoms*bar + 2*16*Lt*GK_Nmoms*N_BARYONS*ip] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[1 + 2*imom + 2*GK_Nmoms*it][ip][bar][ga][gap];
	      }}}}}
    }//-ip

    }//-if GK_timeRank
  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){

    for(int ip=0;ip<2;ip++){
      for(int bar=0;bar<N_BARYONS;bar++){
	for(int ga=0;ga<4;ga++){
	  for(int gap=0;gap<4;gap++){
	    int im=gap+4*ga;
	    for(int it=0;it<Lt;it++){
	      int t_glob = comm_coords(default_topo)[3]*Lt+it;
	      int sign = t_glob < t_src ? -1 : +1;
	      for(int sv=0;sv<SpVol;sv++){
		((Float*)Twop_baryons_HDF5)[0 + 2*im + 2*16*sv + 2*16*SpVol*it + 2*16*SpVol*Lt*ip + 2*16*SpVol*Lt*2*bar] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[0 + 2*sv + 2*SpVol*it][ip][bar][ga][gap];
		((Float*)Twop_baryons_HDF5)[1 + 2*im + 2*16*sv + 2*16*SpVol*it + 2*16*SpVol*Lt*ip + 2*16*SpVol*Lt*2*bar] = sign*((Float(*)[2][N_BARYONS][4][4])corrBaryons)[1 + 2*sv + 2*SpVol*it][ip][bar][ga][gap];
	      }}}}}
    }//-ip
    
  }//-else if

}


//-C.K. New function to write the baryons two-point function in ASCII format
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopBaryons_ASCII(void *corrBaryons, 
		       char *filename_out, 
		       int isource, 
		       CORR_SPACE CorrSpace){

  if(CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopBaryons_ASCII: Supports writing only in momentum-space!\n");

  Float (*GLcorrBaryons)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_totalL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
  if( GLcorrBaryons == NULL )errorQuda("writeTwopBaryons_ASCII: Cannot allocate memory for Baryon two-point function buffer.");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeTwopBaryons_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeTwopBaryons_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*)corrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,
		       DATATYPE,GLcorrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,
		       DATATYPE,0,GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_BARYONS ; ip++)
    for(int it = 0 ; it < GK_totalL[3] ; it++)
    for(int imom = 0 ; imom < GK_Nmoms ; imom++)
    for(int gamma = 0 ; gamma < 4 ; gamma++)
    for(int gammap = 0 ; gammap < 4 ; gammap++){
      int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
      int sign = (it+GK_sourcePosition[isource][3]) 
	>= GK_totalL[3] ? -1 : +1;
      fprintf(ptr_out,
	      "%d \t %d \t %+d %+d %+d \t %d %d \t %+e %+e \t %+e %+e\n",
	      ip,it,
	      GK_moms[imom][0],
	      GK_moms[imom][1],
	      GK_moms[imom][2],gamma,gammap,
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+0][0][ip][gamma][gammap], 
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+1][0][ip][gamma][gammap],
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+0][1][ip][gamma][gammap], 
     sign*GLcorrBaryons[it_shift*GK_Nmoms*2+imom*2+1][1][ip][gamma][gammap]);
    }
    fclose(ptr_out);
  }
  
  free(GLcorrBaryons);
}

//-C.K. Overloaded function to perform the baryon contractions without writing the data
template<typename Float>
void QKXTM_Contraction<Float>::
contractBaryons(QKXTM_Propagator<Float> &prop1,
		QKXTM_Propagator<Float> &prop2, 
		void *corrBaryons, int isource, 
		CORR_SPACE CorrSpace){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractBaryons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractBaryons: Will perform in double precision\n");
  
  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++) 
      run_contractBaryons(texProp1,texProp2,
			  (void*) corrBaryons,it,
			  isource,sizeof(Float),CorrSpace);
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    Float (*corrBaryons_local)[2][N_BARYONS][4][4] = (Float(*)[2][N_BARYONS][4][4]) calloc(GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,sizeof(Float));
    if( corrBaryons_local == NULL ) errorQuda("contractBaryons: Cannot allocate memory for Baryon two-point function contract buffer.\n");

    for(int it = 0 ; it < GK_localL[3] ; it++) 
      run_contractBaryons(texProp1,texProp2,
			  (void*) corrBaryons_local,it,
			  isource,sizeof(Float),CorrSpace);
    
    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;

    MPI_Reduce(corrBaryons_local, (Float*) corrBaryons,GK_localL[3]*GK_Nmoms*2*N_BARYONS*4*4*2,DATATYPE,MPI_SUM,0, GK_spaceComm);

    free(corrBaryons_local);
  }
  else errorQuda("contractBaryons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

//--------------------------------------------------------//
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopMesonsHDF5(void *twopMesons, 
		    char *filename, 
		    qudaQKXTMinfo info, 
		    int isource){

  if(info.CorrSpace==MOMENTUM_SPACE) 
    writeTwopMesonsHDF5_MomSpace((void*)twopMesons, filename, info, isource);
  else if(info.CorrSpace==POSITION_SPACE) 
    writeTwopMesonsHDF5_PosSpace((void*)twopMesons, filename, info, isource);
  else errorQuda("writeTwopMesonsHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}

//-C.K. - New function to write the mesons two-point function in HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopMesonsHDF5_PosSpace(void *twopMesons, 
			     char *filename, 
			     qudaQKXTMinfo info, 
			     int isource){
  
  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeTwopMesonsHDF5_PosSpace: Support for writing the Meson two-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeTwopMesonsHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeTwopMesonsHDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeTwopBuf;

  int Sdim = 6;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;

  // Size of the dataspace -> #Baryons, volume, spin, re-im
  hsize_t dims[6]  = {2,tL[3],tL[2],tL[1],tL[0],2}; 
  // Dimensions of the "local" dataspace, for each rank
  hsize_t ldims[6] = {2,lL[3],lL[2],lL[1],lL[0],2}; 
  // start position for each rank
  hsize_t start[6] = {0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0};
  
  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);

  // Attribute writing 
  //- Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", type_id, 
			     attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);

  //- Index identification-ordering, precision
  char *corr_info;
  asprintf(&corr_info,"Position-space meson 2pt-correlator\nIndex Order: [flav, t, z, y, x, real/imag]\nPrecision: %s\0",(typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", type_id_2, 
			       attrdat_id_2, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  for(int mes=0;mes<N_MESONS;mes++){
    char *group3_tag;
    asprintf(&group3_tag,"%s",info.meson_type[mes]);
    hid_t group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
    hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
    hid_t dataset_id = H5Dcreate(group3_id, "twop-meson", DATATYPE_H5, 
				 filespace, H5P_DEFAULT, H5P_DEFAULT, 
				 H5P_DEFAULT);
    filespace = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    writeTwopBuf = &(((Float*)twopMesons)[2*lV*2*mes]);

    herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, 
			     plist_id, writeTwopBuf);
    if(status<0) errorQuda("writeTwopMesonsHDF5_PosSpace: Unsuccessful writing of the dataset. Exiting\n");

    H5Dclose(dataset_id);
    H5Pclose(plist_id);
    H5Sclose(subspace);
    H5Sclose(filespace);
    H5Gclose(group3_id);
  }

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);

}


//-C.K. - New function to write the mesons two-point function in 
// HDF5 format, momentum-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeTwopMesonsHDF5_MomSpace(void *twopMesons, 
			     char *filename, 
			     qudaQKXTMinfo info, 
			     int isource){

  if(info.CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopMesonsHDF5_MomSpace: Support for writing the Meson two-point function only in momentum-space!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeTwopMesonsHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeTwopMesons_HDF5_MomSpace: Will write in double precision\n");
    }

    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];

    int src_rank = t_src/Lt;
    int sink_rank = ((t_src-1)%T)/Lt;
    int h = Lt - t_src%Lt;
    int tail = t_src%Lt;

    Float *writeTwopBuf;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
    
    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);
    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;

    hsize_t dims[2] = {T,2}; // Size of the dataspace

    //-Determine the ldims for each rank (tail not taken into account)
    hsize_t ldims[2];
    ldims[1] = dims[1];
    if(GK_timeRank==src_rank) ldims[0] = h;
    else ldims[0] = Lt;

    //-Determine the start position for each rank
    hsize_t start[2];
    // if src_rank = sink_rank then this is the same
    if(GK_timeRank==src_rank) start[0] = 0;
    else{
      int offs;
      for(offs=0;offs<GK_nProc[3];offs++){
	if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
      }
      offs--;
      start[0] = h + offs*Lt;
    }
    start[1] = 0; //-This is common among all ranks

    for(int mes=0;mes<N_MESONS;mes++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",info.meson_type[mes]);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int imom=0;imom<GK_Nmoms;imom++){
	char *group4_tag;
	asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",
		 GK_moms[imom][0],
		 GK_moms[imom][1],
		 GK_moms[imom][2]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
	
	hid_t filespace  = H5Screate_simple(2, dims, NULL);
	hid_t subspace   = H5Screate_simple(2, ldims, NULL);

	for(int ip=0;ip<2;ip++){
	  char *dset_tag;
	  asprintf(&dset_tag,"twop_meson_%d",ip+1);

	  hid_t dataset_id = H5Dcreate(group4_id, dset_tag, DATATYPE_H5, 
				       filespace, H5P_DEFAULT, H5P_DEFAULT, 
				       H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
			      NULL, ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  if(GK_timeRank==src_rank) writeTwopBuf = &(((Float*)twopMesons)[2*tail + 2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);
	  else writeTwopBuf = &(((Float*)twopMesons)[2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);
	
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeTwopBuf);
	  
	  H5Dclose(dataset_id);
	  H5Pclose(plist_id);
	}//-ip
	H5Sclose(subspace);
	H5Sclose(filespace);
	H5Gclose(group4_id);
      }//-imom
      H5Gclose(group3_id);
    }//-mes

    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);

    //-Write the tail, sink_ranks's task
    if(tail!=0 && GK_timeRank==sink_rank){ 
      Float *tailBuf;

      hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

      ldims[0] = tail;
      ldims[1] = 2;
      start[0] = T-tail;
      start[1] = 0;

      for(int mes=0;mes<N_MESONS;mes++){
	for(int imom=0;imom<GK_Nmoms;imom++){
	  char *group_tag;
	  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/%s/mom_xyz_%+d_%+d_%+d",
		   info.traj,
		   GK_sourcePosition[isource][0],
		   GK_sourcePosition[isource][1],
		   GK_sourcePosition[isource][2],
		   GK_sourcePosition[isource][3],
		   info.meson_type[mes],
		   GK_moms[imom][0],
		   GK_moms[imom][1],
		   GK_moms[imom][2]);  
	  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
	  
	  for(int ip=0;ip<2;ip++){
	    char *dset_tag;
	    asprintf(&dset_tag,"twop_meson_%d",ip+1);
	    
	    hid_t dset_id  = H5Dopen(group_id, dset_tag, H5P_DEFAULT);
	    hid_t mspace_id  = H5Screate_simple(2, ldims, NULL);
	    hid_t dspace_id = H5Dget_space(dset_id);
	    
	    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, NULL, ldims, NULL);
	  
	    tailBuf = &(((Float*)twopMesons)[2*Lt*imom + 2*Lt*GK_Nmoms*mes + 2*Lt*GK_Nmoms*N_MESONS*ip]);

	    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, mspace_id, dspace_id, H5P_DEFAULT, tailBuf);
	    
	    H5Dclose(dset_id);
	    H5Sclose(mspace_id);
	    H5Sclose(dspace_id);
	  }
	  H5Gclose(group_id);
	}//-imom
      }//-mes

      H5Fclose(file_idt);
    }//-tail!=0

  }//-if GK_timeRank >=0 && GK_timeRank < GK_nProc[3]

}

//-C.K. - New function to copy the meson two-point functions into write 
//Buffers for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction<Float>::
copyTwopMesonsToHDF5_Buf(void *Twop_mesons_HDF5, 
			 void *corrMesons, 
			 CORR_SPACE CorrSpace){

  if(CorrSpace==MOMENTUM_SPACE){
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

      for(int ip=0;ip<2;ip++){
	for(int mes=0;mes<N_MESONS;mes++){
	  for(int imom=0;imom<GK_Nmoms;imom++){
	    for(int it=0;it<GK_localL[3];it++){
	      ((Float*)Twop_mesons_HDF5)[0 + 2*it + 2*GK_localL[3]*imom + 2*GK_localL[3]*GK_Nmoms*mes + 2*GK_localL[3]*GK_Nmoms*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[0 + 2*imom + 2*GK_Nmoms*it][ip][mes];
	      ((Float*)Twop_mesons_HDF5)[1 + 2*it + 2*GK_localL[3]*imom + 2*GK_localL[3]*GK_Nmoms*mes + 2*GK_localL[3]*GK_Nmoms*N_MESONS*ip] = ((Float(*)[2][N_MESONS])corrMesons)[1 + 2*imom + 2*GK_Nmoms*it][ip][mes];
	    }}}
      }//-ip

    }//-if GK_timeRank
  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){
    int Lv = GK_localVolume;

    for(int ip=0;ip<2;ip++){
      for(int mes=0;mes<N_MESONS;mes++){
	for(int v=0;v<Lv;v++){
	  ((Float*)Twop_mesons_HDF5)[0 + 2*v + 2*Lv*ip + 2*Lv*2*mes] = ((Float(*)[2][N_MESONS])corrMesons)[0 + 2*v][ip][mes];
	  ((Float*)Twop_mesons_HDF5)[1 + 2*v + 2*Lv*ip + 2*Lv*2*mes] = ((Float(*)[2][N_MESONS])corrMesons)[1 + 2*v][ip][mes];
	}}
    }//-ip

  }//-else if

}


//-C.K. New function to write the mesons two-point function in ASCII format
template<typename Float>
void QKXTM_Contraction<Float>::writeTwopMesons_ASCII(void *corrMesons, char *filename_out, int isource, CORR_SPACE CorrSpace){

  if(CorrSpace!=MOMENTUM_SPACE) errorQuda("writeTwopMesons_ASCII: Supports writing only in momentum-space!\n");

  Float (*GLcorrMesons)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_totalL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));;
  if( GLcorrMesons == NULL )errorQuda("writeTwopMesons_ASCII: Cannot allocate memory for Meson two-point function buffer.\n");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeTwopMesons_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeTwopMesons_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*) corrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,GLcorrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,0,GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  FILE *ptr_out = NULL;
  if(comm_rank() == 0){
    ptr_out = fopen(filename_out,"w");
    if(ptr_out == NULL) errorQuda("Error opening file for writing\n");
    for(int ip = 0 ; ip < N_MESONS ; ip++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  fprintf(ptr_out,"%d \t %d \t %+d %+d %+d \t %+e %+e \t %+e %+e\n",ip,it,GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2],
		  GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+0][0][ip], GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+1][0][ip],
		  GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+0][1][ip], GLcorrMesons[it_shift*GK_Nmoms*2+imom*2+1][1][ip]);
	}
    fclose(ptr_out);
  }

  free(GLcorrMesons);
}

//-C.K. Overloaded function to perform the meson contractions without 
//writing the data
template<typename Float>
void QKXTM_Contraction<Float>::
contractMesons(QKXTM_Propagator<Float> &prop1,
	       QKXTM_Propagator<Float> &prop2, 
	       void *corrMesons, 
	       int isource, 
	       CORR_SPACE CorrSpace){
  cudaTextureObject_t texProp1, texProp2;
  prop1.createTexObject(&texProp1);
  prop2.createTexObject(&texProp2);

  if( typeid(Float) == typeid(float))  
    printfQuda("contractMesons: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractMesons: Will perform in double precision\n");

  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++) {
      run_contractMesons(texProp1,
			 texProp2,
			 (void*) corrMesons,
			 it,isource,
			 sizeof(Float),CorrSpace);
    }
  }
  else if( CorrSpace==MOMENTUM_SPACE ){
    Float (*corrMesons_local)[2][N_MESONS] = (Float(*)[2][N_MESONS]) calloc(GK_localL[3]*GK_Nmoms*2*N_MESONS*2,sizeof(Float));
    if( corrMesons_local == NULL )errorQuda("contractMesons: Cannot allocate memory for Meson two-point function contract buffer.\n");

    for(int it = 0 ; it < GK_localL[3] ; it++) run_contractMesons(texProp1,texProp2,(void*) corrMesons_local,it,isource,sizeof(Float),CorrSpace);

    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;
    
    MPI_Reduce(corrMesons_local, (Float*)corrMesons,GK_localL[3]*GK_Nmoms*2*N_MESONS*2,DATATYPE,MPI_SUM,0, GK_spaceComm);

    free(corrMesons_local);
  }
  else errorQuda("contractMesons: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  prop1.destroyTexObject(texProp1);
  prop2.destroyTexObject(texProp2);
}

//--------------------------------------------------------//

template<typename Float>
void QKXTM_Contraction<Float>::
seqSourceFixSinkPart1(QKXTM_Vector<Float> &vec, 
		      QKXTM_Propagator3D<Float> &prop1, 
		      QKXTM_Propagator3D<Float> &prop2, 
		      int tsinkMtsource, int nu, int c2, 
		      WHICHPROJECTOR PID, 
		      WHICHPARTICLE testParticle){
  
  cudaTextureObject_t tex1,tex2;
  prop1.createTexObject(&tex1);
  prop2.createTexObject(&tex2);

  run_seqSourceFixSinkPart1(vec.D_elem(), tsinkMtsource, tex1, 
			    tex2, nu, c2, PID, testParticle, sizeof(Float));
  
  prop1.destroyTexObject(tex1);
  prop2.destroyTexObject(tex2);
  checkCudaError();
  
}

template<typename Float>
void QKXTM_Contraction<Float>::
seqSourceFixSinkPart2(QKXTM_Vector<Float> &vec, 
		      QKXTM_Propagator3D<Float> &prop, 
		      int tsinkMtsource, int nu, int c2, 
		      WHICHPROJECTOR PID, 
		      WHICHPARTICLE testParticle){
  cudaTextureObject_t tex;
  prop.createTexObject(&tex);
  
  run_seqSourceFixSinkPart2(vec.D_elem(), tsinkMtsource, tex, 
			    nu, c2, PID, testParticle, sizeof(Float));
  
  prop.destroyTexObject(tex);
  
  checkCudaError();
}

template<typename Float>
void QKXTM_Contraction<Float>::
writeThrpHDF5(void *Thrp_local_HDF5, 
	      void *Thrp_noether_HDF5, 
	      void **Thrp_oneD_HDF5, 
	      char *filename, 
	      qudaQKXTMinfo info, 
	      int isource, 
	      WHICHPARTICLE NUCLEON){
  
  if(info.CorrSpace==MOMENTUM_SPACE)      
    writeThrpHDF5_MomSpace((void*) Thrp_local_HDF5, 
			   (void*) Thrp_noether_HDF5, 
			   (void**)Thrp_oneD_HDF5, 
			   filename, info, 
			   isource, NUCLEON);  
  else if(info.CorrSpace==POSITION_SPACE) 
    writeThrpHDF5_PosSpace((void*) Thrp_local_HDF5, 
			   (void*) Thrp_noether_HDF5, 
			   (void**)Thrp_oneD_HDF5, 
			   filename, info, 
			   isource, NUCLEON);
  else errorQuda("writeThrpHDF5: Unsupported value for info.CorrSpace! Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

}


//-C.K. - New function to write the three-point function in HDF5 format, position-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeThrpHDF5_PosSpace(void *Thrp_local_HDF5, 
		       void *Thrp_noether_HDF5, 
		       void **Thrp_oneD_HDF5, 
		       char *filename, 
		       qudaQKXTMinfo info, 
		       int isource, 
		       WHICHPARTICLE NUCLEON){
  
  if(info.CorrSpace!=POSITION_SPACE) errorQuda("writeThrpHDF5_PosSpace: Support for writing the three-point function only in position-space!\n");

  hid_t DATATYPE_H5;
  if( typeid(Float) == typeid(float) ){
    DATATYPE_H5 = H5T_NATIVE_FLOAT;
    printfQuda("writeThrpHDF5_PosSpace: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE_H5 = H5T_NATIVE_DOUBLE;
    printfQuda("writeThrp_HDF5_PosSpace: Will write in double precision\n");
  }

  Float *writeThrpBuf;

  int Nsink = info.Ntsink;
  int pc[4];
  int tL[4];
  int lL[4];
  for(int i=0;i<4;i++){
    pc[i] = comm_coords(default_topo)[i];
    tL[i] = GK_totalL[i];
    lL[i] = GK_localL[i];
  }
  int lV = GK_localVolume;

  hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
  H5Pclose(fapl_id);

  char *group1_tag;
  asprintf(&group1_tag,"conf_%04d",info.traj);
  hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  char *group2_tag;
  asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Attribute writing 
  //- Need these further on
  char *dir_order,*operator_list;
  asprintf(&dir_order,"[0,1,2,3] = [x,y,z,t]");
  asprintf(&operator_list,"0 = g5\n1 = gx\n2 = gy\n3 = gz\n4 = g0\n5 = Unity\n6 = g5gx\n7 = g5gy\n8 = g5gz\n9 = g5g0\n10 = g5sixy\n11 = g5sixz\n12 = g5siyz\n13 = g5si0x\n14 = g5si0y\n15 = g5si0z\n");

  //- Write the Source position
  char *src_pos;
  asprintf(&src_pos," [x, y, z, t] = [%02d, %02d, %02d, %02d]\0",
	   GK_sourcePosition[isource][0],
	   GK_sourcePosition[isource][1],
	   GK_sourcePosition[isource][2],
	   GK_sourcePosition[isource][3]);
  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  hid_t type_id = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id, strlen(src_pos));
  hid_t attr_id = H5Acreate2(group2_id, "source-position", 
			     type_id, attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id, type_id, src_pos);
  H5Aclose(attr_id);
  H5Tclose(type_id);
  H5Sclose(attrdat_id);
  
  //- Write general Correlator Info
  char *corr_info;
  asprintf(&corr_info,"Position-space %s 3pt-correlator\nIncludes ultra-local and one-derivative operators, noether current\nPrecision: %s\0",
	   (NUCLEON==PROTON)?"proton":"neutron",
	   (typeid(Float) == typeid(float)) ? "single" : "double");
  hid_t attrdat_id_2 = H5Screate(H5S_SCALAR);
  hid_t type_id_2 = H5Tcopy(H5T_C_S1);
  H5Tset_size(type_id_2, strlen(corr_info));
  hid_t attr_id_2 = H5Acreate2(file_id, "Correlator-info", type_id_2, 
			       attrdat_id_2, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr_id_2, type_id_2, corr_info);
  H5Aclose(attr_id_2);
  H5Tclose(type_id_2);
  H5Sclose(attrdat_id_2);
  //------------------------------------------------------------

  hid_t group3_id;
  hid_t group4_id;
  hid_t group5_id;

  for(int its=0;its<Nsink;its++){
    int tsink = info.tsinkSource[its];
    char *group3_tag;
    asprintf(&group3_tag,"tsink_%02d",tsink);
    group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			  H5P_DEFAULT, H5P_DEFAULT);
    
    for(int ipr=0;ipr<info.Nproj[its];ipr++){
      char *group4_tag;
      asprintf(&group4_tag,"proj_%s",
	       info.thrp_proj_type[info.proj_list[its][ipr]]);
      group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      for(int thrp_int=0;thrp_int<3;thrp_int++){
	THRP_TYPE type = (THRP_TYPE) thrp_int;

	char *group5_tag;
	asprintf(&group5_tag,"%s", info.thrp_type[thrp_int]);
	group5_id = H5Gcreate(group4_id, group5_tag, 
			      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	if(type==THRP_LOCAL){
	  char *attr_info;
	  asprintf(&attr_info,"Ultra-local operators:\nIndex-order: [operator, up-0/down-1, t, z, y, x, real/imag]\nOperator list:\n%s\0",operator_list);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "Ultra-local Info", 
				       type_id_c, attrdat_id_c, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 16;
	  int Sdim = 7;
	  // Size of the dataspace -> Operator, up-down, localVolume, Re-Im
	  hsize_t dims[7]  = {Mel, 2, tL[3], tL[2], tL[1], tL[0], 2};
	  // Dimensions of the local dataspace for each rank
	  hsize_t ldims[7] = {Mel, 2, lL[3], lL[2], lL[1], lL[0], 2}; 
	  hsize_t start[7] = {0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
	  
	  writeThrpBuf = &(((Float*)Thrp_local_HDF5)[2*lV*2*Mel*its + 
						     2*lV*2*Mel*Nsink*ipr]);
	  
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//-ultra_local
	else if(type==THRP_NOETHER){
	  char *attr_info;
	  asprintf(&attr_info,"Noether current:\nIndex-order: [direction, up-0/down-1, t, z, y, x, real/imag]\nDirection order:\n%s\0",dir_order);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "Noether current Info", type_id_c, attrdat_id_c, H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 4;
	  int Sdim = 7;
	  // Size of the dataspace -> Operator, up-down, localVolume, Re-Im
	  hsize_t dims[7]  = {Mel,2,tL[3],tL[2],tL[1],tL[0],2}; 
	  // Dimensions of the local dataspace for each rank
	  hsize_t ldims[7] = {Mel,2,lL[3],lL[2],lL[1],lL[0],2}; 
	  hsize_t start[7] = {0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  writeThrpBuf=&(((Float*)Thrp_noether_HDF5)[2*lV*2*Mel*its + 
						     2*lV*2*Mel*Nsink*ipr]);
	  
	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, 
				   filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//- noether
	else if(type==THRP_ONED){
	  char *attr_info;
	  asprintf(&attr_info,"One-derivative operators:\nIndex-order: [direction, operator, up-0/down-1, t, z, y, x, real/imag]\nOperator list:%s\nDirection order:\n%s\0", operator_list,dir_order);
	  hid_t attrdat_id_c = H5Screate(H5S_SCALAR);
	  hid_t type_id_c = H5Tcopy(H5T_C_S1);
	  H5Tset_size(type_id_c, strlen(attr_info));
	  hid_t attr_id_c = H5Acreate2(group5_id, "One-derivative Info", 
				       type_id_c, attrdat_id_c, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  H5Awrite(attr_id_c, type_id_c, attr_info);
	  H5Aclose(attr_id_c);
	  H5Tclose(type_id_c);
	  H5Sclose(attrdat_id_c);

	  int Mel = 16;
	  int Sdim = 8;
	  
	  hsize_t dims[8]  = {4,Mel,2,tL[3],tL[2],tL[1],tL[0],2}; // Size of the dataspace -> Direction, Operator, up-down, localVolume, Re-Im
	  hsize_t ldims[8] = {4,Mel,2,lL[3],lL[2],lL[1],lL[0],2}; // Dimensions of the local dataspace for each rank
	  hsize_t start[8] = {0,0,0,pc[3]*lL[3],pc[2]*lL[2],pc[1]*lL[1],pc[0]*lL[0],0}; // start position for each rank

	  hid_t filespace  = H5Screate_simple(Sdim, dims,  NULL);
	  hid_t subspace   = H5Screate_simple(Sdim, ldims, NULL);
	  hid_t dataset_id = H5Dcreate(group5_id, "threep", DATATYPE_H5, 
				       filespace, H5P_DEFAULT, 
				       H5P_DEFAULT, H5P_DEFAULT);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, 
			      ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  writeThrpBuf = NULL;
	  if( (writeThrpBuf = (Float*) malloc(2*lV*2*Mel*4*sizeof(Float))) == NULL ) 
	    errorQuda("writeThrpHDF5_PosSpace: Cannot allocate writeBuffer for one-derivative three-point correlator\n");

	  for(int dir=0;dir<4;dir++) memcpy(&(writeThrpBuf[2*lV*2*Mel*dir]), &(((Float*)Thrp_oneD_HDF5[dir])[2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*ipr]), 2*lV*2*Mel*sizeof(Float));

	  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);
	  if(status<0) errorQuda("writeThrpHDF5_PosSpace: Unsuccessful writing of the %s dataset. Exiting\n",info.thrp_type[thrp_int]);

	  free(writeThrpBuf);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);
	}//- oneD

	H5Gclose(group5_id);
      }//-thrp_int
      H5Gclose(group4_id);
    }//-ipr
    H5Gclose(group3_id);
  }//-its

  H5Gclose(group2_id);
  H5Gclose(group1_id);
  H5Fclose(file_id);


  return;
}


//-C.K. - New function to write the three-point function in HDF5 format, 
// momentum-space
template<typename Float>
void QKXTM_Contraction<Float>::
writeThrpHDF5_MomSpace(void *Thrp_local_HDF5, 
		       void *Thrp_noether_HDF5, 
		       void **Thrp_oneD_HDF5, 
		       char *filename, 
		       qudaQKXTMinfo info, 
		       int isource, 
		       WHICHPARTICLE NUCLEON){
  
  if(info.CorrSpace!=MOMENTUM_SPACE) errorQuda("writeThrpHDF5_MomSpace: Support for writing the three-point function only in momentum-space!\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){

    hid_t DATATYPE_H5;
    if( typeid(Float) == typeid(float) ){
      DATATYPE_H5 = H5T_NATIVE_FLOAT;
      printfQuda("writeThrpHDF5_MomSpace: Will write in single precision\n");
    }
    if( typeid(Float) == typeid(double)){
      DATATYPE_H5 = H5T_NATIVE_DOUBLE;
      printfQuda("writeThrp_HDF5_MomSpace: Will write in double precision\n");
    }

    Float *writeThrpBuf;

    int Nsink = info.Ntsink;
    int t_src = GK_sourcePosition[isource][3];
    int Lt = GK_localL[3];
    int T  = GK_totalL[3];
    int Mel;

    int src_rank = t_src/Lt;
    int h = Lt - t_src%Lt;
    int w = t_src%Lt;

    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, GK_timeComm, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    char *group1_tag;
    asprintf(&group1_tag,"conf_%04d",info.traj);
    hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    char *group2_tag;
    asprintf(&group2_tag,"sx%02dsy%02dsz%02dst%02d",
	     GK_sourcePosition[isource][0],
	     GK_sourcePosition[isource][1],
	     GK_sourcePosition[isource][2],
	     GK_sourcePosition[isource][3]);

    hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);

    hid_t group3_id;
    hid_t group4_id;
    hid_t group5_id;
    hid_t group6_id;
    hid_t group7_id;
    hid_t group8_id;

    hsize_t dims[3],ldims[3],start[3];

    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      char *group3_tag;
      asprintf(&group3_tag,"tsink_%02d",tsink);
      group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, 
			    H5P_DEFAULT, H5P_DEFAULT);

      bool all_print = false;
      if( tsink >= (T - t_src%Lt) ) all_print = true;

      int sink_rank = ((t_src+tsink)%T)/Lt;
      int l = ((t_src+tsink)%T)%Lt + 1; //-Significant only for sink_rank

      //-Determine which processes will print for this tsink
      bool print_rank;
      if(all_print) print_rank = true;
      else{
	print_rank = false;
	for(int i=0;i<GK_nProc[3];i++){
	  if( GK_timeRank == ((src_rank+i)%GK_nProc[3]) ) print_rank = true;
	  if( ((src_rank+i)%GK_nProc[3]) == sink_rank ) break;
	}
      }
      
      //-Determine the start position for each rank
      if(print_rank){
	// if src_rank = sink_rank then this is the same
	if(GK_timeRank==src_rank) start[0] = 0; 
	else{
	  int offs;
	  for(offs=0;offs<GK_nProc[3];offs++){
	    if( GK_timeRank == ((src_rank+offs)%GK_nProc[3]) ) break;
	  }
	  offs--;
	  start[0] = h + offs*Lt;
	}
      }
      else start[0] = 0; // Need to set this to zero when a given rank does not print. Otherwise the dimensions will not fit   
      start[1] = 0; //
      start[2] = 0; //-These are common among all ranks

      for(int ipr=0;ipr<info.Nproj[its];ipr++){
	char *group4_tag;
	asprintf(&group4_tag,"proj_%s",
		 info.thrp_proj_type[info.proj_list[its][ipr]]);
	group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, 
			      H5P_DEFAULT, H5P_DEFAULT);
      
	for(int part=0;part<2;part++){
	  char *group5_tag;
	  asprintf(&group5_tag,"%s", (part==0) ? "up" : "down");
	  group5_id = H5Gcreate(group4_id, group5_tag, H5P_DEFAULT, 
				H5P_DEFAULT, H5P_DEFAULT);
	
	  for(int thrp_int=0;thrp_int<3;thrp_int++){
	    THRP_TYPE type = (THRP_TYPE) thrp_int;

	    char *group6_tag;
	    asprintf(&group6_tag,"%s", info.thrp_type[thrp_int]);
	    group6_id = H5Gcreate(group5_id, group6_tag, H5P_DEFAULT, 
				  H5P_DEFAULT, H5P_DEFAULT);
	  
	    //-Determine the global dimensions
	    if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
	    else if (type==THRP_NOETHER) Mel = 4;
	    else errorQuda("writeThrpHDF5_MomSpace: Undefined three-point function type.\n");
	    dims[0] = tsink+1;
	    dims[1] = Mel;
	    dims[2] = 2;

	    //-Determine ldims for print ranks
	    if(all_print){
	      ldims[1] = dims[1];
	      ldims[2] = dims[2];
	      if(GK_timeRank==src_rank) ldims[0] = h;
	      else ldims[0] = Lt;
	    }
	    else{
	      if(print_rank){
		ldims[1] = dims[1];
		ldims[2] = dims[2];
		if(src_rank != sink_rank){
		  if(GK_timeRank==src_rank) ldims[0] = h;
		  else if(GK_timeRank==sink_rank) ldims[0] = l;
		  else ldims[0] = Lt;
		}
		else ldims[0] = dims[0];
	      }
	      //- Non-print ranks get zero space
	      else for(int i=0;i<3;i++) ldims[i] = 0; 
	    }
	    
	    for(int imom=0;imom<GK_Nmoms;imom++){
	      char *group7_tag;
	      asprintf(&group7_tag,"mom_xyz_%+d_%+d_%+d",
		       GK_moms[imom][0],
		       GK_moms[imom][1],
		       GK_moms[imom][2]);
	      group7_id = H5Gcreate(group6_id, group7_tag, H5P_DEFAULT, 
				    H5P_DEFAULT, H5P_DEFAULT);
	    
	      if(type==THRP_ONED){
		for(int mu=0;mu<4;mu++){
		  char *group8_tag;
		  asprintf(&group8_tag,"dir_%02d",mu);
		  group8_id = H5Gcreate(group7_id, group8_tag, H5P_DEFAULT, 
					H5P_DEFAULT, H5P_DEFAULT);

		  hid_t filespace  = H5Screate_simple(3, dims, NULL);
		  hid_t dataset_id = H5Dcreate(group8_id, "threep", 
					       DATATYPE_H5, filespace, 
					       H5P_DEFAULT, H5P_DEFAULT, 
					       H5P_DEFAULT);
		  hid_t subspace   = H5Screate_simple(3, ldims, NULL);
		  filespace = H5Dget_space(dataset_id);
		  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, 
				      NULL, ldims, NULL);
		  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		  if(GK_timeRank==src_rank) writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*w + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		  else writeThrpBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		  herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, 
					   subspace, filespace, 
					   plist_id, writeThrpBuf);

		  H5Sclose(subspace);
		  H5Dclose(dataset_id);
		  H5Sclose(filespace);
		  H5Pclose(plist_id);

		  H5Gclose(group8_id);
		}//-mu	      
	      }//-if
	      else{
		Float *thrpBuf;
		if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
		else if(type==THRP_NOETHER) 
		  thrpBuf = (Float*)Thrp_noether_HDF5;
		
		hid_t filespace  = H5Screate_simple(3, dims, NULL);
		hid_t dataset_id = H5Dcreate(group7_id, "threep", 
					     DATATYPE_H5, filespace, 
					     H5P_DEFAULT, H5P_DEFAULT, 
					     H5P_DEFAULT);
		hid_t subspace   = H5Screate_simple(3, ldims, NULL);
		filespace = H5Dget_space(dataset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL,
				    ldims, NULL);
		hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		if(GK_timeRank==src_rank) writeThrpBuf = &(thrpBuf[2*Mel*w + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		else writeThrpBuf = &(thrpBuf[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		herr_t status = H5Dwrite(dataset_id, DATATYPE_H5, subspace, filespace, plist_id, writeThrpBuf);
	      
		H5Sclose(subspace);
		H5Dclose(dataset_id);
		H5Sclose(filespace);
		H5Pclose(plist_id);
	      }//-else	  
	      H5Gclose(group7_id);
	    }//-imom	 
	    H5Gclose(group6_id);
	  }//-thrp_int
	  H5Gclose(group5_id);
	}//-part
	H5Gclose(group4_id);
      }//-projector
      H5Gclose(group3_id);
    }//-its
    
    H5Gclose(group2_id);
    H5Gclose(group1_id);
    H5Fclose(file_id);


    for(int its=0;its<Nsink;its++){
      int tsink = info.tsinkSource[its];
      int l = ((t_src+tsink)%T)%Lt + 1;
      
      int sink_rank = ((t_src+tsink)%T)/Lt;
      
      // No need to write something else
      if( tsink < (T - t_src%Lt) ) continue; 
      
      if(GK_timeRank==sink_rank){
	Float *tailBuf;
    
	hid_t file_idt = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

	start[0] = tsink + 1 - l;
	start[1] = 0;
	start[2] = 0;

	for(int ipr=0;ipr<info.Nproj[its];ipr++){
	  for(int part=0;part<2;part++){
	    for(int thrp_int=0;thrp_int<3;thrp_int++){
	      THRP_TYPE type = (THRP_TYPE) thrp_int;
	    
	      //-Determine the global dimensions
	      if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
	      else if (type==THRP_NOETHER) Mel = 4;
	      else errorQuda("writeThrp_HDF5: Undefined three-point function type.\n");
	      dims[0] = tsink+1;
	      dims[1] = Mel;
	      dims[2] = 2;

	      ldims[0] = l;
	      ldims[1] = Mel;
	      ldims[2] = 2;

	      for(int imom=0;imom<GK_Nmoms;imom++){
		if(type==THRP_ONED){
		  for(int mu=0;mu<4;mu++){
		    char *group_tag;
		    asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s/mom_xyz_%+d_%+d_%+d/dir_%02d",
			     info.traj,
			     GK_sourcePosition[isource][0],
			     GK_sourcePosition[isource][1],
			     GK_sourcePosition[isource][2],
			     GK_sourcePosition[isource][3],
			     tsink, 
			     info.thrp_proj_type[info.proj_list[its][ipr]], 
			     (part==0) ? "up" : "down", 
			     info.thrp_type[thrp_int], 
			     GK_moms[imom][0], 
			     GK_moms[imom][1], 
			     GK_moms[imom][2], mu);
		    hid_t group_id = H5Gopen(file_idt, group_tag, 
					     H5P_DEFAULT);

		    hid_t dset_id  = H5Dopen(group_id, "threep", 
					     H5P_DEFAULT);
		    hid_t mspace_id = H5Screate_simple(3, ldims, NULL);
		    hid_t dspace_id = H5Dget_space(dset_id);

		    H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start, 
					NULL, ldims, NULL);

		    tailBuf = &(((Float*)Thrp_oneD_HDF5[mu])[2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*part + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);

		    herr_t status = H5Dwrite(dset_id, DATATYPE_H5, 
					     mspace_id, dspace_id, 
					     H5P_DEFAULT, tailBuf);

		    H5Dclose(dset_id);
		    H5Sclose(mspace_id);
		    H5Sclose(dspace_id);
		    H5Gclose(group_id);
		  }//-mu
		}
		else{
		  Float *thrpBuf;
		  if(type==THRP_LOCAL) thrpBuf = (Float*)Thrp_local_HDF5;
		  else if(type==THRP_NOETHER) 
		    thrpBuf = (Float*)Thrp_noether_HDF5;

		  char *group_tag;
		  asprintf(&group_tag,"conf_%04d/sx%02dsy%02dsz%02dst%02d/tsink_%02d/proj_%s/%s/%s/mom_xyz_%+d_%+d_%+d",info.traj,
			   GK_sourcePosition[isource][0],
			   GK_sourcePosition[isource][1],
			   GK_sourcePosition[isource][2],
			   GK_sourcePosition[isource][3],
			   tsink, 
			   info.thrp_proj_type[info.proj_list[its][ipr]], 
			   (part==0) ? "up" : "down", 
			   info.thrp_type[thrp_int], 
			   GK_moms[imom][0], 
			   GK_moms[imom][1], 
			   GK_moms[imom][2]);
		  hid_t group_id = H5Gopen(file_idt, group_tag, H5P_DEFAULT);
		  hid_t dset_id  = H5Dopen(group_id, "threep", H5P_DEFAULT);
		  hid_t mspace_id  = H5Screate_simple(3, ldims, NULL);
		  hid_t dspace_id = H5Dget_space(dset_id);

		  H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, 
				      start, NULL, ldims, NULL);
		
		  tailBuf = &(thrpBuf[2*Mel*Lt*imom + 
				      2*Mel*Lt*GK_Nmoms*part + 
				      2*Mel*Lt*GK_Nmoms*2*its + 
				      2*Mel*Lt*GK_Nmoms*2*Nsink*ipr]);
		  
		  herr_t status = H5Dwrite(dset_id, DATATYPE_H5, 
					   mspace_id, dspace_id, 
					   H5P_DEFAULT, tailBuf);
		  
		  H5Dclose(dset_id);
		  H5Sclose(mspace_id);
		  H5Sclose(dspace_id);
		  H5Gclose(group_id);
		}
	      }//-imom
	    }//-thrp_int
	  }//-part
	}//-projector
	H5Fclose(file_idt);
      }//-if GK_timeRank==sink_rank
    }//-its

  }//-if
}


//-C.K. - New function to copy the three-point data into write Buffers 
// for writing in HDF5 format
template<typename Float>
void QKXTM_Contraction<Float>::
copyThrpToHDF5_Buf(void *Thrp_HDF5, 
		   void *corrThp,  
		   int mu, int uORd, 
		   int its, int Nsink, 
		   int pr, int sign, 
		   THRP_TYPE type, 
		   CORR_SPACE CorrSpace){

  int Mel;
  if(type==THRP_LOCAL || type==THRP_ONED) Mel = 16;
  else if(type==THRP_NOETHER) Mel = 4;
  else errorQuda("Undefined THRP_TYPE passed to copyThrpToHDF5_Buf.\n");

  int Lt = GK_localL[3];

  if(CorrSpace==MOMENTUM_SPACE){
    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      if(type==THRP_LOCAL || type==THRP_NOETHER){
	for(int it = 0; it<Lt; it++){
	  for(int imom = 0; imom<GK_Nmoms; imom++){
	    for(int im = 0; im<Mel; im++){
	      ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*uORd + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*imom + 2*Mel*GK_Nmoms*it];
	      ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*uORd + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*imom + 2*Mel*GK_Nmoms*it];
	    }
	  }
	}
      }
      else if(type==THRP_ONED){
	for(int it = 0; it<Lt; it++){
	  for(int imom = 0; imom<GK_Nmoms; imom++){
	    for(int im = 0; im<Mel; im++){
	      ((Float*)Thrp_HDF5)[0 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*uORd + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[0 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*GK_Nmoms*it];
	      ((Float*)Thrp_HDF5)[1 + 2*im + 2*Mel*it + 2*Mel*Lt*imom + 2*Mel*Lt*GK_Nmoms*uORd + 2*Mel*Lt*GK_Nmoms*2*its + 2*Mel*Lt*GK_Nmoms*2*Nsink*pr] = sign*((Float*)corrThp)[1 + 2*im + 2*Mel*mu + 2*Mel*4*imom + 2*Mel*4*GK_Nmoms*it];
	    }
	  }
	}      
      }
    }//-if GK_timeRank
  }//-if CorrSpace
  else if(CorrSpace==POSITION_SPACE){
    int lV = GK_localVolume;
    Float *tmp3pt;
    if(type==THRP_LOCAL || type==THRP_NOETHER) tmp3pt = ((Float*)corrThp);
    else if(type==THRP_ONED) tmp3pt = &(((Float*)corrThp)[2*16*lV*mu]);

    for(int v = 0; v<lV; v++){
      for(int im = 0; im<Mel; im++){
	((Float*)Thrp_HDF5)[0 + 2*v + 2*lV*uORd + 2*lV*2*im + 2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*pr] = sign*tmp3pt[0 + 2*im + 2*Mel*v];

	((Float*)Thrp_HDF5)[1 + 2*v + 2*lV*uORd + 2*lV*2*im + 2*lV*2*Mel*its + 2*lV*2*Mel*Nsink*pr] = sign*tmp3pt[1 + 2*im + 2*Mel*v];
      }
    }
  }//-else if

}


//-C.K. - New function to write the three-point function in ASCII format
template<typename Float>
void QKXTM_Contraction<Float>::
writeThrp_ASCII(void *corrThp_local, 
		void *corrThp_noether,
		void *corrThp_oneD, 
		WHICHPARTICLE testParticle, 
		int partflag , 
		char *filename_out, 
		int isource, 
		int tsinkMtsource, 
		CORR_SPACE CorrSpace){
  
  if(CorrSpace!=MOMENTUM_SPACE) 
    errorQuda("writeThrp_ASCII: Supports writing only in momentum-space!\n");
  
  Float *GLcorrThp_local   = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16  *2,sizeof(Float));
  Float *GLcorrThp_noether = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms   *4*2,sizeof(Float));
  Float *GLcorrThp_oneD    = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16*4*2,sizeof(Float));
  if(GLcorrThp_local == NULL || 
     GLcorrThp_noether == NULL || 
     GLcorrThp_oneD == NULL) 
    errorQuda("writeThrp_ASCII: Cannot allocate memory for write Buffers.");

  MPI_Datatype DATATYPE = -1;
  if( typeid(Float) == typeid(float)){
    DATATYPE = MPI_FLOAT;
    printfQuda("writeThrp_ASCII: Will write in single precision\n");
  }
  if( typeid(Float) == typeid(double)){
    DATATYPE = MPI_DOUBLE;
    printfQuda("writeThrp_ASCII: Will write in double precision\n");
  }

  int error;
  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    error = MPI_Gather((Float*)corrThp_local,GK_localL[3]*GK_Nmoms*16*2, 
		       DATATYPE, GLcorrThp_local, 
		       GK_localL[3]*GK_Nmoms*16*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");

    error = MPI_Gather((Float*)corrThp_noether,GK_localL[3]*GK_Nmoms*4*2, 
		       DATATYPE, GLcorrThp_noether, 
		       GK_localL[3]*GK_Nmoms*4*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");

    error = MPI_Gather((Float*)corrThp_oneD,GK_localL[3]*GK_Nmoms*4*16*2, 
		       DATATYPE, GLcorrThp_oneD, 
		       GK_localL[3]*GK_Nmoms*4*16*2, 
		       DATATYPE, 0, GK_timeComm);
    if(error != MPI_SUCCESS) errorQuda("Error in MPI_gather");
  }

  char fname_local[257];
  char fname_noether[257];
  char fname_oneD[257];
  char fname_particle[257];
  char fname_upORdown[257];

  if(testParticle == PROTON){
    strcpy(fname_particle,"proton");
    if(partflag == 1)strcpy(fname_upORdown,"up");
    else if(partflag == 2)strcpy(fname_upORdown,"down");
    else errorQuda("writeThrp_ASCII: Got the wrong part! Should be either 1 or 2.");
  }
  else{
    strcpy(fname_particle,"neutron");
    if(partflag == 1)strcpy(fname_upORdown,"down");
    else if(partflag == 2)strcpy(fname_upORdown,"up");
    else errorQuda("writeThrp_ASCII: Got the wrong part! Should be either 1 or 2.");
  }

  sprintf(fname_local,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"ultra_local",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_noether,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"noether",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_oneD,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"oneD",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);

  FILE *ptr_local = NULL;
  FILE *ptr_noether = NULL;
  FILE *ptr_oneD = NULL;

  if( comm_rank() == 0 ){
    ptr_local = fopen(fname_local,"w");
    ptr_noether = fopen(fname_noether,"w");
    ptr_oneD = fopen(fname_oneD,"w");
    // local //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_local,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*GLcorrThp_local[it_shift*GK_Nmoms*16*2 + 
				       imom*16*2 + iop*2 + 0], 
		  sign*GLcorrThp_local[it_shift*GK_Nmoms*16*2 + 
				       imom*16*2 + iop*2 + 1]);
	}
    // noether //
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_noether,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*GLcorrThp_noether[it_shift*GK_Nmoms*4*2 + 
					 imom*4*2 + iop*2 + 0], 
		  sign*GLcorrThp_noether[it_shift*GK_Nmoms*4*2 + 
					 imom*4*2 + iop*2 + 1]);
	}
    // oneD //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int it = 0 ; it < GK_totalL[3] ; it++)
	  for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	    int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	    int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	      >= GK_totalL[3] ? -1 : +1;
	    fprintf(ptr_oneD,"%d \t %d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		    iop, dir, it, 
		    GK_moms[imom][0],
		    GK_moms[imom][1],
		    GK_moms[imom][2],
		    sign*GLcorrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
					imom*4*16*2 + dir*16*2 + iop*2 + 0], 
		    sign*GLcorrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
					imom*4*16*2 + dir*16*2 + iop*2 + 1]);
	  }
    fclose(ptr_local);
    fclose(ptr_noether);
    fclose(ptr_oneD);
  }

  free(GLcorrThp_local);
  free(GLcorrThp_noether);
  free(GLcorrThp_oneD);
}

//-C.K. Overloaded function to perform the contractions without 
// writing the data
template<typename Float>
void QKXTM_Contraction<Float>::
contractFixSink(QKXTM_Propagator<Float> &seqProp,
		QKXTM_Propagator<Float> &prop, 
		QKXTM_Gauge<Float> &gauge, 
		void *corrThp_local, void *corrThp_noether, 
		void *corrThp_oneD, 
		WHICHPROJECTOR typeProj , 
		WHICHPARTICLE testParticle, 
		int partflag, int isource, 
		CORR_SPACE CorrSpace){
  
  if( typeid(Float) == typeid(float))  
    printfQuda("contractFixSink: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractFixSink: Will perform in double precision\n");
  
  // seq prop apply gamma5 and conjugate
  seqProp.apply_gamma5();
  seqProp.conjugate();

  gauge.ghostToHost();
  // communicate gauge
  gauge.cpuExchangeGhost(); 
  gauge.ghostToDevice();
  comm_barrier();

  prop.ghostToHost();
  // communicate forward propagator
  prop.cpuExchangeGhost(); 
  prop.ghostToDevice();
  comm_barrier();

  seqProp.ghostToHost();
  // communicate sequential propagator
  seqProp.cpuExchangeGhost();
  seqProp.ghostToDevice();
  comm_barrier();

  cudaTextureObject_t seqTex, fwdTex, gaugeTex;
  seqProp.createTexObject(&seqTex);
  prop.createTexObject(&fwdTex);
  gauge.createTexObject(&gaugeTex);

  if(CorrSpace==POSITION_SPACE){
    for(int it = 0 ; it < GK_localL[3] ; it++)
      run_fixSinkContractions((void*)corrThp_local, 
			      (void*)corrThp_noether, 
			      (void*)corrThp_oneD, 
			      fwdTex, seqTex, gaugeTex, 
			      testParticle, 
			      partflag, it, isource, 
			      sizeof(Float), CorrSpace);
  }
  else if(CorrSpace==MOMENTUM_SPACE){
    Float *corrThp_local_local   = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*16*2,
						   sizeof(Float));

    Float *corrThp_noether_local = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*4*2,
						   sizeof(Float));

    Float *corrThp_oneD_local    = (Float*) calloc(GK_localL[3]*
						   GK_Nmoms*16*4*2,
						   sizeof(Float));
    
    if(corrThp_local_local == NULL || 
       corrThp_noether_local == NULL || 
       corrThp_oneD_local == NULL) 
      errorQuda("contractFixSink: Cannot allocate memory for three-point function contract buffers.\n");
    
    for(int it = 0 ; it < GK_localL[3] ; it++)
      run_fixSinkContractions(corrThp_local_local, 
			      corrThp_noether_local, 
			      corrThp_oneD_local, 
			      fwdTex, seqTex, gaugeTex, 
			      testParticle, 
			      partflag, it, isource, 
			      sizeof(Float), CorrSpace);

    MPI_Datatype DATATYPE = -1;
    if( typeid(Float) == typeid(float))  DATATYPE = MPI_FLOAT;
    if( typeid(Float) == typeid(double)) DATATYPE = MPI_DOUBLE;
    
    MPI_Reduce(corrThp_local_local, (Float*)corrThp_local, 
	       GK_localL[3]*GK_Nmoms*16*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_noether_local, (Float*)corrThp_noether, 
	       GK_localL[3]*GK_Nmoms*4*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_oneD_local, (Float*)corrThp_oneD, 
	       GK_localL[3]*GK_Nmoms*16*4*2, DATATYPE, 
	       MPI_SUM, 0, GK_spaceComm);
    
    free(corrThp_local_local);
    free(corrThp_noether_local);
    free(corrThp_oneD_local);
  }
  else errorQuda("contractFixSink: Supports only POSITION_SPACE and MOMENTUM_SPACE!\n");

  seqProp.destroyTexObject(seqTex);
  prop.destroyTexObject(fwdTex);
  gauge.destroyTexObject(gaugeTex);

}

//---------------------//

template<typename Float>
void QKXTM_Contraction<Float>::
contractFixSink(QKXTM_Propagator<Float> &seqProp,
		QKXTM_Propagator<Float> &prop, 
		QKXTM_Gauge<Float> &gauge, 
		WHICHPROJECTOR typeProj , 
		WHICHPARTICLE testParticle, 
		int partflag , 
		char *filename_out, 
		int isource, 
		int tsinkMtsource){
  
  errorQuda("contractFixSink: This version of the function is obsolete. Cannot guarantee correct results. Please call the overloaded-updated version of this function with the corresponding list of arguments.\n");

  if( typeid(Float) == typeid(float))  
    printfQuda("contractFixSink: Will perform in single precision\n");
  if( typeid(Float) == typeid(double)) 
    printfQuda("contractFixSink: Will perform in double precision\n");

  // seq prop apply gamma5 and conjugate
  // do the communication for gauge, prop and seqProp
  seqProp.apply_gamma5();
  seqProp.conjugate();

  gauge.ghostToHost();

  // communicate gauge
  gauge.cpuExchangeGhost();
  gauge.ghostToDevice();
  comm_barrier();

  prop.ghostToHost();

  // communicate forward propagator
  prop.cpuExchangeGhost(); 
  prop.ghostToDevice();
  comm_barrier();

  seqProp.ghostToHost();

  // communicate sequential propagator
  seqProp.cpuExchangeGhost(); 
  seqProp.ghostToDevice();
  comm_barrier();

  cudaTextureObject_t seqTex, fwdTex, gaugeTex;
  seqProp.createTexObject(&seqTex);
  prop.createTexObject(&fwdTex);
  gauge.createTexObject(&gaugeTex);

  Float *corrThp_local_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD_local = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local_local == NULL || 
     corrThp_noether_local == NULL || 
     corrThp_oneD_local == NULL) 
    errorQuda("Error problem to allocate memory");

  Float *corrThp_local_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD_reduced = 
    (Float*) calloc(GK_localL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local_reduced == NULL || 
     corrThp_noether_reduced == NULL || 
     corrThp_oneD_reduced == NULL) 
    errorQuda("Error problem to allocate memory");

  Float *corrThp_local = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*16*2,sizeof(Float));
  Float *corrThp_noether = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*4*2,sizeof(Float));
  Float *corrThp_oneD = 
    (Float*) calloc(GK_totalL[3]*GK_Nmoms*4*16*2,sizeof(Float));
  if(corrThp_local == NULL || 
     corrThp_noether == NULL || 
     corrThp_oneD == NULL) 
    errorQuda("Error problem to allocate memory");

  for(int it = 0 ; it < GK_localL[3] ; it++)
    run_fixSinkContractions(corrThp_local_local,
			    corrThp_noether_local,
			    corrThp_oneD_local,
			    fwdTex,seqTex,gaugeTex,
			    testParticle,partflag,it,
			    isource,sizeof(Float),MOMENTUM_SPACE);
  
  int error;
  if( typeid(Float) == typeid(float)){
    MPI_Reduce(corrThp_local_local, 
	       corrThp_local_reduced, 
	       GK_localL[3]*GK_Nmoms*16*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_noether_local, 
	       corrThp_noether_reduced, 
	       GK_localL[3]*GK_Nmoms*4*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    MPI_Reduce(corrThp_oneD_local, 
	       corrThp_oneD_reduced, 
	       GK_localL[3]*GK_Nmoms*4*16*2, 
	       MPI_FLOAT, MPI_SUM, 0, GK_spaceComm);

    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corrThp_local_reduced,
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_FLOAT, corrThp_local, 
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_noether_reduced,
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_FLOAT, corrThp_noether, 
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_oneD_reduced,
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_FLOAT, corrThp_oneD, 
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_FLOAT, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");
    }
  }
  else{
    MPI_Reduce(corrThp_local_local, corrThp_local_reduced, 
	       GK_localL[3]*GK_Nmoms*16*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);
    MPI_Reduce(corrThp_noether_local, corrThp_noether_reduced, 
	       GK_localL[3]*GK_Nmoms*4*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);
    MPI_Reduce(corrThp_oneD_local, corrThp_oneD_reduced, 
	       GK_localL[3]*GK_Nmoms*4*16*2, 
	       MPI_DOUBLE, MPI_SUM, 0, GK_spaceComm);

    if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
      error = MPI_Gather(corrThp_local_reduced,
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_DOUBLE, corrThp_local, 
			 GK_localL[3]*GK_Nmoms*16*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_noether_reduced,
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_DOUBLE, corrThp_noether, 
			 GK_localL[3]*GK_Nmoms*4*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");

      error = MPI_Gather(corrThp_oneD_reduced,
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_DOUBLE, corrThp_oneD, 
			 GK_localL[3]*GK_Nmoms*4*16*2, 
			 MPI_DOUBLE, 0, GK_timeComm);
      if(error != MPI_SUCCESS) 
	errorQuda("Error in MPI_gather");
    }
  }
  char fname_local[257];
  char fname_noether[257];
  char fname_oneD[257];
  char fname_particle[257];
  char fname_upORdown[257];

  if(testParticle == PROTON){
    strcpy(fname_particle,"proton");
    if(partflag == 1)strcpy(fname_upORdown,"up");
    else if(partflag == 2)strcpy(fname_upORdown,"down");
    else errorQuda("Error wrong part got");
  }
  else{
    strcpy(fname_particle,"neutron");
    if(partflag == 1)strcpy(fname_upORdown,"down");
    else if(partflag == 2)strcpy(fname_upORdown,"up");
    else errorQuda("Error wrong part got");
  }

  sprintf(fname_local,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"ultra_local",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_noether,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"noether",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  sprintf(fname_oneD,"%s.%s.%s.%s.SS.%02d.%02d.%02d.%02d.dat",
	  filename_out,fname_particle,fname_upORdown,"oneD",
	  GK_sourcePosition[isource][0],
	  GK_sourcePosition[isource][1],
	  GK_sourcePosition[isource][2],
	  GK_sourcePosition[isource][3]);
  
  FILE *ptr_local = NULL;
  FILE *ptr_noether = NULL;
  FILE *ptr_oneD = NULL;

  if( comm_rank() == 0 ){
    ptr_local = fopen(fname_local,"w");
    ptr_noether = fopen(fname_noether,"w");
    ptr_oneD = fopen(fname_oneD,"w");
    // local //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_local,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*corrThp_local[it_shift*GK_Nmoms*16*2 + 
				     imom*16*2 + iop*2 + 0], 
		  sign*corrThp_local[it_shift*GK_Nmoms*16*2 + 
				     imom*16*2 + iop*2 + 1]);
	}
    // noether //
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int it = 0 ; it < GK_totalL[3] ; it++)
	for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	  int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	  int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	    >= GK_totalL[3] ? -1 : +1;
	  fprintf(ptr_noether,"%d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		  iop, it, 
		  GK_moms[imom][0],
		  GK_moms[imom][1],
		  GK_moms[imom][2],
		  sign*corrThp_noether[it_shift*GK_Nmoms*4*2 + 
				       imom*4*2 + iop*2 + 0], 
		  sign*corrThp_noether[it_shift*GK_Nmoms*4*2 + 
				       imom*4*2 + iop*2 + 1]);
	}
    // oneD //
    for(int iop = 0 ; iop < 16 ; iop++)
      for(int dir = 0 ; dir < 4 ; dir++)
	for(int it = 0 ; it < GK_totalL[3] ; it++)
	  for(int imom = 0 ; imom < GK_Nmoms ; imom++){
	    int it_shift = (it+GK_sourcePosition[isource][3])%GK_totalL[3];
	    int sign = (tsinkMtsource+GK_sourcePosition[isource][3]) 
	      >= GK_totalL[3] ? -1 : +1;
	    fprintf(ptr_oneD,"%d \t %d \t %d \t %+d %+d %+d \t %+e %+e\n", 
		    iop, dir, it, 
		    GK_moms[imom][0],
		    GK_moms[imom][1],
		    GK_moms[imom][2],
		    sign*corrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
				      imom*4*16*2 + dir*16*2 + iop*2 + 0], 
		    sign*corrThp_oneD[it_shift*GK_Nmoms*4*16*2 + 
				      imom*4*16*2 + dir*16*2 + iop*2 + 1]);
	  }
    fclose(ptr_local);
    fclose(ptr_noether);
    fclose(ptr_oneD);
  }

  free(corrThp_local_local);
  free(corrThp_local_reduced);
  free(corrThp_local);

  free(corrThp_noether_local);
  free(corrThp_noether_reduced);
  free(corrThp_noether);

  free(corrThp_oneD_local);
  free(corrThp_oneD_reduced);
  free(corrThp_oneD);

  seqProp.destroyTexObject(seqTex);
  prop.destroyTexObject(fwdTex);
  gauge.destroyTexObject(gaugeTex);

}
*/

//----------------------------------//
// class QKXTM_ Propagator3D //
//----------------------------------//
/*
template<typename Float>
QKXTM_Propagator3D<Float>::
QKXTM_Propagator3D(ALLOCATION_FLAG alloc_flag, 
			  CLASS_ENUM classT): 
  QKXTM_Field<Float>(alloc_flag, classT){
  if(alloc_flag != BOTH)
    errorQuda("Propagator3D class is only implemented to allocate memory for both\n");
}

template<typename Float>
void QKXTM_Propagator3D<Float>::
absorbTimeSliceFromHost(QKXTM_Propagator<Float> &prop, 
			int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  
  for(int mu = 0 ; mu < 4 ; mu++)
  for(int nu = 0 ; nu < 4 ; nu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
  for(int c2 = 0 ; c2 < 3 ; c2++)
  for(int iv3 = 0 ; iv3 < V3 ; iv3++)
  for(int ipart = 0 ; ipart < 2 ; ipart++)
    CC::h_elem[ (mu*GK_nSpin*GK_nColor*GK_nColor*V3 + 
		 nu*GK_nColor*GK_nColor*V3 + 
		 c1*GK_nColor*V3 + 
		 c2*V3 + iv3)*2 + ipart] = 
      prop.H_elem()[(mu*GK_nSpin*GK_nColor*GK_nColor*GK_localVolume + 
		     nu*GK_nColor*GK_nColor*GK_localVolume + 
		     c1*GK_nColor*GK_localVolume + 
		     c2*GK_localVolume + 
		     timeslice*V3 + iv3)*2 + ipart];
  
  cudaMemcpy(CC::d_elem,CC::h_elem,
	     GK_nSpin*GK_nSpin*GK_nColor*GK_nColor*V3*2*sizeof(Float),
	     cudaMemcpyHostToDevice);
  checkCudaError();
}

template<typename Float>
void QKXTM_Propagator3D<Float>::
absorbTimeSlice(QKXTM_Propagator<Float> &prop, int timeslice){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;

  for(int mu=0; mu<4; mu++)
    for(int nu=0; nu<4; nu++)
      for(int c1=0; c1<3; c1++)
	for(int c2=0; c2<3; c2++){
	  pointer_dst = (CC::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
			 c1*3*V3*2 + c2*V3*2);
	  pointer_src = (prop.D_elem() + mu*4*3*3*GK_localVolume*2 + 
			 nu*3*3*GK_localVolume*2 + c1*3*GK_localVolume*2 + 
			 c2*GK_localVolume*2 + timeslice*V3*2);
	  cudaMemcpy(pointer_dst, pointer_src, V3*2*sizeof(Float), 
		     cudaMemcpyDeviceToDevice);
	}
  checkCudaError();
  pointer_src = NULL;
  pointer_dst = NULL;
}

template<typename Float>
void QKXTM_Propagator3D<Float>::
absorbVectorTimeSlice(QKXTM_Vector<Float> &vec, 
		      int timeslice, int nu, int c2){
  int V3 = GK_localVolume/GK_localL[3];
  Float *pointer_src = NULL;
  Float *pointer_dst = NULL;
  
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int c1 = 0 ; c1 < 3 ; c1++){
      pointer_dst = (CC::d_elem + mu*4*3*3*V3*2 + nu*3*3*V3*2 + 
		     c1*3*V3*2 + c2*V3*2);
      pointer_src = (vec.D_elem() + mu*3*GK_localVolume*2 + 
		     c1*GK_localVolume*2 + timeslice*V3*2);
      cudaMemcpy(pointer_dst, pointer_src, V3*2 * sizeof(Float), 
		 cudaMemcpyDeviceToDevice);
    }
}

template<typename Float>
void QKXTM_Propagator3D<Float>::broadcast(int tsink){
  cudaMemcpy(CC::h_elem , CC::d_elem , CC::bytes_total_length , 
	     cudaMemcpyDeviceToHost);
  checkCudaError();
  comm_barrier();
  int bcastRank = tsink/GK_localL[3];
  int V3 = GK_localVolume/GK_localL[3];
  if( typeid(Float) == typeid(float) ){
    int error = MPI_Bcast(CC::h_elem , 4*4*3*3*V3*2 , MPI_FLOAT , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");
  }
  else if( typeid(Float) == typeid(double) ){
    int error = MPI_Bcast(CC::h_elem , 4*4*3*3*V3*2 , MPI_DOUBLE , 
			  bcastRank , GK_timeComm );
    if(error != MPI_SUCCESS)errorQuda("Error in mpi broadcasting");    
  }
  cudaMemcpy(CC::d_elem , CC::h_elem , CC::bytes_total_length, 
	     cudaMemcpyHostToDevice);
  checkCudaError();
}
*/

/*
#ifdef HAVE_ARPACK
//------------------------------//
// class QKXTM_Deflation_Kelper //
//------------------------------//

//-C.K. Constructor for the even-odd operator functions
template<typename Float>
QKXTM_Deflation<Float>::
QKXTM_Deflation(int N_EigenVectors, bool isEven): 
  h_elem(NULL), eigenValues(NULL){
  if(GK_init_qudaQKXTM_flag == false)
    errorQuda("You must initialize QKXTM library first\n");
  NeV=N_EigenVectors;
  if(NeV == 0){
    warningQuda("You chose zero eigenVectors\n");
    return;
  }

  isEv=isEven;
  isFullOp = false;

  field_length = 4*3;

  total_length_per_NeV = (GK_localVolume/2)*field_length;
  bytes_total_length_per_NeV = total_length_per_NeV*2*sizeof(Float);
  total_length = NeV*(GK_localVolume/2)*field_length;
  bytes_total_length = total_length*2*sizeof(Float);

  h_elem = (Float*)malloc(NeV*bytes_total_length_per_NeV);
  if(h_elem == NULL) errorQuda("Error: Out of memory for eigenVectors.\n");
  memset(h_elem,0,NeV*bytes_total_length_per_NeV);

  eigenValues = (Float*)malloc(2*NeV*sizeof(Float));
  if(eigenValues == NULL)errorQuda("Error with allocation host memory for deflation class\n");
}


//-C.K. Constructor for the full Operator functions
template<typename Float>
QKXTM_Deflation<Float>::
QKXTM_Deflation(QudaInvertParam *param, 
		       qudaQKXTM_arpackInfo arpackInfo): 
  h_elem(NULL), eigenValues(NULL), diracOp(NULL){
  if(GK_init_qudaQKXTM_flag == false)
    errorQuda("You must initialize QKXTM library first\n");

  PolyDeg = arpackInfo.PolyDeg;
  NeV = arpackInfo.nEv;
  NkV = arpackInfo.nKv;
  // for which part of the spectrum we want to solve
  spectrumPart = arpackInfo.spectrumPart; 

  isACC = arpackInfo.isACC;
  tolArpack = arpackInfo.tolArpack;
  maxIterArpack = arpackInfo.maxIterArpack;
  strcpy(arpack_logfile,arpackInfo.arpack_logfile);
  amin = arpackInfo.amin;
  amax = arpackInfo.amax;
  isEv = arpackInfo.isEven;
  isFullOp = arpackInfo.isFullOp;
  flavor_sign = param->twist_flavor;

  if(NeV == 0){
    printfQuda("###############################\n");
    printfQuda("######### Got NeV = 0 #########\n");
    printfQuda("###############################\n");
    return;
  }

  invert_param = param;
  if(isFullOp) invert_param->solve_type = QUDA_NORMOP_SOLVE;
  else invert_param->solve_type = QUDA_NORMOP_PC_SOLVE;

  field_length = 4*3;

  fullorHalf = (isFullOp) ? 1 : 2;

  total_length_per_NeV = (GK_localVolume/fullorHalf)*field_length*2;
  bytes_total_length_per_NeV = total_length_per_NeV*sizeof(Float);
  //NeV*(GK_localVolume/fullorHalf)*field_length;
  total_length =  NeV*total_length_per_NeV;   
  //total_length*2*sizeof(Float);
  bytes_total_length = NeV*bytes_total_length_per_NeV; 
  
  h_elem = (Float*)malloc(NkV*bytes_total_length_per_NeV);
  if(h_elem == NULL) errorQuda("Error: Out of memory for eigenVectors.\n");
  memset(h_elem,0,NkV*bytes_total_length_per_NeV);

  eigenValues = (Float*)malloc(2*NkV*sizeof(Float));
  if(eigenValues == NULL)errorQuda("Error: Out of memory of eigenValues.\n");

  DiracParam diracParam;
  setDiracParam(diracParam,invert_param,!isFullOp);
  diracOp = Dirac::create(diracParam);
}

template<typename Float>
QKXTM_Deflation<Float>::~QKXTM_Deflation(){
  if(NeV == 0)return;

  free(h_elem);
  free(eigenValues);
  if(diracOp != NULL) delete diracOp;
}

template<typename Float>
void QKXTM_Deflation<Float>::printInfo(){
  printfQuda("\n======= DEFLATION INFO =======\n"); 
  if(isFullOp){
    printfQuda(" The EigenVectors are for the Full %smu operator\n", 
	       (flavor_sign==QUDA_TWIST_PLUS) ? "+" : "-");
  }
  else{
    printfQuda(" Will calculate EigenVectors for the %s %smu operator\n", 
	       isEv ? "even-even" : "odd-odd", 
	       (flavor_sign==QUDA_TWIST_PLUS) ? "+" : "-" );
  }

  printfQuda(" Number of requested EigenVectors is %d in precision %d\n",
	     NeV,(int) sizeof(Float));
  printfQuda(" The Size of Krylov space is %d\n",NkV);

  printfQuda(" Allocated Gb for the eigenVectors space for each node are %lf and the pointer is %p\n", NeV * ( (double)bytes_total_length_per_NeV/((double) 1024.*1024.*1024.) ),h_elem);
  printfQuda("==============================\n");
}
//==================================================

template<typename Float>
void QKXTM_Deflation<Float>::ApplyMdagM(Float *vec_out, 
					       Float *vec_in, 
					       QudaInvertParam *param){

  bool opFlag;

  if(isFullOp){
    printfQuda("Applying the Full Operator\n");
    opFlag = false;

    cudaColorSpinorField *in    = NULL;
    cudaColorSpinorField *out   = NULL;
    
    QKXTM_Vector<double> *Kvec = 
      new QKXTM_Vector<double>(BOTH,VECTOR);

    ColorSpinorParam cpuParam((void*)vec_in,*param,GK_localL,opFlag);
    ColorSpinorParam cudaParam(cpuParam, *param);

    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    in  = new cudaColorSpinorField(cudaParam);
    out = new cudaColorSpinorField(cudaParam);
    
    Kvec->packVector(vec_in);
    Kvec->loadVector();
    Kvec->uploadToCuda(in,opFlag);
    
    diracOp->MdagM(*out,*in);
    
    Kvec->downloadFromCuda(out,opFlag);
    Kvec->unloadVector();
    Kvec->unpackVector();
    
    memcpy(vec_out,Kvec->H_elem(),bytes_total_length_per_NeV);

    delete in;
    delete out;
    delete Kvec;
  }
  else{
    printfQuda("Applying the %s Operator\n",isEv ? "Even-Even" : "Odd-Odd");

    cudaColorSpinorField *in = NULL;
    cudaColorSpinorField *out = NULL;

    opFlag = isEv;
    bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
      (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

    ColorSpinorParam cpuParam((void*)vec_in,*param,GK_localL,pc_solution);

    ColorSpinorField *h_b = 
      (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
    
    cpuParam.v = vec_out;
    ColorSpinorField *h_x = 
      (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    ColorSpinorParam cudaParam(cpuParam, *param);
    cudaParam.create = QUDA_COPY_FIELD_CREATE;
    in = new cudaColorSpinorField(*h_b, cudaParam);
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    out = new cudaColorSpinorField(cudaParam);

    QKXTM_Vector<double> *Kvec = 
      new QKXTM_Vector<double>(BOTH,VECTOR);

    Kvec->packVector(vec_in);
    Kvec->loadVector();
    Kvec->uploadToCuda(in,opFlag);
    
    diracOp->MdagM(*out,*in);
    
    Kvec->downloadFromCuda(out,opFlag);
    Kvec->unloadVector();
    Kvec->unpackVector();
    
    memcpy(vec_out,Kvec->H_elem(),bytes_total_length_per_NeV);

    delete in;
    delete out;
    delete h_b;
    delete h_x;
    delete Kvec;
  }

  printfQuda("ApplyMdagM: Completed successfully\n");
}
//==================================================

template<typename Float>
void QKXTM_Deflation<Float>::MapEvenOddToFull(){

  if(!isFullOp){ warningQuda("MapEvenOddToFull: This function only works with the Full Operator\n");
    return;
  }

  if(NeV==0) return;

  size_t bytes_eo = bytes_total_length_per_NeV/2;
  int size_eo = total_length_per_NeV/2;

  int site_size = 4*3*2;
  size_t bytes_per_site = site_size*sizeof(Float);

  if((bytes_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid bytes for eo vector\n");
  if((size_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid size for eo vector\n");

  Float *vec_odd = (Float*) malloc(bytes_eo);
  Float *vec_evn = (Float*) malloc(bytes_eo);

  if(vec_odd==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_odd\n");
  if(vec_evn==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_evn\n");

  printfQuda("MapEvenOddToFull: Vecs allocated\n");

  for(int i=0;i<NeV;i++){
    // Save the even half of the eigenvector
    memcpy(vec_evn,&(h_elem[i*total_length_per_NeV]),bytes_eo); 
    // Save the odd half of the eigenvector
    memcpy(vec_odd,&(h_elem[i*total_length_per_NeV+size_eo]),bytes_eo); 

    int k=0;
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++){
	    int oddBit     = (x+y+z+t) & 1;
	    if(oddBit) memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			      &(vec_odd[site_size*(k/2)]),bytes_per_site);
	    else       memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			      &(vec_evn[site_size*(k/2)]),bytes_per_site);
	    k++;
	  }	  
  }

  printfQuda("MapEvenOddToFull: Completed successfully\n");
}

//-For a single vector
template<typename Float>
void QKXTM_Deflation<Float>::MapEvenOddToFull(int i){

  if(!isFullOp) errorQuda("MapEvenOddToFull: This function only works with the Full Operator\n");

  if(NeV==0) return;

  size_t bytes_eo = bytes_total_length_per_NeV/2;
  int size_eo = total_length_per_NeV/2;

  int site_size = 4*3*2;
  size_t bytes_per_site = site_size*sizeof(Float);

  if((bytes_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid bytes for eo vector\n");
  if((size_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid size for eo vector\n");

  Float *vec_odd = (Float*) malloc(bytes_eo);
  Float *vec_evn = (Float*) malloc(bytes_eo);

  if(vec_odd==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_odd\n");
  if(vec_evn==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_evn\n");

  printfQuda("MapEvenOddToFull: Vecs allocated\n");

  // Save the even half of the eigenvector
  memcpy(vec_evn,&(h_elem[i*total_length_per_NeV]),bytes_eo); 
  // Save the odd half of the eigenvector
  memcpy(vec_odd,&(h_elem[i*total_length_per_NeV+size_eo]),bytes_eo);

  int k=0;
  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++){
	  int oddBit     = (x+y+z+t) & 1;
	  if(oddBit) memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			    &(vec_odd[site_size*(k/2)]),bytes_per_site);
	  else       memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			    &(vec_evn[site_size*(k/2)]),bytes_per_site);
	  k++;
	}	  
  
  printfQuda("MapEvenOddToFull: Vector %d completed successfully\n",i);
}


template<typename Float>
void QKXTM_Deflation<Float>::
copyEigenVectorToQKXTM_Vector(int eigenVector_id, Float *vec){
  if(NeV == 0)return;
  
  if(!isFullOp){
    printfQuda("Copying elements of Eigenvector %d according to %s Operator format\n", eigenVector_id, isEv ? "even-even" : "odd-odd");
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++)
	    for(int mu=0; mu<4; mu++)
	      for(int c1=0; c1<3; c1++)
		{
		  int oddBit     = (x+y+z+t) & 1;
		  if(oddBit){
		    if(isEv == false){
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		    else{
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		    }
		  } // if for odd
		  else{
		    if(isEv == true){
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		    else{
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		    }
		  }
		}
  }//-isFullOp check
  else if(isFullOp){
    printfQuda("Copying elements of Eigenvector %d according to Full Operator format\n",eigenVector_id);
    memcpy(vec,&(h_elem[eigenVector_id*total_length_per_NeV]),
	   bytes_total_length_per_NeV);
  }

}

template<typename Float>
void QKXTM_Deflation<Float>::
copyEigenVectorFromQKXTM_Vector(int eigenVector_id,Float *vec){
  if(NeV == 0)return;
  
  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++)
	    for(int mu=0; mu<4; mu++)
	      for(int c1=0; c1<3; c1++)
		{
		  int oddBit     = (x+y+z+t) & 1;
		  if(oddBit){
		    if(isEv == false){
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 0];
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		  } // if for odd
		  else{
		    if(isEv == true){
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 0];
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		  }
		}
  }//-isFullOp check
  else if(isFullOp){
    memcpy(&(h_elem[eigenVector_id*total_length_per_NeV]),
	   vec,bytes_total_length_per_NeV);
  }

}

template<typename Float>
void QKXTM_Deflation<Float>::copyToEigenVector(Float *vec, 
						      Float *vals){
  memcpy(&(h_elem[0]), vec, bytes_total_length);
  memcpy(&(eigenValues[0]), vals, NeV*2*sizeof(Float));
}


//-C.K: This member function performs the operation vec_defl = U (\Lambda)^(-1) U^dag vec_in
template <typename Float>
void QKXTM_Deflation<Float>::
deflateVector(QKXTM_Vector<Float> &vec_defl, 
	      QKXTM_Vector<Float> &vec_in){
  if(NeV == 0){
    vec_defl.zero_device();
    return;
  }
  
  Float *tmp_vec = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec_lex = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float));
  Float *out_vec = (Float*) calloc(NeV*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV*2,sizeof(Float)) ;
  
  if(tmp_vec        == NULL || 
     tmp_vec_lex    == NULL || 
     out_vec        == NULL || 
     out_vec_reduce == NULL)
    errorQuda("Error with memory allocation in deflation method\n");
  
  Float *tmp_vec_even = tmp_vec;
  Float *tmp_vec_odd = tmp_vec + (GK_localVolume/2)*4*3*2;
  
  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit = (x+y+z+t) & 1;
	if(oddBit){
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			  z*GK_localL[1]*GK_localL[0] + 
			  y*GK_localL[0] + 
			  x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	  (Float) vec_in.H_elem()[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] +
				   z*GK_localL[1]*GK_localL[0] + 
				   y*GK_localL[0] + 
				   x)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
	else{
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	  (Float) vec_in.H_elem()[(t*GK_localL[2]*GK_localL[1]*GK_localL[0]+ 
				   z*GK_localL[1]*GK_localL[0] + 
				   y*GK_localL[0] + 
				   x)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
      }  
  }  
  else if(isFullOp){
    memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV);
  }

  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;

  Float *ptr_elem = NULL;

  if(!isFullOp){
    if(isEv == true){
      ptr_elem = tmp_vec_even;
    }
    else{
      ptr_elem = tmp_vec_odd;
    }
  }
  else{
    ptr_elem = tmp_vec;
  }


  if( typeid(Float) == typeid(float) ){
    //-C.K: out_vec = H_elem^dag * ptr_elem -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, ptr_elem, incx, 
		(void*) beta, out_vec, incy ); 
    //-C.K_CHECK: This might not be needed
    memset(ptr_elem,0,NN*2*sizeof(Float));
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,
		  MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      //-Eigenvalues are real!
      out_vec_reduce[i*2+0] /= eigenValues[i*2+0]; 
      //-C.K: out_vec_reduce -> \Lambda^(-1) * U^dag * vec_in
      out_vec_reduce[i*2+1] /= eigenValues[i*2+0]; 
    }
    //-C.K: ptr_elem = H_elem * out_vec_reduce -> ptr_elem = U * \Lambda^(-1) * U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );
  }
  else if ( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, ptr_elem, incx, 
		(void*) beta, out_vec, incy );
    //-C.K_CHECK: This might not be needed
    memset(ptr_elem,0,NN*2*sizeof(Float)); 
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,
		  MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[2*i+0];
      out_vec_reduce[i*2+1] /= eigenValues[2*i+0];
    }
    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );    
  }
  
  

  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit     = (x+y+z+t) & 1;
	if(oddBit){
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_lex[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			 z*GK_localL[1]*GK_localL[0] + 
			 y*GK_localL[0] + 
			 x)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	      tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			    z*GK_localL[1]*GK_localL[0] + 
			    y*GK_localL[0] + 
			    x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
	else{
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_lex[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			 z*GK_localL[1]*GK_localL[0] + 
			 y*GK_localL[0] + 
			 x)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	      tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
      }  
  }
  else{
    //    memcpy(tmp_vec_lex,tmp_vec,bytes_total_length_per_NeV);
    memcpy(tmp_vec_lex,ptr_elem,bytes_total_length_per_NeV);
  }

  vec_defl.packVector((Float*) tmp_vec_lex);
  vec_defl.loadVector();


  free(out_vec);
  free(out_vec_reduce);
  free(tmp_vec);
  free(tmp_vec_lex);

  //  printfQuda("deflateVector: Deflation of the initial guess completed succesfully\n");
}


template<typename Float>
void QKXTM_Deflation<Float>::
writeEigenVectors_ASCII(char *prefix_path){
  if(NeV == 0)return;
  char filename[257];
  if(comm_rank() != 0) return;
  FILE *fid;
  //int n_elem_write = 240;
  int n_elem_write = (GK_localVolume/fullorHalf)*4*3;
  for(int nev = 0 ; nev < NeV ; nev++){
    sprintf(filename,"%s.%04d.txt",prefix_path,nev);
    fid = fopen(filename,"w");		  
    for(int ir = 0 ; ir < n_elem_write ; ir++)
      fprintf(fid,"%+e %+e\n",h_elem[nev*total_length_per_NeV + ir*2 + 0], 
	      h_elem[nev*total_length_per_NeV + ir*2 + 1]);
    
    fclose(fid);
  }
}


template<typename Float>
void QKXTM_Deflation<Float>::
polynomialOperator(cudaColorSpinorField &out, 
		   const cudaColorSpinorField &in){
  
  if(typeid(Float) != typeid(double)) errorQuda("Single precision is not implemented in member function of polynomial operator\n");

  double delta,theta;
  double sigma,sigma1,sigma_old;
  double d1,d2,d3;

  double a = amin;
  double b = amax;

  delta = (b-a)/2.0;
  theta = (b+a)/2.0;

  sigma1 = -delta/theta;
  blas::copy(out,in);

  if( PolyDeg == 0 ){
    printfQuda("Got degree of the polynomial to be 0. Proceeding anyway.\n");
    return;
  }
  
  d1 =  sigma1/delta;
  d2 =  1.0;

  //  (*matDiracOp)(out,in); //!!!!! check if I need (2*k)^2
  diracOp->MdagM(out,in); //!!!!! check if I need (2*k)^2
  blas::axpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);

  if( PolyDeg == 1 )
    return;

  cudaColorSpinorField *tm1 = new cudaColorSpinorField(in);
  cudaColorSpinorField *tm2 = new cudaColorSpinorField(in);

  blas::copy(*tm1,in);
  blas::copy(*tm2,out);

  sigma_old = sigma1;

  for(int i=2; i <= PolyDeg; i++){
    sigma = 1.0/(2.0/sigma1-sigma_old);
    
    d1 = 2.0*sigma/delta;
    d2 = -d1*theta;
    d3 = -sigma*sigma_old;
    
    //    (*matDiracOp)( out, *tm2); //!!!!! check if I need (2*k)^2
    diracOp->MdagM( out, *tm2); //!!!!! check if I need (2*k)^2
    // axCuda(1./(2.*shift),out);
    
    blas::ax(d3,*tm1);
    std::complex<double> d1c(d1,0);
    std::complex<double> d2c(d2,0);
    blas::cxpaypbz(*tm1,d2c,*tm2,d1c,out);
    blas::copy(*tm1,*tm2);
    blas::copy(*tm2,out);
    sigma_old  = sigma;
  }

  delete tm1;
  delete tm2;

}

#include <sortingFunctions.h>
#include <arpackHeaders.h>

template<typename Float>
void QKXTM_Deflation<Float>::eigenSolver(){

  double t1,t2,t_ini,t_fin;

  if(NeV==0){
    printfQuda("eigenSolver: Got NeV=%d. Returning...\n",NeV);
    return;
  }

  //-print the input:

  char *which_evals_req;
  if (spectrumPart==SR)      which_evals_req = strdup("SR");
  else if (spectrumPart==LR) which_evals_req = strdup("LR");
  else if (spectrumPart==SM) which_evals_req = strdup("SM");
  else if (spectrumPart==LM) which_evals_req = strdup("LM");
  else if (spectrumPart==SI) which_evals_req = strdup("SI");
  else if (spectrumPart==LI) which_evals_req = strdup("LI");
  else{    
    errorQuda("eigenSolver: Option for spectrumPart is suspicious\n");
    exit(-1);
  }
  

  char *which_evals;
  if(isACC){
    if (spectrumPart==SR)      which_evals = strdup("LR");
    else if (spectrumPart==LR) which_evals = strdup("SR");
    else if (spectrumPart==SM) which_evals = strdup("LM");
    else if (spectrumPart==LM) which_evals = strdup("SM");
    else if (spectrumPart==SI) which_evals = strdup("LI");
    else if (spectrumPart==LI) which_evals = strdup("SI");
  }
  else{
    if (spectrumPart==SR)      which_evals = strdup("SR");
    else if (spectrumPart==LR) which_evals = strdup("LR");
    else if (spectrumPart==SM) which_evals = strdup("SM");
    else if (spectrumPart==LM) which_evals = strdup("LM");
    else if (spectrumPart==SI) which_evals = strdup("SI");
    else if (spectrumPart==LI) which_evals = strdup("LI");    
  }

  printfQuda("\neigenSolver: Input to ARPACK\n");
  printfQuda("========================================\n");
  printfQuda(" Number of Ritz eigenvalues requested: %d\n", NeV);
  printfQuda(" Size of Krylov space is: %d\n", NkV);
  printfQuda(" Part of the spectrum requested: %s\n", which_evals_req);
  printfQuda(" Part of the spectrum passed to ARPACK (may be different due to Poly. Acc.): %s\n", which_evals);
  printfQuda(" Polynomial acceleration: %s\n", isACC ? "yes" : "no");
  if(isACC) printfQuda(" Chebyshev polynomial paramaters: Degree = %d, amin = %+e, amax = %+e\n",PolyDeg,amin,amax); 
  printfQuda(" The convergence criterion is %+e\n", tolArpack);
  printfQuda(" Maximum number of iterations for ARPACK is %d\n",maxIterArpack);
  printfQuda("========================================\n\n");

  //--------------------------------------------//
  
  //- create the MPI communicator
#ifdef MPI_COMMS
  MPI_Fint mpi_comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif

  // control of the action taken by reverse communications 
  // (set initially to zero) 
  int ido=0;               
  // Specifies that the right hand side matrix should be the 
  // identity matrix; this makes the problem a standard eigenvalue problem.
  char *bmat=strdup("I");  
                               
  QudaInvertParam *param = invert_param;
 
  //- matrix dimensions 
  int LDV = (GK_localVolume/fullorHalf)*4*3;
  int N   = (GK_localVolume/fullorHalf)*4*3;
  printfQuda("eigenSolver: Number of complex elements: %d\n",LDV);

  //- Define all the necessary pointers
  std::complex<Float> *helem_cplx = NULL;
  helem_cplx = (std::complex<Float>*) &(h_elem[0]);

  std::complex<Float> *evals_cplx = NULL;
  evals_cplx = (std::complex<Float>*) &(eigenValues[0]);

  int *ipntr              = (int *) malloc(14 *sizeof(int));
  //since all Ritz vectors or Schur vectors are computed no need to 
  //initialize this array
  int *select             = (int *) malloc(NkV*sizeof(int)); 
  int *sorted_evals_index = (int *) malloc(NkV*sizeof(int)); 
  int *iparam             = (int *) malloc(11 *sizeof(int));
  // always call the subroutine that computes orthonormal basis for 
  // the eigenvectors
  int rvec = 1;  
  // just allocate more space
  int lworkl = (3*NkV*NkV+5*NkV)*2;

  //always compute orthonormal basis
  char *howmany = strdup("P"); 

  double *rwork        = (double *) malloc(NkV*sizeof(double));
  //will be used to sort the eigenvalues
  double *sorted_evals = (double *) malloc(NkV*sizeof(double)); 
  
  std::complex<Float> *resid  = 
    (std::complex<Float> *) malloc(LDV   *sizeof(std::complex<Float>));
  std::complex<Float> *workd  = 
    (std::complex<Float> *) malloc(3*LDV *sizeof(std::complex<Float>)); 
  std::complex<Float> *workl  = 
    (std::complex<Float> *) malloc(lworkl*sizeof(std::complex<Float>));
  std::complex<Float> *workev = 
    (std::complex<Float> *) malloc(2*NkV *sizeof(std::complex<Float>));
  std::complex<Float> sigma;

  if(resid == NULL)  errorQuda("eigenSolver: not enough memory for resid allocation in eigenSolver.\n");
  if(iparam == NULL) errorQuda("eigenSolver: not enough memory for iparam allocation in eigenSolver.\n");

  if((ipntr == NULL) || 
     (workd==NULL) || 
     (workl==NULL) || 
     (rwork==NULL) || 
     (select==NULL) || 
     (workev==NULL) || 
     (sorted_evals==NULL) || 
     (sorted_evals_index==NULL)){
    errorQuda("eigenSolver: not enough memory for ipntr,workd,workl,rwork,select,workev,sorted_evals,sorted_evals_index in eigenSolver.\n");
  }
  
  iparam[0] = 1;  //use exact shifts
  iparam[2] = maxIterArpack;
  iparam[3] = 1;
  iparam[6] = 1;

  double d1,d2,d3;

  int info;
  //means use a random starting vector with Arnoldi
  info = 0;               
  
  int i,j;

  // Code added to print the log of ARPACK  
  int arpack_log_u = 9999;

#ifndef MPI_COMMS
  if ( NULL != arpack_logfile ) {
    // correctness of this code depends on alignment in Fortran and C 
    // being the same ; if you observe crashes, disable this part 
    
    _AFT(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
    int msglvl0 = 0,
      msglvl1 = 1,
      msglvl2 = 2,
      msglvl3 = 3;
    _AFT(mcinitdebug)(
		      &arpack_log_u,      //logfil
		      &msglvl3,           //mcaupd
		      &msglvl3,           //mcaup2
		      &msglvl0,           //mcaitr
		      &msglvl3,           //mceigh
		      &msglvl0,           //mcapps
		      &msglvl0,           //mcgets
		      &msglvl3            //mceupd
		      );
    
    printfQuda("eigenSolver: Log info:\n");
    printfQuda(" ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
    printfQuda(" output is directed to %s\n",arpack_logfile);
  }
#else  
  if ( NULL != arpack_logfile && (comm_rank() == 0) ) {
    // correctness of this code depends on alignment in Fortran and C 
    // being the same ; if you observe crashes, disable this part 
    _AFT(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
    int msglvl0 = 0,
      msglvl1 = 1,
      msglvl2 = 2,
      msglvl3 = 3;
    _AFT(pmcinitdebug)(
		       &arpack_log_u,      //logfil
		       &msglvl3,           //mcaupd
		       &msglvl3,           //mcaup2
		       &msglvl0,           //mcaitr
		       &msglvl3,           //mceigh
		       &msglvl0,           //mcapps
		       &msglvl0,           //mcgets
		       &msglvl3            //mceupd
		       );
    
    printfQuda("eigenSolver: Log info:\n");
    printfQuda(" ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
    printfQuda(" output is directed to %s\n",arpack_logfile);
  }
#endif   

  cpuColorSpinorField *h_v = NULL;
  cudaColorSpinorField *d_v = NULL;

  cpuColorSpinorField *h_v2 = NULL;
  cudaColorSpinorField *d_v2 = NULL;

  int nconv;

  //M A I N   L O O P (Reverse communication)  

  bool checkIdo = true;

  t_ini = MPI_Wtime();

  do{

#ifndef MPI_COMMS 
    _AFT(znaupd)(&ido,"I", &N, which_evals, &NeV, &tolArpack, resid, &NkV,
		 helem_cplx, &N, iparam, ipntr, workd, 
		 workl, &lworkl,rwork,&info,1,2); 
#else
    _AFT(pznaupd)(&mpi_comm_f, &ido,"I", &N, which_evals, 
		  &NeV, &tolArpack, resid, &NkV,
		  helem_cplx, &N, iparam, ipntr, workd, 
		  workl, &lworkl,rwork,&info,1,2);
#endif

    if(checkIdo){
      // !!!!!!! please check that ipntr[0] does not change 
      ColorSpinorParam cpuParam(workd+ipntr[0]-1,*param,GK_localL,!isFullOp);
      h_v = new cpuColorSpinorField(cpuParam);
      cpuParam.v=workd+ipntr[1]-1;
      h_v2 = new cpuColorSpinorField(cpuParam);

      ColorSpinorParam cudaParam(cpuParam, *param);
      cudaParam.create = QUDA_ZERO_FIELD_CREATE;
      d_v = new cudaColorSpinorField( cudaParam);
      d_v2 = new cudaColorSpinorField( cudaParam);
      checkIdo = false;
    }
	
    if (ido == 99 || info == 1)
      break;

    if( (ido==-1) || (ido==1) ){
      *d_v = *h_v;
      if(isACC){
	polynomialOperator(*d_v2,*d_v);
      }
      else{
	diracOp->MdagM(*d_v2,*d_v);
      }
      *h_v2= *d_v2;
    }

  } while (ido != 99);
  
  //Check for convergence 
  
  if ( (info) < 0 ){
    printfQuda("eigenSolver: Error with _naupd, info = %d\n", info);
  }
  else{ 
    nconv = iparam[4];
    printfQuda("eigenSolver: Number of converged eigenvalues: %d\n", nconv);
    t_fin = MPI_Wtime();
    printfQuda("eigenSolver: TIME_REPORT - Eigenvalue calculation: %f sec\n"
	       ,t_fin-t_ini);
    printfQuda("eigenSolver: Computing eigenvectors...\n");
    t_ini = MPI_Wtime();

    //compute eigenvectors 
#ifndef MPI_COMMS
    _AFT(zneupd) (&rvec,"P", select,evals_cplx,helem_cplx,&N,&sigma, 
		  workev,"I",&N,which_evals,&NeV,&tolArpack,resid,&NkV, 
		  helem_cplx,&N,iparam,ipntr,workd,workl,&lworkl, 
		  rwork,&info,1,1,2);
#else
    _AFT(pzneupd) (&mpi_comm_f,&rvec,"P", select,evals_cplx, 
		   helem_cplx,&N,&sigma, 
		   workev,"I",&N,which_evals,&NeV,&tolArpack, resid,&NkV, 
		   helem_cplx,&N,iparam,ipntr,workd,workl,&lworkl, 
		   rwork,&info,1,1,2);
#endif

    if( (info)!=0){
      printfQuda("eigenSolver: Error with _neupd, info = %d \n",(info));
      printfQuda("eigenSolver: Check the documentation of _neupd. \n");
    }
    else{ //report eiegnvalues and their residuals
      t_fin = MPI_Wtime();
      printfQuda("eigenSolver: TIME_REPORT - Eigenvector calculation: %f sec\n",t_fin-t_ini);
      printfQuda("Ritz Values and their errors\n");
      printfQuda("============================\n");

      // print out the computed ritz values and their error estimates 
      nconv = iparam[4];
      for(j=0; j< nconv; j++){
	printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		   real(evals_cplx[j]),
		   imag(evals_cplx[j]),
		   std::abs(*(workl+ipntr[10]-1+j)));
	sorted_evals_index[j] = j;
	sorted_evals[j] = std::abs(evals_cplx[j]);
      }

      //SORT THE EIGENVALUES in absolute ascending order
      t1 = MPI_Wtime();
      //quicksort(nconv,sorted_evals,sorted_evals_index);
      sortAbs(sorted_evals,nconv,false,sorted_evals_index);
      //Print sorted evals
      t2 = MPI_Wtime();
      printfQuda("Sorting time: %f sec\n",t2-t1);
      printfQuda("Sorted eigenvalues based on their absolute values:\n");
      
      // print out the computed ritz values and their error estimates 
      for(j=0; j< nconv; j++){
	printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		   real(evals_cplx[sorted_evals_index[j]]),
		   imag(evals_cplx[sorted_evals_index[j]]),
		   std::abs(*(workl+ipntr[10]-1+sorted_evals_index[j])) );
      }      
    }

    // Print additional convergence information.
    if( (info)==1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info==3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
  }//- if(info < 0) else part

#ifndef MPI_COMMS
  if (NULL != arpack_logfile)
    _AFT(finilog)(&arpack_log_u);
#else
  if(comm_rank() == 0){
    if (NULL != arpack_logfile){
      _AFT(finilog)(&arpack_log_u);
    }
  }
#endif     

  //- calculate eigenvalues of the actual operator
  printfQuda("Eigenvalues of the %s Dirac operator:\n",
	     isFullOp ? "Full" : "Even-Odd");
  printfQuda("===========\n");

  t1 = MPI_Wtime();
  // !!!!!!! please check that ipntr[0] does not change 
  ColorSpinorParam cpuParam3(helem_cplx,*param,GK_localL,!isFullOp); 
  cpuColorSpinorField *h_v3 = NULL;
  for(int i =0 ; i < NeV ; i++){
    cpuParam3.v = (helem_cplx+i*LDV);
    h_v3 = new cpuColorSpinorField(cpuParam3);
    *d_v = *h_v3;                                    //d_v = v
    diracOp->MdagM(*d_v2,*d_v);                      //d_v2 = M*v
    evals_cplx[i]=blas::cDotProduct(*d_v,*d_v2);     //lambda = v^dag * M*v
    blas::axpby(1.0,*d_v2,-real(evals_cplx[i]),*d_v);//d_v=||M*v-lambda*v||

    //QKXTM: DMH careful here. It might be norm() in a different namespace...
    double norma = blas::norm2(*d_v);
    printfQuda("Eval[%04d] = %+e  %+e    Residual: %+e\n",
	       i,real(evals_cplx[i]),imag(evals_cplx[i]),sqrt(norma));
    delete h_v3;
  }
  t2 = MPI_Wtime();
  printfQuda("\neigenSolver: TIME_REPORT - Eigenvalues of Dirac operator: %f sec\n",t2-t1);

  //free memory
  free(resid);
  free(iparam);
  free(ipntr);
  free(workd);
  free(workl);
  free(rwork);
  free(sorted_evals);
  free(sorted_evals_index);
  free(select);
  free(workev);

  delete h_v;
  delete h_v2;
  delete d_v;
  delete d_v2;

  return;
}

template<typename Float>
void QKXTM_Deflation<Float>::rotateFromChiralToUKQCD(){
  if(NeV == 0) return;
  std::complex<Float> transMatrix[4][4];
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++){
      transMatrix[mu][nu].real(0.0);
      transMatrix[mu][nu].imag(0.0);
    }

  Float value = 1./sqrt(2.);

  transMatrix[0][0].real(-value); // g4*g5*U
  transMatrix[1][1].real(-value);
  transMatrix[2][2].real(value);
  transMatrix[3][3].real(value);

  transMatrix[0][2].real(value);
  transMatrix[1][3].real(value);
  transMatrix[2][0].real(value);
  transMatrix[3][1].real(value);

  std::complex<Float> tmp[4];
  std::complex<Float> *vec_cmlx = NULL;

  for(int i = 0 ; i < NeV ; i++){
    vec_cmlx = (std::complex<Float>*) &(h_elem[i*total_length_per_NeV]);
    for(int iv = 0 ; iv < (GK_localVolume)/fullorHalf ; iv++){
      for(int ic = 0 ; ic < 3 ; ic++){
	memset(tmp,0,4*2*sizeof(Float));
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++)
	    tmp[mu] = tmp[mu] + transMatrix[mu][nu] * ( *(vec_cmlx+(iv*4*3+nu*3+ic)) );
	for(int mu = 0 ; mu < 4 ; mu++)
	  *(vec_cmlx+(iv*4*3+mu*3+ic)) = tmp[mu];
      }//-ic
    }//-iv
  }//-i

  printfQuda("Rotation to UKQCD basis completed successfully\n");
}

template<typename Float>
void QKXTM_Deflation<Float>::multiply_by_phase(){
  if(NeV == 0)return;
  Float phaseRe, phaseIm;
  Float tmp0,tmp1;

  if(!isFullOp){
    for(int ivec = 0 ; ivec < NeV ; ivec++)
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit     = (x+y+z+t) & 1;
	if(oddBit){
	  continue;
	}
	else{
	  phaseRe = cos(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
	  phaseIm = sin(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
	  int pos = ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
		      z*GK_localL[1]*GK_localL[0]+
		      y*GK_localL[0]+
		      x)/2)*4*3*2 + mu*3*2 + c1*2 ;
	  tmp0 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseRe - 
		  h_elem[ivec*total_length_per_NeV + pos + 1] * phaseIm);
	  tmp1 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseIm + 
		  h_elem[ivec*total_length_per_NeV + pos + 1] * phaseRe);
	  h_elem[ivec*total_length_per_NeV + pos + 0] = tmp0;
	  h_elem[ivec*total_length_per_NeV + pos + 1] = tmp1;
	}
      }
  }
  else{
    for(int ivec = 0 ; ivec < NeV ; ivec++)
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++){
      phaseRe = cos(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
      phaseIm = sin(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
      int pos = (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
		 z*GK_localL[1]*GK_localL[0]+
		 y*GK_localL[0]+
		 x)*4*3*2 + mu*3*2 + c1*2 ;
      tmp0 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseRe - 
	      h_elem[ivec*total_length_per_NeV + pos + 1] * phaseIm);
      tmp1 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseIm + 
	      h_elem[ivec*total_length_per_NeV + pos + 1] * phaseRe);
      h_elem[ivec*total_length_per_NeV + pos + 0] = tmp0;
      h_elem[ivec*total_length_per_NeV + pos + 1] = tmp1;
    }
  }//-else

  printfQuda("Multiplication by phase completed successfully\n");
}


template<typename Float>
void QKXTM_Deflation<Float>::readEigenVectors(char *prefix_path){
  if(NeV == 0)return;
  LimeReader *limereader;
  FILE *fid;
  char *lime_type,*lime_data;
  unsigned long int lime_data_size;
  char dummy;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  unsigned int i,j;
  unsigned short int chunksize,mu,c1;
  char *buffer;
  unsigned int x,y,z,t;
  int  isDouble; // default precision
  int error_occured=0;
  int next_rec_is_prop = 0;
  char filename[257];
   
  for(int nev = 0 ; nev < NeV ; nev++){
    sprintf(filename,"%s.%05d",prefix_path,nev);
    if(comm_rank() == 0) {
      // read lime header 
      fid=fopen(filename,"r");
      if(fid==NULL) {
	fprintf(stderr,"process 0: Error in %s Could not open %s for reading\n",__func__, filename);
	error_occured=1;
      }
      if ((limereader = limeCreateReader(fid))==NULL) {
	fprintf(stderr,"process 0: Error in %s! Could not create limeReader\n", __func__);
	error_occured=1;
      }
      if(!error_occured) {
	while(limeReaderNextRecord(limereader) != LIME_EOF ) {
	  lime_type = limeReaderType(limereader);
	  if(strcmp(lime_type,"propagator-type")==0) {
	    lime_data_size = limeReaderBytes(limereader);
	    lime_data = (char * )malloc(lime_data_size);
	    limeReaderReadData((void *)lime_data,&lime_data_size,limereader);
	    
	    if (strncmp ("DiracFermion_Source_Sink_Pairs", lime_data, 
			 strlen ("DiracFermion_Source_Sink_Pairs"))!=0 &&
		strncmp ("DiracFermion_Sink", lime_data, 
			 strlen ("DiracFermion_Sink"))!=0 ) {
	      fprintf (stderr, " process 0: Error in %s! Got %s for \"propagator-type\", expecting %s or %s\n", __func__, lime_data, 
		       "DiracFermion_Source_Sink_Pairs", 
		       "DiracFermion_Sink");
	      error_occured = 1;
	      break;
	    }
	    free(lime_data);
	  }
	  //lime_type="scidac-binary-data";
	  if((strcmp(lime_type,"etmc-propagator-format")==0) || 
	     (strcmp(lime_type,"etmc-source-format")==0) || 
	     (strcmp(lime_type,"etmc-eigenvectors-format")==0) || 
	     (strcmp(lime_type,"eigenvector-info")==0)) {
	    lime_data_size = limeReaderBytes(limereader);
	    lime_data = (char * )malloc(lime_data_size);
	    limeReaderReadData((void *)lime_data,&lime_data_size,limereader);
	    sscanf(qcd_getParam("<precision>",lime_data, 
				lime_data_size),"%i",&isDouble);    
	    //		     printf("got precision: %i\n",isDouble);
	    free(lime_data);
	    
	    next_rec_is_prop = 1;
	  }
	  if(strcmp(lime_type,"scidac-binary-data")==0 && 
	     next_rec_is_prop) {	      
	    break;
	  }
	}
	// read 1 byte to set file-pointer to start of binary data 
	lime_data_size=1;
	limeReaderReadData(&dummy,&lime_data_size,limereader);
	offset = ftell(fid)-1;
	limeDestroyReader(limereader);      
	fclose(fid);
      }     
    }//end myid==0 condition
    
    MPI_Bcast(&error_occured,1,MPI_INT,0,MPI_COMM_WORLD);
    if(error_occured) errorQuda("Error with reading eigenVectors\n");
    //     if(isDouble != 32 && isDouble != 64 )isDouble = 32;     
    MPI_Bcast(&isDouble,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
    
    //     printfQuda("I have precision %d\n",isDouble);
    
    if( typeid(Float) == typeid(double) ){
      if( isDouble != 64 ) errorQuda("Your precisions does not agree");
    }
    else if(typeid(Float) == typeid(float) ){
      if( isDouble != 32 ) errorQuda("Your precisions does not agree");
    }
    else
      errorQuda("Problem with the precision\n");

    if(isDouble==64)
      isDouble=1;      
    else if(isDouble==32)
      isDouble=0; 
    else
      {
	fprintf(stderr,"process %i: Error in %s! Unsupported precision\n",
		comm_rank(), __func__);
      }  
     
    if(isDouble)
      {

	sizes[0] = GK_totalL[3];
	sizes[1] = GK_totalL[2];
	sizes[2] = GK_totalL[1];
	sizes[3] = GK_totalL[0];
	sizes[4] = (4*3*2);
	 
	lsizes[0] = GK_localL[3];
	lsizes[1] = GK_localL[2];
	lsizes[2] = GK_localL[1];
	lsizes[3] = GK_localL[0];
	lsizes[4] = sizes[4];
	 
	starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	starts[4]      = 0;


	 
	MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,
				 MPI_DOUBLE,&subblock);
	MPI_Type_commit(&subblock);
      
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, 
		      MPI_INFO_NULL, &mpifid);
	MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, 
			  "native", MPI_INFO_NULL);
	 
	//load time-slice by time-slice:
	chunksize=4*3*2*sizeof(double);
	buffer = (char*) malloc(chunksize*GK_localVolume);
	if(buffer==NULL)
	  {
	    fprintf(stderr,"process %i: Error in %s! Out of memory\n",
		    comm_rank(), __func__);
	    return;
	  }
	MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, 
			  MPI_DOUBLE, &status);
	if(!qcd_isBigEndian())      
	  qcd_swap_8((double*)buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
	i=0;
	if(!isFullOp){
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      int oddBit     = (x+y+z+t) & 1;
		      if(oddBit){
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 0 ] = ((double*)buffer)[i];
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 1 ] = ((double*)buffer)[i+1];
			i+=2;
		      }
		      else{
			i+=2;
		      }
		    }
	}
	else{
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 0 ] = ((double*)buffer)[i];
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 1 ] = ((double*)buffer)[i+1];
		      i+=2;
		    }
	}


	free(buffer);
	MPI_File_close(&mpifid);
	MPI_Type_free(&subblock);
	 
	continue;
      }//end isDouble condition
    else
      {
	sizes[0] = GK_totalL[3];
	sizes[1] = GK_totalL[2];
	sizes[2] = GK_totalL[1];
	sizes[3] = GK_totalL[0];
	sizes[4] = (4*3*2);
	 
	lsizes[0] = GK_localL[3];
	lsizes[1] = GK_localL[2];
	lsizes[2] = GK_localL[1];
	lsizes[3] = GK_localL[0];
	lsizes[4] = sizes[4];
	 
	starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	starts[4]      = 0;

	//	 for(int ii = 0 ; ii < 5 ; ii++)
	//  printf("%d %d %d %d\n",comm_rank(),sizes[ii],lsizes[ii],starts[ii]);

	MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,
				 MPI_FLOAT,&subblock);
	MPI_Type_commit(&subblock);
      
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, 
		      MPI_INFO_NULL, &mpifid);
	MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, 
			  "native", MPI_INFO_NULL);
      
	//load time-slice by time-slice:
	chunksize=4*3*2*sizeof(float);
	buffer = (char*) malloc(chunksize*GK_localVolume);
	if(buffer==NULL)
	  {
	    fprintf(stderr,"process %i: Error in %s! Out of memory\n",
		    comm_rank(), __func__);
	    return;
	  }
	MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, 
			  MPI_FLOAT, &status);

	if(!qcd_isBigEndian())
	  qcd_swap_4((float*) buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
      
	i=0;
	if(!isFullOp){
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      int oddBit     = (x+y+z+t) & 1;
		      if(oddBit){
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 0 ] = 
			  *((float*)(buffer + i)); i+=4;
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 1 ] = 
			  *((float*)(buffer + i)); i+=4;
		      }
		      else{
			i+=8;
		      }
		    }    
	}  
	else{
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 0 ] = 
			*((float*)(buffer + i)); i+=4;
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + mu*3*2 + c1*2 + 1 ] = 
			*((float*)(buffer + i)); i+=4;
		    }    
	}
      
	free(buffer);
	MPI_File_close(&mpifid);
	MPI_Type_free(&subblock);            
      
	continue;
      }//end isDouble condition
  }
  printfQuda("Eigenvectors loaded successfully\n");
}//end qcd_getVectorLime 


template<typename Float>
void QKXTM_Deflation<Float>::readEigenValues(char *filename){
  if(NeV == 0)return;
  FILE *ptr;
  Float dummy;
  ptr = fopen(filename,"r");
  if(ptr == NULL)errorQuda("Error cannot open file to read eigenvalues\n");
  char stringFormat[257];
  if(typeid(Float) == typeid(double))
    strcpy(stringFormat,"%lf");
  else if(typeid(Float) == typeid(float))
    strcpy(stringFormat,"%f");

  for(int i = 0 ; i < NeV ; i++){
    fscanf(ptr,stringFormat,&(EigenValues()[2*i]),&dummy);
    EigenValues()[2*i+1] = 0.0;
  }

  printfQuda("Eigenvalues loaded successfully\n");
  fclose(ptr);
}

//-C.K: This member function performs the operation 
// vec_defl = vec_in - (U U^dag) vec_in
template <typename Float>
void QKXTM_Deflation<Float>::
projectVector(QKXTM_Vector<Float> &vec_defl, 
	      QKXTM_Vector<Float> &vec_in, 
	      int is){
  
  if(!isFullOp) errorQuda("projectVector: This function only works with the Full Operator\n");
  
  if(NeV == 0){
    printfQuda("NeV = %d. Will not deflate source vector!!!\n",NeV);
    vec_defl.packVector((Float*) vec_in.H_elem());
    vec_defl.loadVector();

    return;
  }

  Float *ptr_elem = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec  = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  Float *out_vec        = (Float*) calloc(NeV*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV*2,sizeof(Float)) ;
  
  if(ptr_elem == NULL || 
     tmp_vec == NULL || 
     out_vec == NULL || 
     out_vec_reduce == NULL) 
    errorQuda("projectVector: Error with memory allocation\n");
  
  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  Float al[2] = {-1.0,0.0};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;
  
  //-C.K. tmp_vec = vec_in
  memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); 
  memset(out_vec,0,NeV*2*sizeof(Float));
  memset(out_vec_reduce,0,NeV*2*sizeof(Float));
  memset(ptr_elem,0,NN*2*sizeof(Float));

  if( typeid(Float) == typeid(float) ){
    
    //-C.K: out_vec_reduce = h_elem^dag * tmp_vec -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		tmp_vec, incx, (void*) beta, out_vec, incy );
    
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,MPI_SUM,
		  MPI_COMM_WORLD); 
    
    //-C.K: ptr_elem = h_elem * out_vec_reduce -> ptr_elem = U*U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV, (void*) alpha, 
		(void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );
    
    //-C.K. tmp_vec = -1.0*ptr_elem + tmp_vec -> 
    //       tmp_vec = vec_in - U*U^dag * vec_in
    cblas_caxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  else if( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		tmp_vec, incx, (void*) beta, out_vec, incy );
    
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,MPI_SUM,
		  MPI_COMM_WORLD);
    
    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );
    
    cblas_zaxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  

  //   Float udotb[2];
  //   Float udotb_reduce[2];
  //   Float *tmp_vec2 = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  //   char fbase[257];
  
  //   if(tmp_vec2 == NULL)errorQuda("projectVector: Error with memory allocation\n");

  //   memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); //-C.K. tmp_vec = vec_in
  //   memset(tmp_vec2,0,NN*2*sizeof(Float));

  //   //  dumpVector(tmp_vec,is,"MdagSource");

  //   if( typeid(Float) == typeid(float) ){
  //     for(int iv = 0;iv<NeV;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV]),bytes_total_length_per_NeV);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_cdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_caxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }
  //     cblas_caxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);
  //   }
  //   else if( typeid(Float) == typeid(double) ){
  //     for(int iv = 0;iv<NeV;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV]),bytes_total_length_per_NeV);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_zdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("*** projectVector: evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_zaxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }    
  //     cblas_zaxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);  //-C.K.: tmp_vec = tmp_vec - tmp_vec2 = vec_in  - UU^dag * vec_in
  //   }
  //   free(tmp_vec2);

  //  dumpVector(tmp_vec,is,"deflatedSource");

  vec_defl.packVector((Float*) tmp_vec);
  vec_defl.loadVector();

  free(ptr_elem);
  free(tmp_vec);
  free(out_vec);
  free(out_vec_reduce);

  printfQuda("projectVector: Deflation of the source vector completed succesfully\n");
}

//-C.K: This member function performs the operation 
//vec_defl = vec_in - (U U^dag) vec_in
template <typename Float>
void QKXTM_Deflation<Float>::
projectVector(QKXTM_Vector<Float> &vec_defl, 
	      QKXTM_Vector<Float> &vec_in, 
	      int is, int NeV_defl){
  
  if(!isFullOp) errorQuda("projectVector: This function only works with the Full Operator\n");
  
  if(NeV_defl == 0){
    printfQuda("NeV = %d. Will not deflate source vector!\n",NeV_defl);
    vec_defl.packVector((Float*) vec_in.H_elem());
    vec_defl.loadVector();
    return;
  }

  Float *ptr_elem = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec  = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  Float *out_vec        = (Float*) calloc(NeV_defl*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV_defl*2,sizeof(Float)) ;
  
  if(ptr_elem == NULL || 
     tmp_vec == NULL || 
     out_vec == NULL || 
     out_vec_reduce == NULL) 
    errorQuda("projectVector: Error with memory allocation\n");
  
  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  Float al[2] = {-1.0,0.0};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;
  
  //-C.K. tmp_vec = vec_in
  memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV);
  memset(out_vec,0,NeV_defl*2*sizeof(Float));
  memset(out_vec_reduce,0,NeV_defl*2*sizeof(Float));
  memset(ptr_elem,0,NN*2*sizeof(Float));

  if( typeid(Float) == typeid(float) ){
    //-C.K: out_vec_reduce = h_elem^dag * tmp_vec -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, 
		NN, tmp_vec, incx, (void*) beta, out_vec, incy );  
    MPI_Allreduce(out_vec,out_vec_reduce,NeV_defl*2,MPI_FLOAT,
		  MPI_SUM,MPI_COMM_WORLD);
    
    //-C.K: ptr_elem = h_elem * out_vec_reduce -> ptr_elem = U*U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, NN,
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );  
    
    //-C.K. tmp_vec = -1.0*ptr_elem + tmp_vec -> 
    //      tmp_vec = vec_in - U*U^dag * vec_in
    cblas_caxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  else if( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, 
		NN, tmp_vec, incx, (void*) beta, out_vec, incy );
    MPI_Allreduce(out_vec,out_vec_reduce,NeV_defl*2,MPI_DOUBLE,MPI_SUM,
		  MPI_COMM_WORLD);

    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, NN,
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );    
    
    cblas_zaxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }

  
  //   Float udotb[2];
  //   Float udotb_reduce[2];
  //   Float *tmp_vec2 = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  //   char fbase[257];
  
  //   if(tmp_vec2 == NULL)errorQuda("projectVector: Error with memory allocation\n");

  //   memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); //-C.K. tmp_vec = vec_in
  //   memset(tmp_vec2,0,NN*2*sizeof(Float));

  //   //  dumpVector(tmp_vec,is,"MdagSource");

  //   if( typeid(Float) == typeid(float) ){
  //     for(int iv = 0;iv<NeV_defl;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV_defl]),bytes_total_length_per_NeV_defl);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_cdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_caxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }
  //     cblas_caxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);
  //   }
  //   else if( typeid(Float) == typeid(double) ){
  //     for(int iv = 0;iv<NeV_defl;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV_defl]),bytes_total_length_per_NeV_defl);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_zdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("*** projectVector: evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_zaxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }    
  //     cblas_zaxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);  //-C.K.: tmp_vec = tmp_vec - tmp_vec2 = vec_in  - UU^dag * vec_in
  //   }
  //   free(tmp_vec2);

  //  dumpVector(tmp_vec,is,"deflatedSource");

  vec_defl.packVector((Float*) tmp_vec);
  vec_defl.loadVector();

  free(ptr_elem);
  free(tmp_vec);
  free(out_vec);
  free(out_vec_reduce);

  printfQuda("projectVector: Deflation of the source vector completed succesfully\n");
}

//------------------------------------------------------------------//
//- C.K. Functions to perform and write the exact part of the loop -//
//------------------------------------------------------------------//

template<typename Float>
void QKXTM_Deflation<Float>::
Loop_w_One_Der_FullOp_Exact(int n, QudaInvertParam *param,
			    void *gen_uloc,void *std_uloc,
			    void **gen_oneD, 
			    void **std_oneD, 
			    void **gen_csvC, 
			    void **std_csvC){
  
  if(!isFullOp) errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: This function only works with the full operator\n");
  
  void *h_ctrn, *ctrnS, *ctrnC;

  double t1,t2;

  if((cudaMallocHost(&h_ctrn, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: Error allocating memory for contraction results in CPU.\n");
  cudaMemset(h_ctrn, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);
  
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnS, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  if((cudaMalloc(&ctrnC, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnC, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  checkCudaError();

  //- Set the eigenvector into cudaColorSpinorField format and save to x
  bool pc_solve = false;
  cudaColorSpinorField *x1 = NULL;

  double *eigVec = (double*) malloc(bytes_total_length_per_NeV);
 memcpy(eigVec,&(h_elem[n*total_length_per_NeV]),bytes_total_length_per_NeV);

  QKXTM_Vector<double> *Kvec = 
    new QKXTM_Vector<double>(BOTH,VECTOR);
  
  ColorSpinorParam cpuParam((void*)eigVec,*param,GK_localL,pc_solve);
  ColorSpinorParam cudaParam(cpuParam, *param);
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  x1 = new cudaColorSpinorField(cudaParam);

  Kvec->packVector(eigVec);
  Kvec->loadVector();
  Kvec->uploadToCuda(x1,pc_solve);

  Float eVal = eigenValues[2*n+0];

  cudaColorSpinorField *tmp1 = NULL;
  cudaColorSpinorField *tmp2 = NULL;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  tmp1 = new cudaColorSpinorField(cudaParam);
  tmp2 = new cudaColorSpinorField(cudaParam);
  blas::zero(*tmp1);
  blas::zero(*tmp2);

  cudaColorSpinorField &tmp3 = *tmp1;
  cudaColorSpinorField &tmp4 = *tmp2;
  cudaColorSpinorField &x = *x1;
  //------------------------------------------------------------------------
  
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
    DiracClover *dW = new DiracClover(dWParam);
    dW->M(tmp4,x);
    delete dW;
  } 
  else if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH){
    dWParam.type = QUDA_WILSON_DIRAC;
    DiracWilson *dW = new DiracWilson(dWParam);
    dW->M(tmp4,x);
    delete dW;
  }
  else{
    errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: One end trick works only for twisted mass fermions\n");
  }
  checkCudaError();

  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Even()), 
	     static_cast<cudaColorSpinorField*>(&tmp4.Even()));
  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Odd()), 
	     static_cast<cudaColorSpinorField*>(&tmp4.Odd()));

  long int sizeBuffer;
  sizeBuffer = 
    sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];
  CovD *cov = new CovD(gaugePrecise, profileCovDev);

  int NN = 16*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];
  int incx = 1;
  int incy = 1;
  Float pceval[2] = {1.0/eVal,0.0};
  Float mceval[2] = {-1.0/eVal,0.0};

  // ULTRA-LOCAL Generalized one-end trick
  contract(x, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  if( typeid(Float) == typeid(float) ) 
    cblas_caxpy(NN,(void*)pceval,(void*)h_ctrn,incx,(void*)gen_uloc,incy);
  else if( typeid(Float) == typeid(double) ) 
    cblas_zaxpy(NN,(void*)pceval,(void*)h_ctrn,incx,(void*)gen_uloc,incy);
  //------------------------------------------------

  // ULTRA-LOCAL Standard one-end trick
  contract(x, x, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  if( typeid(Float) == typeid(float) ) 
    cblas_caxpy(NN,(void*)mceval,(void*)h_ctrn,incx,(void*)std_uloc,incy);
  else if( typeid(Float) == typeid(double) ) 
    cblas_zaxpy(NN,(void*)mceval,(void*)h_ctrn,incx,(void*)std_uloc,incy);
  //------------------------------------------------

  cudaDeviceSynchronize();

  // ONE-DERIVATIVE Generalized one-end trick
  for(int mu=0; mu<4; mu++){
    cov->M(tmp4,tmp3,mu);
    // Term 0
    contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5); 
    
    cov->M  (tmp4, x,  mu+4);
    // Term 0 + Term 3
    contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_PLUS);
    cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
    
    cov->M  (tmp4, x, mu);
    // Term 0 + Term 3 + Term 2 (C Sum)
    contract(tmp4, tmp3, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 (D Dif)
    contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);                
    
    cov->M  (tmp4, tmp3,  mu+4);
    // Term 0 + Term 3 + Term 2 + Term 1 (C Sum)
    contract(x, tmp4, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 - Term 1 (D Dif)
    contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);
    cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (void*) pceval, (void*) h_ctrn, incx, 
		  (void*) gen_oneD[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (void*) pceval, (void*) h_ctrn, incx, 
		  (void*) gen_oneD[mu], incy);
    
    cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (void*) pceval, (void*) h_ctrn, incx, 
		  (void*) gen_csvC[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (void*) pceval, (void*) h_ctrn, incx, 
		  (void*) gen_csvC[mu], incy);
  }
  
  //------------------------------------------------

  // ONE-DERIVATIVE Standard one-end trick
  for(int mu=0; mu<4; mu++){
    cov->M  (tmp4, x,  mu);
    cov->M  (tmp3, x,  mu+4);
    // Term 0
    contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5);
    // Term 0 + Term 3
    contract(tmp3, x, ctrnS, QUDA_CONTRACT_GAMMA5_PLUS);
    cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
    
    // Term 0 + Term 3 + Term 2 (C Sum)
    contract(tmp4, x, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 (D Dif)
    contract(tmp4, x, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);
    // Term 0 + Term 3 + Term 2 + Term 1 (C Sum)
    contract(x, tmp3, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 - Term 1 (D Dif)
    contract(x, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);
    cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (void*) mceval, (void*) h_ctrn, incx, 
		  (void*) std_oneD[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (void*) mceval, (void*) h_ctrn, incx, 
		  (void*) std_oneD[mu], incy);
    
    cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
    
    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (void*) mceval, (void*) h_ctrn, incx, 
		  (void*) std_csvC[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (void*) mceval, (void*) h_ctrn, incx, 
		  (void*) std_csvC[mu], incy);
  }

  //------------------------------------------------

  delete Kvec;
  delete x1;
  delete tmp1;
  delete tmp2;
  free(eigVec);

  delete cov;
  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  cudaFree(ctrnC);
  checkCudaError();
}
#endif
*/

//QKXTM: DMH 
/* Quarantined Code 
template<typename Float>
void oneEndTrick(ColorSpinorField &x, cudaColorSpinorField &tmp3, 
		 cudaColorSpinorField &tmp4, QudaInvertParam *param, 
		 void *cnRes_gv,void *cnRes_vv){
  void *h_ctrn, *ctrnS;
  
  if((cudaMallocHost(&h_ctrn, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in CPU.\n");
  cudaMemset(h_ctrn, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);
  if((cudaMalloc(&ctrnS, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnS, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);
  checkCudaError();
  
  DiracParam dWParam;
  dWParam.matpcType        = QUDA_MATPC_EVEN_EVEN;
  dWParam.dagger           = QUDA_DAG_NO;
  dWParam.gauge            = gaugePrecise;
  dWParam.kappa            = param->kappa;
  dWParam.mass             = 1./(2.*param->kappa) - 4.;
  dWParam.m5               = 0.;
  dWParam.mu               = 0.;
  for     (int i=0; i<4; i++)
    dWParam.commDim[i]       = 1;

  if(param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    dWParam.type           = QUDA_CLOVER_DIRAC;
    dWParam.clover                 = cloverPrecise;
    DiracClover   *dW      = new DiracClover(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  } 
  else if (param->dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    dWParam.type           = QUDA_WILSON_DIRAC;
    DiracWilson   *dW      = new DiracWilson(dWParam);
    dW->M(tmp4,x);
    delete  dW;
  }
  else{
    errorQuda("Error one end trick works only for twisted mass fermions\n");
  }
  checkCudaError();

  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Even()), static_cast<cudaColorSpinorField*>(&tmp4.Even()));
  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Odd()), static_cast<cudaColorSpinorField*>(&tmp4.Odd()));

  long int sizeBuffer;
  sizeBuffer = sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  contract(x, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_gv)[ix] += ((Float*)h_ctrn)[ix]; // generalized one end trick

  contract(x, x, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
    ((Float*) cnRes_vv)[ix] -= ((Float*)h_ctrn)[ix]; // standard one end trick
  cudaDeviceSynchronize();

  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  checkCudaError();
}
*/

