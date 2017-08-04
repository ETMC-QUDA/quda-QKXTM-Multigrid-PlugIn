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
			    void **std_csvC,GaugeCovDev *cov){
  
  if(!isFullOp) errorQuda("oneEndTrick_w_One_Der_FullOp_Exact: This function only works with the full operator\n");
  if(cov==NULL) errorQuda("CovDev has not been constructed");
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

  int NN = 16*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];
  int incx = 1;
  int incy = 1;
  Float pceval[2] = {1.0/eVal,0.0};
  Float mceval[2] = {-1.0/eVal,0.0};

  // ULTRA-LOCAL Generalized one-end trick
  contract(x, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  if( typeid(Float) == typeid(float) ) 
    cblas_caxpy(NN,(float*)pceval,(float*)h_ctrn,incx,(float*)gen_uloc,incy);
  else if( typeid(Float) == typeid(double) ) 
    cblas_zaxpy(NN,(double*)pceval,(double*)h_ctrn,incx,(double*)gen_uloc,incy);
  //------------------------------------------------

  // ULTRA-LOCAL Standard one-end trick
  contract(x, x, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

  if( typeid(Float) == typeid(float) ) 
    cblas_caxpy(NN,(float*)mceval,(float*)h_ctrn,incx,(float*)std_uloc,incy);
  else if( typeid(Float) == typeid(double) ) 
    cblas_zaxpy(NN,(double*)mceval,(double*)h_ctrn,incx,(double*)std_uloc,incy);
  //------------------------------------------------

  cudaDeviceSynchronize();

  //Create Covarant derivative.
  //  GaugeCovDev *cov = new GaugeCovDev(dWParam);

  // ONE-DERIVATIVE Generalized one-end trick
  for(int mu=0; mu<4; mu++){
    cov->MCD(tmp4,tmp3,mu);

    // Term 0
    contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5); 
    
    cov->MCD  (tmp4, x,  mu+4);
    // Term 0 + Term 3
    contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_PLUS);
    cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);
    
    cov->MCD  (tmp4, x, mu);
    // Term 0 + Term 3 + Term 2 (C Sum)
    contract(tmp4, tmp3, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 (D Dif)
    contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);                
    
    cov->MCD  (tmp4, tmp3,  mu+4);
    // Term 0 + Term 3 + Term 2 + Term 1 (C Sum)
    contract(x, tmp4, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
    // Term 0 + Term 3 - Term 2 - Term 1 (D Dif)
    contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);
    cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);

    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (float*) pceval, (float*) h_ctrn, incx, (float*) gen_oneD[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (double*) pceval, (double*) h_ctrn, incx, (double*) gen_oneD[mu], incy);
    
    cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);

    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (float*) pceval, (float*) h_ctrn, incx, (float*) gen_csvC[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (double*) pceval, (double*) h_ctrn, incx, (double*) gen_csvC[mu], incy);
  }

  //------------------------------------------------

  // ONE-DERIVATIVE Standard one-end trick
  for(int mu=0; mu<4; mu++){
    cov->MCD  (tmp4, x,  mu);
    cov->MCD  (tmp3, x,  mu+4);
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
      cblas_caxpy(NN, (float*) mceval, (float*) h_ctrn, incx, (float*) std_oneD[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (double*) mceval, (double*) h_ctrn, incx, (double*) std_oneD[mu], incy);
    
    cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
    
    if( typeid(Float) == typeid(float) ) 
      cblas_caxpy(NN, (float*) mceval, (float*) h_ctrn, incx, (float*) std_csvC[mu], incy);
    else if( typeid(Float) == typeid(double) ) 
      cblas_zaxpy(NN, (double*) mceval, (double*) h_ctrn, incx, (double*) std_csvC[mu], incy);
  }

  //------------------------------------------------
  //  delete cov;
  

  delete Kvec;
  delete x1;
  delete tmp1;
  delete tmp2;
  free(eigVec);


  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  cudaFree(ctrnC);
  checkCudaError();
}


template<typename Float>
void oneEndTrick_w_One_Der(ColorSpinorField &x, ColorSpinorField &tmp3, 
			   ColorSpinorField &tmp4, QudaInvertParam *param,
			   void *cnRes_gv,void *cnRes_vv, void **cnD_gv, 
			   void **cnD_vv, void **cnC_gv, void **cnC_vv, GaugeCovDev *cov){
  
  void *h_ctrn, *ctrnS, *ctrnC;
  if(cov==NULL) errorQuda("CovDev has not been constructed");
  if((cudaMallocHost(&h_ctrn, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in CPU.\n");
  cudaMemset(h_ctrn, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  if((cudaMalloc(&ctrnS, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnS, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

  if((cudaMalloc(&ctrnC, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3])) == cudaErrorMemoryAllocation)
    errorQuda("Error allocating memory for contraction results in GPU.\n");
  cudaMemset(ctrnC, 0, sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]);

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

  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Even()), 
	     static_cast<cudaColorSpinorField*>(&tmp4.Even()));
  gamma5Cuda(static_cast<cudaColorSpinorField*>(&tmp3.Odd()), 
	     static_cast<cudaColorSpinorField*>(&tmp4.Odd()));
  
  long int sizeBuffer;
  sizeBuffer = 
    sizeof(Float)*32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  int NN = 16*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];
  int incx = 1;
  int incy = 1;
  Float pceval[2] = {1.0,0.0};
  Float mceval[2] = {-1.0,0.0};

  ///////////////// LOCAL ///////////////////////////
  contract(x, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
  
  if( typeid(Float) == typeid(float) ) 
    cblas_caxpy(NN,(float*)pceval,(float*)h_ctrn,incx,(float*)cnRes_gv,incy);
  else if( typeid(Float) == typeid(double) ) 
    cblas_zaxpy(NN,(double*)pceval,(double*)h_ctrn,incx,(double*)cnRes_gv,incy);
  
  //    for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
  //      ((Float*) cnRes_gv)[ix] += ((Float*)h_ctrn)[ix]; // generalized one end trick
  
  contract(x, x, ctrnS, QUDA_CONTRACT_GAMMA5);
  cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
  
  //    for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
  //      ((Float*) cnRes_vv)[ix] -= ((Float*)h_ctrn)[ix]; // standard one end trick
  
  if( typeid(Float) == typeid(float) ) {
    cblas_caxpy(NN, (float*) mceval, (float*) h_ctrn, incx, (float*) cnRes_vv, incy);
  }
  else if( typeid(Float) == typeid(double) ) {
    cblas_zaxpy(NN, (double*) mceval, (double*) h_ctrn, incx, (double*) cnRes_vv, incy);
  }  
  cudaDeviceSynchronize();

  ////////////////// DERIVATIVES //////////////////////////////

  //Create Covarant derivative.
  //  GaugeCovDev *cov = new GaugeCovDev(dWParam);

  // for generalized one-end trick
  for(int mu=0; mu<4; mu++)	
    {
      cov->MCD(static_cast<cudaColorSpinorField&>(tmp4),static_cast<cudaColorSpinorField&>(tmp3),mu);

      // Term 0
      contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5);

      cov->MCD(static_cast<cudaColorSpinorField&>(tmp4),static_cast<cudaColorSpinorField&>(x),mu+4);
      // Term 0 + Term 3
      contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_PLUS);
      cudaMemcpy(ctrnC, ctrnS, sizeBuffer, cudaMemcpyDeviceToDevice);

      // Term 0 + Term 3 + Term 2 (C Sum)
      cov->MCD(static_cast<cudaColorSpinorField&>(tmp4),static_cast<cudaColorSpinorField&>(x),mu);
      contract(tmp4, tmp3, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
      // Term 0 + Term 3 - Term 2 (D Dif)
      contract(tmp4, tmp3, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);

      cov->MCD(static_cast<cudaColorSpinorField&>(tmp4),static_cast<cudaColorSpinorField&>(tmp3),mu+4);
      // Term 0 + Term 3 + Term 2 + Term 1 (C Sum)
      contract(x, tmp4, ctrnC, QUDA_CONTRACT_GAMMA5_PLUS);
      // Term 0 + Term 3 - Term 2 - Term 1 (D Dif)
      contract(x, tmp4, ctrnS, QUDA_CONTRACT_GAMMA5_MINUS);
      cudaMemcpy(h_ctrn, ctrnS, sizeBuffer, cudaMemcpyDeviceToHost);
      
      if( typeid(Float) == typeid(float) ) {
	cblas_caxpy(NN, (float*) pceval, (float*) h_ctrn, incx,  (float*) cnD_gv[mu], incy);
      }
      else if( typeid(Float) == typeid(double) ) {
	cblas_zaxpy(NN, (double*) pceval, (double*) h_ctrn, incx, (double*) cnD_gv[mu], incy);
      }

      //      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnD_gv[mu])[ix] += ((Float*)h_ctrn)[ix];
      
      cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
      
      if( typeid(Float) == typeid(float) ) {
	cblas_caxpy(NN, (float*) pceval, (float*) h_ctrn, incx, (float*) cnC_gv[mu], incy);
      }
      else if( typeid(Float) == typeid(double) ) {
	cblas_zaxpy(NN, (double*) pceval, (double*) h_ctrn, incx, (double*) cnC_gv[mu], incy);
      }
      //      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnC_gv[mu])[ix] += ((Float*)h_ctrn)[ix];
    }
  
  for(int mu=0; mu<4; mu++) // for standard one-end trick
    {
      cov->MCD(static_cast<cudaColorSpinorField&>(tmp4),static_cast<cudaColorSpinorField&>(x),mu);
      cov->MCD(static_cast<cudaColorSpinorField&>(tmp3),static_cast<cudaColorSpinorField&>(x),mu+4);
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
      if( typeid(Float) == typeid(float) ) {
	cblas_caxpy(NN, (float*) mceval, (float*) h_ctrn, incx, (float*) cnD_vv[mu], incy);
      }
      else if( typeid(Float) == typeid(double) ) { 
	cblas_zaxpy(NN, (double*) mceval, (double*) h_ctrn, incx, (double*) cnD_vv[mu], incy);
      }
      //      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnD_vv[mu])[ix]  -= ((Float*)h_ctrn)[ix];
      
      cudaMemcpy(h_ctrn, ctrnC, sizeBuffer, cudaMemcpyDeviceToHost);
      
      if( typeid(Float) == typeid(float) ) {
	cblas_caxpy(NN, (float*) mceval, (float*) h_ctrn, incx, (float*) cnC_vv[mu], incy);
      }
      else if( typeid(Float) == typeid(double) ) {
	cblas_zaxpy(NN, (double*) mceval, (double*) h_ctrn, incx, (double*) cnC_vv[mu], incy);
      }
      
      //      for(int ix=0; ix < 32*GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3]; ix++)
      //	((Float *) cnC_vv[mu])[ix] -= ((Float*)h_ctrn)[ix];
    }

  //  delete cov;

  cudaFreeHost(h_ctrn);
  cudaFree(ctrnS);
  cudaFree(ctrnC);
  checkCudaError();
}


//-C.K. This is a new function to print all the loops in ASCII format
template<typename Float>
void writeLoops_ASCII(Float *writeBuf, const char *Pref, 
		      qudaQKXTM_loopInfo loopInfo,  int type, 
		      int mu, bool exact_loop){

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    FILE *ptr;
    char file_name[512];
    char *lpart,*ptrVal;
    int Nprint,Ndump;
    int Nmoms = loopInfo.Nmoms;

    if(exact_loop) Nprint = 1;
    else{
	Nprint = loopInfo.Nprint;
	Ndump  = loopInfo.Ndump;
    }

    for(int iPrint=0;iPrint<Nprint;iPrint++){
      if(exact_loop) asprintf(&ptrVal,"%d_%d", GK_nProc[3], GK_timeRank);
      else asprintf(&ptrVal,"%04d.%d_%d",(iPrint+1)*Ndump, GK_nProc[3], GK_timeRank);

      sprintf(file_name, "%s_%s.loop.%s",Pref,loopInfo.loop_type[type],ptrVal);

      if(loopInfo.loop_oneD[type] && mu!=0) ptr = fopen(file_name,"a");
      else ptr = fopen(file_name,"w");
      if(ptr == NULL) errorQuda("Cannot open %s to write the loop\n",file_name);

      if(loopInfo.loop_oneD[type]){
	for(int ip=0; ip < Nmoms; ip++){
	  for(int lt=0; lt < GK_localL[3]; lt++){
	    int t  = lt+comm_coords(default_topo)[3]*GK_localL[3];
	    for(int gm=0; gm<16; gm++){
	      fprintf(ptr, "%02d %02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, mu, GK_moms[ip][0], GK_moms[ip][1], GK_moms[ip][2],
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
	      fprintf(ptr, "%02d %02d %+d %+d %+d %+16.15e %+16.15e\n",t, gm, GK_moms[ip][0], GK_moms[ip][1], GK_moms[ip][2],
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


//-C.K: Copy the HDF5 dataset chunk into writeBuf (for writing in Standard-Momenta Form)
template<typename Float>
void getLoopWriteBuf_StrdMomForm(Float *writeBuf, Float *loopBuf, int iPrint, int Nmoms, int imom, bool oneD){

  double fct;
  if(oneD) fct = 0.25;
  else     fct = 1.00;

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    for(int lt=0;lt<GK_localL[3];lt++){
      for(int gm=0;gm<16;gm++){
        writeBuf[0+2*gm+2*16*lt] = fct*loopBuf[0+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
        writeBuf[1+2*gm+2*16*lt] = fct*loopBuf[1+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
      }
    }
  }//-if GK_timeRank

}

//-C.K: Copy the HDF5 dataset chunk into writeBuf (for writing in High-Momenta Form)
template<typename Float>
void getLoopWriteBuf_HighMomForm(Float *writeBuf, Float *loopBuf, int iPrint, int Nmoms, bool oneD){

  double fct;
  if(oneD) fct = 0.25;
  else     fct = 1.00;

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    for(int lt=0;lt<GK_localL[3];lt++){
      for(int imom=0;imom<Nmoms;imom++){
        for(int gm=0;gm<16;gm++){
          writeBuf[0 + 2*gm + 2*16*imom + 2*16*Nmoms*lt] = fct*loopBuf[0+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
          writeBuf[1 + 2*gm + 2*16*imom + 2*16*Nmoms*lt] = fct*loopBuf[1+2*imom+2*Nmoms*lt+2*Nmoms*GK_localL[3]*gm+2*Nmoms*GK_localL[3]*16*iPrint];
        }
      }
    }
  }//-if GK_timeRank

}


//-C.K: Funtion to write the loops in HDF5 format (Standard Momenta Form)
template<typename Float>
void writeLoops_HDF5_StrdMomForm(Float *buf_std_uloc, Float *buf_gen_uloc, 
				 Float **buf_std_oneD, Float **buf_std_csvC, 
				 Float **buf_gen_oneD, Float **buf_gen_csvC, 
				 char *file_pref, 
				 qudaQKXTM_loopInfo loopInfo,
				 bool exact_loop){

  if(loopInfo.HighMomForm) errorQuda("writeLoops_HDF5_StrdMomForm: This function works only with Standard Momenta Form! (Got HighMomForm=true)\n");


  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    char fname[512];
    int Nprint,Ndump;

    if(exact_loop){
      Nprint = 1;
      sprintf(fname,"%s_Qsq%d.h5",file_pref,loopInfo.Qsq);
    }
    else{
	Nprint = loopInfo.Nprint;
	Ndump  = loopInfo.Ndump;
	sprintf(fname,"%s_Ns%04d_step%04d_Qsq%d.h5",file_pref,loopInfo.Nstoch,Ndump,loopInfo.Qsq);
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
	asprintf(&group2_tag,"Nstoch_%04d",(iPrint+1)*Ndump);
	group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }

      for(int it=0;it<6;it++){
	char *group3_tag;
	asprintf(&group3_tag,"%s",loopInfo.loop_type[it]);

	if(exact_loop) group3_id = H5Gcreate(group1_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	else           group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	for(int imom=0;imom<loopInfo.Nmoms;imom++){
	  char *group4_tag;
	  asprintf(&group4_tag,"mom_xyz_%+d_%+d_%+d",GK_moms[imom][0],GK_moms[imom][1],GK_moms[imom][2]);
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

	      getLoopWriteBuf_StrdMomForm(writeBuf,loopBuf,iPrint,loopInfo.Nmoms,imom, loopInfo.loop_oneD[it]);

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

	    getLoopWriteBuf_StrdMomForm(writeBuf,loopBuf,iPrint,loopInfo.Nmoms,imom, loopInfo.loop_oneD[it]);

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

//-C.K: Funtion to write the loops in HDF5 format (High Momenta Form)
template<typename Float>
void writeLoops_HDF5_HighMomForm(Float *buf_std_uloc, Float *buf_gen_uloc,
                                 Float **buf_std_oneD, Float **buf_std_csvC,
                                 Float **buf_gen_oneD, Float **buf_gen_csvC,
                                 char *file_pref,
                                 qudaQKXTM_loopInfo loopInfo,
                                 bool exact_loop){

  if(!loopInfo.HighMomForm) errorQuda("writeLoops_HDF5_HighMomForm: This function works only with High Momenta Form! (Got HighMomForm=false)\n");

  if(GK_timeRank >= 0 && GK_timeRank < GK_nProc[3] ){
    char fname[512];
    int Nprint,Ndump;
    int Nmoms = loopInfo.Nmoms;

    if(exact_loop){
      Nprint = 1;
      sprintf(fname,"%s_Qsq%d.h5",file_pref,loopInfo.Qsq);
    }
    else{
      Nprint = loopInfo.Nprint;
      Ndump  = loopInfo.Ndump;
      sprintf(fname,"%s_Ns%04d_step%04d_Qsq%d.h5",file_pref,loopInfo.Nstoch,Ndump,loopInfo.Qsq);
    }

  double *loopBuf = NULL;
  double *writeBuf = (double*) malloc(GK_localL[3]*Nmoms*16*2*sizeof(double));

  hsize_t start[4]  = {GK_timeRank*GK_localL[3], 0, 0, 0};

  // Dimensions of the dataspace
  hsize_t dims[4]  = {GK_totalL[3], Nmoms, 16, 2}; // Global
  hsize_t ldims[4] = {GK_localL[3], Nmoms, 16, 2}; // Local

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

  for(int iPrint=0;iPrint<Nprint;iPrint++){

    if(!exact_loop){
      char *group2_tag;
      asprintf(&group2_tag,"Nstoch_%04d",(iPrint+1)*Ndump);
      group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    }

    for(int it=0;it<6;it++){
      char *group3_tag;
      asprintf(&group3_tag,"%s",loopInfo.loop_type[it]);

      if(exact_loop) group3_id = H5Gcreate(group1_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      else           group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      if(loopInfo.loop_oneD[it]){
	for(int mu=0;mu<4;mu++){
	  if(strcmp(loopInfo.loop_type[it],"Loops")==0)   loopBuf = buf_std_oneD[mu];
	  if(strcmp(loopInfo.loop_type[it],"LoopsCv")==0) loopBuf = buf_std_csvC[mu];
	  if(strcmp(loopInfo.loop_type[it],"LpsDw")==0)   loopBuf = buf_gen_oneD[mu];
	  if(strcmp(loopInfo.loop_type[it],"LpsDwCv")==0) loopBuf = buf_gen_csvC[mu];

	  char *group4_tag;
	  asprintf(&group4_tag,"dir_%02d",mu);
	  group4_id = H5Gcreate(group3_id, group4_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	  hid_t filespace  = H5Screate_simple(4, dims, NULL);
	  hid_t dataset_id = H5Dcreate(group4_id, "loop", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	  hid_t subspace   = H5Screate_simple(4, ldims, NULL);
	  filespace = H5Dget_space(dataset_id);
	  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
	  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	  getLoopWriteBuf_HighMomForm(writeBuf, loopBuf, iPrint, loopInfo.Nmoms, loopInfo.loop_oneD[it]);

	  herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, subspace, filespace, plist_id, writeBuf);

	  H5Sclose(subspace);
	  H5Dclose(dataset_id);
	  H5Sclose(filespace);
	  H5Pclose(plist_id);

	  H5Gclose(group4_id);
	}//-mu
      }//-if
      else{
	if(strcmp(loopInfo.loop_type[it],"Scalar")==0) loopBuf = buf_std_uloc;
	if(strcmp(loopInfo.loop_type[it],"dOp")==0)    loopBuf = buf_gen_uloc;

	hid_t filespace  = H5Screate_simple(4, dims, NULL);
	hid_t dataset_id = H5Dcreate(group3_id, "loop", H5T_NATIVE_DOUBLE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	hid_t subspace   = H5Screate_simple(4, ldims, NULL);
	filespace = H5Dget_space(dataset_id);
	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
	hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	getLoopWriteBuf_HighMomForm(writeBuf, loopBuf, iPrint, loopInfo.Nmoms, loopInfo.loop_oneD[it]);

	herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, subspace, filespace, plist_id, writeBuf);

	H5Sclose(subspace);
	H5Dclose(dataset_id);
	H5Sclose(filespace);
	H5Pclose(plist_id);
      }
      H5Gclose(group3_id);
    }//-it

    if(!exact_loop) H5Gclose(group2_id);

  }//-iPrint
  H5Gclose(group1_id);
  H5Fclose(file_id);

  free(writeBuf);

  //-C.K. Write the momenta in a separate dataset (including some attributes)
  //- Only one task needs to do this
  if(GK_timeRank==0){
    hid_t file_idt = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);

    char *cNmoms, *cQsq,*loop_info,*ens_info;
    char *loop_str1,*loop_str2,*loop_str3,*loop_str4,*loop_types[6];
    asprintf(&loop_types[0],"Scalar  - Ultra-local    operators, Standard    one-end trick\n");
    asprintf(&loop_types[1],"dOp     - Ultra-local    operators, Generalized one-end trick\n");
    asprintf(&loop_types[2],"Loops   - One-derivative operators, Standard    one-end trick\n");
    asprintf(&loop_types[3],"LpsDw   - One-derivative operators, Generalized one-end trick\n");
    asprintf(&loop_types[4],"LoopsCv - Concerved current,        Standard    one-end trick\n");
    asprintf(&loop_types[5],"LpsDwCv - Concerved current,        Generalized one-end trick\n");
    asprintf(&loop_str1,"Momentum-space quark loop\nQuark field basis: Twisted\n");
    asprintf(&loop_str2,"Precision: %s\n",(typeid(Float) == typeid(float)) ? "single" : "double");
    asprintf(&loop_str3,"Inversion tolerance: %e\n", loopInfo.inv_tol);
    asprintf(&loop_str4,"Loop types:\n  %s  %s  %s  %s  %s  %s\n",loop_types[0],loop_types[1],loop_types[2],loop_types[3],loop_types[4],loop_types[5]);


    asprintf(&cNmoms,"%d\0",Nmoms);
    asprintf(&cQsq,"%d\0",loopInfo.Qsq);
    asprintf(&loop_info,"%s%s%s%s\0",loop_str1,loop_str2,loop_str3,loop_str4);
    asprintf(&ens_info,"kappa = %10.8f\nmu = %8.6f\nCsw = %8.6f\0",loopInfo.kappa, loopInfo.mu, loopInfo.csw);
    hid_t attrdat_id1 = H5Screate(H5S_SCALAR);
    hid_t attrdat_id2 = H5Screate(H5S_SCALAR);
    hid_t attrdat_id3 = H5Screate(H5S_SCALAR);
    hid_t attrdat_id4 = H5Screate(H5S_SCALAR);
    hid_t type_id1 = H5Tcopy(H5T_C_S1);
    hid_t type_id2 = H5Tcopy(H5T_C_S1);
    hid_t type_id3 = H5Tcopy(H5T_C_S1);
    hid_t type_id4 = H5Tcopy(H5T_C_S1);
    H5Tset_size(type_id1, strlen(cNmoms));
    H5Tset_size(type_id2, strlen(cQsq));
    H5Tset_size(type_id3, strlen(loop_info));
    H5Tset_size(type_id4, strlen(ens_info));
    hid_t attr_id1 = H5Acreate2(file_idt, "Nmoms", type_id1, attrdat_id1, H5P_DEFAULT, H5P_DEFAULT);
    hid_t attr_id2 = H5Acreate2(file_idt, "Qsq"  , type_id2, attrdat_id2, H5P_DEFAULT, H5P_DEFAULT);
    hid_t attr_id3 = H5Acreate2(file_idt, "Correlator-info", type_id3, attrdat_id3, H5P_DEFAULT, H5P_DEFAULT);
    hid_t attr_id4 = H5Acreate2(file_idt, "Ensemble-info", type_id4, attrdat_id4, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr_id1, type_id1, cNmoms);
    H5Awrite(attr_id2, type_id2, cQsq);
    H5Awrite(attr_id3, type_id3, loop_info);
    H5Awrite(attr_id4, type_id4, ens_info);
    H5Aclose(attr_id1);
    H5Aclose(attr_id2);
    H5Aclose(attr_id3);
    H5Aclose(attr_id4);
    H5Tclose(type_id1);
    H5Tclose(type_id2);
    H5Tclose(type_id3);
    H5Tclose(type_id4);
    H5Sclose(attrdat_id1);
    H5Sclose(attrdat_id2);
    H5Sclose(attrdat_id3);
    H5Sclose(attrdat_id4);

    hid_t MOMTYPE_H5 = H5T_NATIVE_INT;
    char *dset_tag;
    asprintf(&dset_tag,"Momenta_list_xyz");

    hsize_t Mdims[2] = {(hsize_t)Nmoms,3};
    hid_t filespace  = H5Screate_simple(2, Mdims, NULL);
    hid_t dataset_id = H5Dcreate(file_idt, dset_tag, MOMTYPE_H5, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int *Moms_H5 = (int*) malloc(Nmoms*3*sizeof(int));
    for(int im=0;im<Nmoms;im++){
      for(int d=0;d<3;d++) Moms_H5[d + 3*im] = GK_moms[im][d];
    }

    herr_t status = H5Dwrite(dataset_id, MOMTYPE_H5, H5S_ALL, filespace, H5P_DEFAULT, Moms_H5);

    H5Dclose(dataset_id);
    H5Sclose(filespace);
    H5Fclose(file_idt);

    free(Moms_H5);
  }//- if GK_timeRank==0

  }//-if GK_timeRank >= 0 && GK_timeRank < GK_nProc[3]
}



template<typename Float>
void writeLoops_HDF5(Float *buf_std_uloc, Float *buf_gen_uloc,
                     Float **buf_std_oneD, Float **buf_std_csvC,
                     Float **buf_gen_oneD, Float **buf_gen_csvC,
                     char *file_pref,
                     qudaQKXTM_loopInfo loopInfo,
                     bool exact_loop){

  if(loopInfo.HighMomForm)
    writeLoops_HDF5_HighMomForm<double>(buf_std_uloc, buf_gen_uloc,
                                        buf_std_oneD, buf_std_csvC,
                                        buf_gen_oneD, buf_gen_csvC,
                                        file_pref, loopInfo,
                                        exact_loop);
  else
    writeLoops_HDF5_StrdMomForm<double>(buf_std_uloc, buf_gen_uloc,
                                        buf_std_oneD, buf_std_csvC,
                                        buf_gen_oneD, buf_gen_csvC,
                                        file_pref, loopInfo,
                                        exact_loop);

}
