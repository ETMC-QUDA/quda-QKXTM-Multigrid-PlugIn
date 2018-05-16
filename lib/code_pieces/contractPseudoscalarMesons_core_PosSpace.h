
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int locV = blockDim.x * gridDim.x;
int c_stride_spatial = c_stride/c_localL[3];

register FLOAT2 accum;

accum.x = 0.; accum.y = 0.;

#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) ) 

if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
  
  for(int is = 0 ; is < 16 ; is++){
    short int beta = c_mesons_indices[0][is][0];
    short int gamma = c_mesons_indices[0][is][1];
    short int delta = c_mesons_indices[0][is][2];
    short int alpha = c_mesons_indices[0][is][3];
    float value = c_mesons_values[0][is];
    for(int a = 0 ; a < 3 ; a++){
      for(int b = 0 ; b < 3 ; b++){
	accum = accum + value *PROP(prop1Tex,alpha,beta,a,b) * conj(PROP(prop2Tex,delta,gamma,a,b));
      }}
  }
  __syncthreads();

  block[sid] = accum;

  __syncthreads();

 }//-if sid
