
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ FLOAT2 shared_cache[THREADS_PER_BLOCK];

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
    for(int a = 0 ; a < 3 ; a++)
      for(int b = 0 ; b < 3 ; b++){
	accum = accum + value * PROP(prop1Tex,alpha,beta,a,b) * conj(PROP(prop2Tex,delta,gamma,a,b));
      }
  }
 }
//////////////////////////////////////////////////////////////////////////////////////

__syncthreads();

int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;

int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0] - x0;
y = y_id + c_procPosition[1] * c_localL[1] - y0;
z = z_id + c_procPosition[2] * c_localL[2] - z0;

FLOAT phase;
FLOAT2 expon;
int i;
for(int imom = 0 ; imom < c_Nmoms ; imom++){
  phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
  expon.x = cos(phase);
  expon.y = -sin(phase);
  shared_cache[0*THREADS_PER_BLOCK + cacheIndex] = accum * expon; 
  __syncthreads();
  i = blockDim.x/2;
  while (i != 0){
    if(cacheIndex < i){
      shared_cache[cacheIndex].x += shared_cache[cacheIndex + i].x;
      shared_cache[cacheIndex].y += shared_cache[cacheIndex + i].y;
    }
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0){
    block[imom*2*gridDim.x + 0*gridDim.x + blockIdx.x] = shared_cache[0*THREADS_PER_BLOCK + 0];
  }

 } // close momentum
