int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ Float2 shared_cache[2*16*THREADS_PER_BLOCK];
int c_stride_spatial = c_stride/c_localL[3];
int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;

int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0];
y = y_id + c_procPosition[1] * c_localL[1];
z = z_id + c_procPosition[2] * c_localL[2];

Float phase;
Float2 expon;

int ij;


for(int imom=0; imom < c_Nmoms; imom++){
  phase = ( ((Float) c_moms[imom][0]*x)/c_totalL[0] + ((Float) c_moms[imom][1]*y)/c_totalL[1] + ((Float) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
  expon.x = cos(phase);
  expon.y = -sin(phase);
  for(int gm = 0 ; gm < 16 ; gm++){
    int pos = sid + it*c_stride_spatial + gm*c_stride;
    int idx=gm*THREADS_PER_BLOCK+cacheIndex;
    shared_cache[idx] = inV[pos]*expon;
  }
  __syncthreads();
  ij = blockDim.x/2;
  while (ij != 0){
    if(cacheIndex < ij){
      for(int gm = 0 ; gm < 16 ; gm++){
	int idx=gm*THREADS_PER_BLOCK+cacheIndex;
	shared_cache[idx].x += shared_cache[idx+ij].x; 
	shared_cache[idx].y += shared_cache[idx+ij].y; 
      }
    }
    __syncthreads();
    ij /= 2;
  }
  
  if(cacheIndex == 0){
    for(int gm = 0 ; gm < 16 ; gm++){
      block[gm*c_Nmoms*gridDim.x + imom*gridDim.x + blockIdx.x] = shared_cache[gm*THREADS_PER_BLOCK+cacheIndex];
    }
  }
 } // close momentum
