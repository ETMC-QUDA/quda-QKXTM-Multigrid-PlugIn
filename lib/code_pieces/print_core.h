
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

cuPrintf("Propagator values:\n");

Float2 P[4][4];

for(int c1 = 0 ; c1 < c_nColor ; c1++)
  for(int c2 = 0 ; c2 < c_nColor ; c2++){

    for(int mu = 0 ; mu < c_nSpin ; mu++)
      for(int nu = 0 ; nu < c_nSpin; nu++)
	P[mu][nu] = out[mu*c_nSpin*c_nColor*c_nColor*c_stride + nu*c_nColor*c_nColor*c_stride + c1*c_nColor*c_stride + c2*c_stride + sid];
    
    cuPrintf("%d \t %d \t %f %f %f %f\n",c1,c2,P[0][0],P[0][1],P[0][2],P[0][3]);
    cuPrintf("%d \t %d \t %f %f %f %f\n",c1,c2,P[1][0],P[1][1],P[1][2],P[1][3]);
    cuPrintf("%d \t %d \t %f %f %f %f\n",c1,c2,P[2][0],P[2][1],P[2][2],P[2][3]);
    cuPrintf("%d \t %d \t %f %f %f %f\n",c1,c2,P[3][0],P[3][1],P[3][2],P[3][3]);

  }

