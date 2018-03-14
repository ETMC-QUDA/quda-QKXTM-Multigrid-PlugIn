int space_stride = c_localL[0]*c_localL[1]*c_localL[2];

int sid = blockIdx.x*blockDim.x + threadIdx.x;
if(sid >= space_stride) return;


for(int mu = 0 ; mu < 4 ; mu++)
  for(int c1 = 0 ; c1 < 3 ; c1++)
    out[mu*c_nColor*c_stride + c1*c_stride + timeslice*space_stride + sid] = fetch_double2(propagator3DTex1,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * space_stride);
