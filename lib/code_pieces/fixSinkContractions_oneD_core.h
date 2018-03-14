int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ FLOAT2 shared_cache[16*THREADS_PER_BLOCK];
int c_stride_spatial = c_stride/c_localL[3];
register FLOAT2 accum[16];

int x_id[4];
int r1,r2;

r1 = sid / c_localL[0];
x_id[0] = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
x_id[1] = r1 - r2*c_localL[1];
x_id[2] = r2;
x_id[3] = it;
/*
r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id[0] = sid - r1*(c_localL[0]);
x_id[1] = r1 - r2*(c_localL[1]);
x_id[3] = r2/(c_localL[2]);
x_id[2] = r2 - x_id[3]*(c_localL[2]);
*/
//////////////////
// take forward and backward points index
int pointPlus[4]; // for prop
int pointMinus[4]; // for prop
int pointMinusG[4]; // for gauge

pointPlus[0] = LEXIC(x_id[3],x_id[2],x_id[1],(x_id[0]+1)%c_localL[0],c_localL);
pointPlus[1] = LEXIC(x_id[3],x_id[2],(x_id[1]+1)%c_localL[1],x_id[0],c_localL);
pointPlus[2] = LEXIC(x_id[3],(x_id[2]+1)%c_localL[2],x_id[1],x_id[0],c_localL);
pointPlus[3] = LEXIC((x_id[3]+1)%c_localL[3],x_id[2],x_id[1],x_id[0],c_localL);
pointMinus[0] = LEXIC(x_id[3],x_id[2],x_id[1],(x_id[0]-1+c_localL[0])%c_localL[0],c_localL);
pointMinus[1] = LEXIC(x_id[3],x_id[2],(x_id[1]-1+c_localL[1])%c_localL[1],x_id[0],c_localL);
pointMinus[2] = LEXIC(x_id[3],(x_id[2]-1+c_localL[2])%c_localL[2],x_id[1],x_id[0],c_localL);
pointMinus[3] = LEXIC((x_id[3]-1+c_localL[3])%c_localL[3],x_id[2],x_id[1],x_id[0],c_localL);

pointMinusG[0] = pointMinus[0];
pointMinusG[1] = pointMinus[1];
pointMinusG[2] = pointMinus[2];
pointMinusG[3] = pointMinus[3];

// x direction
if(c_dimBreak[0] == true){
  if(x_id[0] == c_localL[0] -1)     
    pointPlus[0] = c_plusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x_id[3],x_id[2],x_id[1],c_localL);
  if(x_id[0] == 0){
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZY(x_id[3],x_id[2],x_id[1],c_localL);
    pointMinusG[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(x_id[3],x_id[2],x_id[1],c_localL);
  }
 }
// y direction
if(c_dimBreak[1] == true){
  if(x_id[1] == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x_id[3],x_id[2],x_id[0],c_localL);
  if(x_id[1] == 0){
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TZX(x_id[3],x_id[2],x_id[0],c_localL);
    pointMinusG[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor +  LEXIC_TZX(x_id[3],x_id[2],x_id[0],c_localL);
  }
 }
//z direction
if(c_dimBreak[2] == true){
  if(x_id[2] == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x_id[3],x_id[1],x_id[0],c_localL);
  if(x_id[2] == 0){
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_TYX(x_id[3],x_id[1],x_id[0],c_localL);
    pointMinusG[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor +  LEXIC_TYX(x_id[3],x_id[1],x_id[0],c_localL);
  }
 }
//t direction
if(c_dimBreak[3] == true){
  if(x_id[3] == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x_id[2],x_id[1],x_id[0],c_localL);
  if(x_id[3] == 0){
    pointMinus[3] = c_minusGhost[3]*c_nSpin*c_nSpin*c_nColor*c_nColor + LEXIC_ZYX(x_id[2],x_id[1],x_id[0],c_localL);
    pointMinusG[3] = c_minusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(x_id[2],x_id[1],x_id[0],c_localL);
  }
 }

///////////////////
int x,y,z;
x = x_id[0] + c_procPosition[0] * c_localL[0] - x0;
y = x_id[1] + c_procPosition[1] * c_localL[1] - y0;
z = x_id[2] + c_procPosition[2] * c_localL[2] - z0;
FLOAT phase;
FLOAT2 expon;

FLOAT2 gamma[4][4];

#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) )
#define GAUGE(tex,dir,c1,c2) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( dir*3*3 + c1*3 + c2 ) * c_stride) )

#define PROPplusSur(tex,dir,mu,nu,a,b) ( FETCH_FLOAT2(tex,pointPlus[dir]  + ( (mu*4+nu)*3*3 + a*3 + b ) * c_surface[dir]) )
#define PROPminusSur(tex,dir,mu,nu,a,b) ( FETCH_FLOAT2(tex,pointMinus[dir] + ( (mu*4+nu)*3*3 + a*3 + b ) * c_surface[dir]) )
#define GAUGEminusSur(tex,dir,c1,c2) ( FETCH_FLOAT2(tex,pointMinusG[dir]  + ( dir*3*3 + c1*3 + c2 ) * c_surface[dir]) )

#define PROPplusStr(tex,dir,mu,nu,a,b) ( FETCH_FLOAT2(tex,pointPlus[dir]  + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) )
#define PROPminusStr(tex,dir,mu,nu,a,b) ( FETCH_FLOAT2(tex,pointMinus[dir] + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) )
#define GAUGEminusStr(tex,dir,c1,c2) ( FETCH_FLOAT2(tex,pointMinusG[dir]  + ( dir*3*3 + c1*3 + c2 ) * c_stride) )

for(int i = 0 ; i < 16 ; i++){
  accum[i].x = 0.; accum[i].y = 0.;
 }

if (sid < c_threads/c_localL[3]){

  for(int iop = 0 ; iop < 16 ; iop++){
    get_Operator(gamma,iop,TESTPARTICLE,partflag);

    for(int ku = 0 ; ku < 4 ; ku++)
      for(int lu = 0 ; lu < 4 ; lu++)
	for(int pu = 0 ; pu < 4 ; pu++)
	  for(int c1 = 0 ; c1 < 3 ; c1++)
	    for(int c2 = 0 ; c2 < 3 ; c2++)
	      for(int c3 = 0 ; c3 < 3 ; c3++){
		if( norm(gamma[ku][lu]) > 1e-4 ){
		  //  x x x+dir (plus) =====                                                                                                                                     
		  if(c_dimBreak[dir] == true && (x_id[dir] == c_localL[dir]-1) )
		    accum[iop] = accum[iop] + PROP(seqTex,ku,pu,c1,c2) * gamma[ku][lu] * GAUGE(gaugeTex,dir,c1,c3) * PROPplusSur(fwdTex,dir,lu,pu,c3,c2);
		  else
		    accum[iop] = accum[iop] + PROP(seqTex,ku,pu,c1,c2) * gamma[ku][lu] * GAUGE(gaugeTex,dir,c1,c3) * PROPplusStr(fwdTex,dir,lu,pu,c3,c2);
		  // x x-dir x-dir  (minus) ====
		  if(c_dimBreak[dir] == true && x_id[dir] == 0)
		    accum[iop] = accum[iop] - PROP(seqTex,ku,pu,c1,c2) * gamma[ku][lu] * conj(GAUGEminusSur(gaugeTex,dir,c3,c1)) * PROPminusSur(fwdTex,dir,lu,pu,c3,c2);
		  else
		    accum[iop] = accum[iop] - PROP(seqTex,ku,pu,c1,c2) * gamma[ku][lu] * conj(GAUGEminusStr(gaugeTex,dir,c3,c1)) * PROPminusStr(fwdTex,dir,lu,pu,c3,c2);
		  // x+dir x x (minus) =======
		  if(c_dimBreak[dir] == true && (x_id[dir] == c_localL[dir]-1) )
		    accum[iop] = accum[iop] - PROPplusSur(seqTex,dir,ku,pu,c1,c2) * gamma[ku][lu] *  conj(GAUGE(gaugeTex,dir,c3,c1)) * PROP(fwdTex,lu,pu,c3,c2);
		  else
		    accum[iop] = accum[iop] - PROPplusStr(seqTex,dir,ku,pu,c1,c2) * gamma[ku][lu] *  conj(GAUGE(gaugeTex,dir,c3,c1)) * PROP(fwdTex,lu,pu,c3,c2);
		  // x-dir x-dir x (plus) ==== 
		  if(c_dimBreak[dir] == true && x_id[dir] == 0)
		    accum[iop] = accum[iop] + PROPminusSur(seqTex,dir,ku,pu,c1,c2) * gamma[ku][lu] * GAUGEminusSur(gaugeTex,dir,c1,c3) * PROP(fwdTex,lu,pu,c3,c2);
		  else
		    accum[iop] = accum[iop] + PROPminusStr(seqTex,dir,ku,pu,c1,c2) * gamma[ku][lu] * GAUGEminusStr(gaugeTex,dir,c1,c3) * PROP(fwdTex,lu,pu,c3,c2);
		}

	      }

  }
 }

/*
if(sid == 0 && it == 0){
  for (int i = 0 ; i < 16 ; i++)
    cuPrintf("%+e %+e\n",accum[i].x,accum[i].y);
 }
*/
__syncthreads();
int i;
for(int imom = 0 ; imom < c_Nmoms ; imom++){
  phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
  expon.x = cos(phase);
  expon.y = sin(phase);
  for(int iop = 0 ; iop < 16 ; iop++){
    shared_cache[iop*THREADS_PER_BLOCK + cacheIndex] = accum[iop] * expon; 
  }
  __syncthreads();
  i = blockDim.x/2;
  while (i != 0){
    if(cacheIndex < i){
      for(int iop = 0 ; iop < 16 ; iop++){
	shared_cache[iop*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + i].x;
	shared_cache[iop*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + i].y;
      }
    }
    __syncthreads();
    i /= 2;
  }
  
  if(cacheIndex == 0){
    for(int iop = 0 ; iop < 16 ; iop++)
      block[imom*16*gridDim.x + iop*gridDim.x + blockIdx.x] = 0.25*shared_cache[iop*THREADS_PER_BLOCK + cacheIndex + 0];	
  }
  
 } // close momentum


#undef PROP
#undef GAUGE

#undef PROPplusSur
#undef PROPminusSur
#undef GAUGEminusSur

#undef PROPplusStr
#undef PROPminusStr
#undef GAUGEminusStr
