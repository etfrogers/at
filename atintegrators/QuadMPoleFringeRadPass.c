#include "mex.h"
#include "elempass.h"
#include "atlalib.c"
#include "driftkickrad.c"	/* drift6.c, strthinkickrad.c */
#include "quadfringe.c"		/* QuadFringePassP, QuadFringePassN */

#define DRIFT1    0.6756035959798286638
#define DRIFT2   -0.1756035959798286639
#define KICK1     1.351207191959657328
#define KICK2    -1.702414383919314656


void QuadMPoleFringeRadPass(double *r, double le, const double *A, const double *B,
			 				double E0, int max_order, int num_int_steps,
			 				const double *T1, const double *T2,	
			 				const double *R1, const double *R2, 
			 				double *fringeIntM0,  /* I0m/K1, I1m/K1, I2m/K1, I3m/K1, Lambda2m/K1 */
                		 	double *fringeIntP0,  /* I0p/K1, I1p/K1, I2p/K1, I3p/K1, Lambda2p/K1 */
						 	double *limits, double *axesptr,int num_particles)
{
   double *r6;
   int c, m;
   bool useT1 = (T1 != NULL);
   bool useT2 = (T2 != NULL);
   bool useR1 = (R1 != NULL);
   bool useR2 = (R2 != NULL);
   bool useLinFrEle = (fringeIntM0 != NULL && fringeIntP0 != NULL);
   double SL = le/num_int_steps;
   double L1 = SL*DRIFT1;
   double L2 = SL*DRIFT2;
   double K1 = SL*KICK1;
   double K2 = SL*KICK2;
   	double *fringeIntM, *fringeIntP;
	double delta;
    double inFringe; 	/*  for linear fringe field, from elegant. 
						The argument inFringe is a flag: 
						-1 for going into the magnet and +1 for going out. */
	VMATRIX *M;
	int i, j;
   
	for (c = 0;c<num_particles;c++) {	/*Loop over particles  */
       r6 = r+c*6;
       if (!mxIsNaN(r6[0])) {
           /*  misalignment at entrance  */
           if (useT1) ATaddvv(r6, T1);
           if (useR1) ATmultmv(r6, R1);
			/* Check physical apertures at the entrance of the magnet */
			if (limits) checkiflostRectangularAp(r6,limits);
			if (axesptr) checkiflostEllipticalAp(r6,axesptr);
			if (useLinFrEle) /*Linear fringe fields from elegant*/
			{
			M=mxMalloc(sizeof(VMATRIX));

			/* quadrupole linear fringe field, from elegant code */	
			inFringe=-1.0;	
		    fringeIntM = fringeIntP0;
		    fringeIntP = fringeIntM0;
		    delta = r6[4];
		    /* determine first linear matrix for this delta */
    		quadPartialFringeMatrix(M, B[1]/(1+delta), inFringe, fringeIntM, 1);

    		r6[0] = M->R[0][0]*r6[0] + M->R[0][1]*r6[1];
    		r6[1] = M->R[1][0]*r6[0] + M->R[1][1]*r6[1];
    		r6[2] = M->R[2][2]*r6[2] + M->R[2][3]*r6[3];
    		r6[3] = M->R[3][2]*r6[2] + M->R[3][3]*r6[3];
    		}
			/* nonlinear fringe field */
           QuadFringePassP(r6,B[1]);
			if (useLinFrEle) /*Linear fringe fields from elegant*/
			{
			inFringe=-1.0;
			/* determine and apply second linear matrix, from elegant code */
    		quadPartialFringeMatrix(M, B[1]/(1+delta), inFringe, fringeIntP, 2);
    		r6[0] = M->R[0][0]*r6[0] + M->R[0][1]*r6[1];
    		r6[1] = M->R[1][0]*r6[0] + M->R[1][1]*r6[1];
    		r6[2] = M->R[2][2]*r6[2] + M->R[2][3]*r6[3];
    		r6[3] = M->R[3][2]*r6[2] + M->R[3][3]*r6[3];
			}	/* end of elegant code*/

           /*  integrator  */
           r6 = r+c*6;
           for (m=0; m < num_int_steps; m++) { /*  Loop over slices */
               drift6(r6,L1);
               strthinkickrad(r6, A, B,  K1, E0,max_order);
               drift6(r6,L2);
               strthinkickrad(r6, A, B, K2, E0,max_order);
               drift6(r6,L2);
               strthinkickrad(r6, A, B,  K1, E0,max_order);
               drift6(r6,L1);
           }

			if (useLinFrEle) /*Linear fringe fields from elegant*/
			{
			/* quadrupole linear fringe field, from elegant code */	
			inFringe=1.0;	
		    fringeIntM = fringeIntM0;
    		fringeIntP = fringeIntP0;
		    delta = r6[4];
		    /* determine first linear matrix for this delta */
    		quadPartialFringeMatrix(M, B[1]/(1+delta), inFringe, fringeIntM, 1);

    		r6[0] = M->R[0][0]*r6[0] + M->R[0][1]*r6[1];
    		r6[1] = M->R[1][0]*r6[0] + M->R[1][1]*r6[1];
    		r6[2] = M->R[2][2]*r6[2] + M->R[2][3]*r6[3];
    		r6[3] = M->R[3][2]*r6[2] + M->R[3][3]*r6[3];
    		}

           QuadFringePassN(r6,B[1]);
			
			if (useLinFrEle) /*Linear fringe fields from elegant*/
			{
			inFringe=1.0;
			/* determine and apply second linear matrix, from elegant code */
    		quadPartialFringeMatrix(M, B[1]/(1+delta), inFringe, fringeIntP, 2);
    
    		r6[0] = M->R[0][0]*r6[0] + M->R[0][1]*r6[1];
    		r6[1] = M->R[1][0]*r6[0] + M->R[1][1]*r6[1];
    		r6[2] = M->R[2][2]*r6[2] + M->R[2][3]*r6[3];
    		r6[3] = M->R[3][2]*r6[2] + M->R[3][3]*r6[3];
			mxFree(M); /* deallocate memory of VMATRIX M */
			}	/* end of elegant code*/

			/* Check physical apertures at the exit of the magnet */
			if (limits) checkiflostRectangularAp(r6,limits);
			if (axesptr) checkiflostEllipticalAp(r6,axesptr);

           /* Misalignment at exit */
           if (useR2) ATmultmv(r6, R2);
           if (useT2) ATaddvv(r6, T2);
       }
   }
}

#ifndef NOMEX

#include "mxutils.c"

ExportMode int* passFunction(const mxArray *ElemData, int *FieldNumbers,
			     double *r_in, int num_particles, int mode)
#define NUM_FIELDS_2_REMEMBER 13
{
   double *A , *B;
   double  *pr1, *pr2, *pt1, *pt2, *limits, *axesptr, *fringeIntM0, *fringeIntP0;   
   int max_order, num_int_steps;
   double le, E0;
   
   switch(mode) {
      case MAKE_LOCAL_COPY: 	/* Find field numbers first
       Save a list of field number in an array
       and make returnptr point to that array
       */
	 /* Allocate memory for integer array of 
	  field numbers for faster futurereference
	  */
	 FieldNumbers = (int *) mxCalloc(NUM_FIELDS_2_REMEMBER, sizeof(int));
	 
	 /*  Populate */
	 
	 FieldNumbers[0] = GetRequiredFieldNumber(ElemData, "PolynomA");
	 FieldNumbers[1] = GetRequiredFieldNumber(ElemData, "PolynomB");
	 FieldNumbers[2] = GetRequiredFieldNumber(ElemData, "MaxOrder");
	 FieldNumbers[3] = GetRequiredFieldNumber(ElemData, "NumIntSteps");
	 FieldNumbers[4] = GetRequiredFieldNumber(ElemData, "Length");
	 FieldNumbers[5] = GetRequiredFieldNumber(ElemData, "Energy");
	 
	 FieldNumbers[6] = mxGetFieldNumber(ElemData,"R1");
	 FieldNumbers[7] = mxGetFieldNumber(ElemData,"R2");
	 FieldNumbers[8] = mxGetFieldNumber(ElemData,"T1");
	 FieldNumbers[9] = mxGetFieldNumber(ElemData,"T2");
	 FieldNumbers[9] = mxGetFieldNumber(ElemData,"RApertures");
     FieldNumbers[10] = mxGetFieldNumber(ElemData,"EApertures");
	 FieldNumbers[11] = mxGetFieldNumber(ElemData,"fringeIntM0");
     FieldNumbers[12] = mxGetFieldNumber(ElemData,"fringeIntP0");

	 /* Fall through next section... */
	 
      case	USE_LOCAL_COPY:	/* Get fields from MATLAB using field numbers
       The second argument pointer to the array of field 
       numbers is previously created with 
       QuadMPoleFringeRadPass(..., MAKE_LOCAL_COPY)
       */
	 A = mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[0]));
	 B = mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[1]));
	 max_order = (int)mxGetScalar(mxGetFieldByNumber(ElemData, 0, FieldNumbers[2]));
	 num_int_steps = (int)mxGetScalar(mxGetFieldByNumber(ElemData, 0, FieldNumbers[3]));
	 le = mxGetScalar(mxGetFieldByNumber(ElemData,0,FieldNumbers[4]));
	 E0 = mxGetScalar(mxGetFieldByNumber(ElemData,0,FieldNumbers[5]));
	 
	 /* Optional fields */
	 pr1 = (FieldNumbers[6] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[6])) : NULL;
	 pr2 = (FieldNumbers[7] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[7])) : NULL;
	 pt1 = (FieldNumbers[8] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[8])) : NULL;
	 pt2 = (FieldNumbers[9] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[9])) : NULL;
	 limits = (FieldNumbers[9] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[9])) : NULL;
	axesptr = (FieldNumbers[10] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[10])) : NULL;
	fringeIntM0 = (FieldNumbers[11] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[11])) : NULL;
	fringeIntP0 = (FieldNumbers[12] >= 0) ? mxGetPr(mxGetFieldByNumber(ElemData, 0, FieldNumbers[12])) : NULL;
	 break;

      default:
	 mexErrMsgTxt("No match for calling mode in function QuadMPoleFringeRadPass\n");

   }
   
   QuadMPoleFringeRadPass(r_in, le, A, B, E0, max_order, num_int_steps,
		       pt1, pt2, pr1, pr2, fringeIntM0, fringeIntP0, limits, axesptr, num_particles);
   return FieldNumbers;
}

void mexFunction(	int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   int m,n;
   double *r_in;
   double E0, le, *A, *B, *pr1, *pr2, *pt1, *pt2, *limits, *axesptr, *fringeIntM0, *fringeIntP0;  
   int max_order, num_int_steps;
   mxArray *tmpmxptr;
   
   if (nrhs==2) {
        double *r_in;
        double *pr1, *pr2, *pt1, *pt2;
        mxArray *tmpmxptr;
        
        double *A = mxGetPr(GetRequiredField(prhs[0], "PolynomA"));
        double *B = mxGetPr(GetRequiredField(prhs[0], "PolynomB"));
        int max_order = (int) mxGetScalar(GetRequiredField(prhs[0], "MaxOrder"));
        int num_int_steps = (int) mxGetScalar(GetRequiredField(prhs[0], "NumIntSteps"));
        double le = mxGetScalar(GetRequiredField(prhs[0], "Length"));
 	double E0 = mxGetScalar(GetRequiredField(prhs[0], "E0"));
        int num_particles = mxGetN(prhs[1]);
        if (mxGetM(prhs[1]) != 6) mexErrMsgIdAndTxt("AT:WrongArg","Second argument must be a 6 x N matrix");
      
        
        /* Optional arguments */
        tmpmxptr = mxGetField(prhs[0],0,"R1");
        pr1 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
        
        tmpmxptr = mxGetField(prhs[0],0,"R2");
        pr2 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
        
        tmpmxptr = mxGetField(prhs[0],0,"T1");
        pt1 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
        
        tmpmxptr = mxGetField(prhs[0],0,"T2");
        pt2 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
      
		tmpmxptr = mxGetField(prhs[0],0,"RApertures");
        limits = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
        
        tmpmxptr = mxGetField(prhs[0],0,"EApertures");
        axesptr = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;

		tmpmxptr = mxGetField(prhs[0],0,"fringeIntM0");
        fringeIntM0 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;
        
        tmpmxptr = mxGetField(prhs[0],0,"fringeIntP0");
        fringeIntP0 = tmpmxptr ? mxGetPr(tmpmxptr) : NULL;

        /* ALLOCATE memory for the output array of the same size as the input  */
        plhs[0] = mxDuplicateArray(prhs[1]);
        r_in = mxGetPr(plhs[0]);
	QuadMPoleFringeRadPass(r_in, le, A, B, E0, max_order, num_int_steps,
			  pt1, pt2, pr1, pr2, fringeIntM0, fringeIntP0, limits, axesptr, n);
   }
   else if (nrhs == 0) {
      /* list of required fields */
      plhs[0] = mxCreateCellMatrix(6,1);
      mxSetCell(plhs[0],0,mxCreateString("Length"));
      mxSetCell(plhs[0],1,mxCreateString("PolynomA"));
      mxSetCell(plhs[0],2,mxCreateString("PolynomB"));
      mxSetCell(plhs[0],3,mxCreateString("MaxOrder"));
      mxSetCell(plhs[0],4,mxCreateString("NumIntSteps"));	    
      mxSetCell(plhs[0],5,mxCreateString("Energy"));	    
      
      if (nlhs>1) {
	/* list of optional fields */
	 plhs[1] = mxCreateCellMatrix(8,1);
	 mxSetCell(plhs[1],0,mxCreateString("T1"));
	 mxSetCell(plhs[1],1,mxCreateString("T2"));
	 mxSetCell(plhs[1],2,mxCreateString("R1"));
	 mxSetCell(plhs[1],3,mxCreateString("R2"));
	mxSetCell(plhs[1],4,mxCreateString("RApertures"));
	mxSetCell(plhs[1],5,mxCreateString("EApertures"));
	mxSetCell(plhs[1],6,mxCreateString("fringeIntM0"));
	mxSetCell(plhs[1],7,mxCreateString("fringeIntP0"));
      }
   }
   else {
    mexErrMsgIdAndTxt("AT:WrongArg","Needs 0 or 2 arguments");
   }
}
#endif
