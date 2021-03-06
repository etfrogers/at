
#include "atelem.c"
#include "atlalib.c"
#include "atphyslib.c"		/* edge, edge_fringe */
#include "driftkickrad.c"	/* drift6.c, bndthinkickrad.c */
#include "quadfringe.c"		/* QuadFringePassP, QuadFringePassN */

#define DRIFT1    0.6756035959798286638
#define DRIFT2   -0.1756035959798286639
#define KICK1     1.351207191959657328
#define KICK2    -1.702414383919314656

struct elem 
{
    double Length;
    double *PolynomA;
    double *PolynomB;
    int MaxOrder;
    int NumIntSteps;
    double BendingAngle;
    double EntranceAngle;
    double ExitAngle;
    double Energy;
    /* Optional fields */
    double FringeInt1;
    double FringeInt2;
    double FullGap;
    double *R1;
    double *R2;
    double *T1;
    double *T2;
    double *RApertures;
    double *EApertures;
};

void BndMPoleSymplectic4FrgFRadPass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, 	double exit_angle,
        double fint1, double fint2, double gap,
        double *T1, double *T2,
        double *R1, double *R2,
        double *RApertures, double *EApertures,
        double E0, int num_particles)
{
    double *r6;
    int c, m;
    bool useT1 = (T1 != NULL);
    bool useT2 = (T2 != NULL);
    bool useR1 = (R1 != NULL);
    bool useR2 = (R2 != NULL);
    double SL = le/num_int_steps;
    double L1 = SL*DRIFT1;
    double L2 = SL*DRIFT2;
    double K1 = SL*KICK1;
    double K2 = SL*KICK2;
    bool useFringe1 = ((fint1!=0) && (gap!=0));
    bool useFringe2 = ((fint2!=0) && (gap!=0));
    
    for (c = 0; c<num_particles; c++) {	/* Loop over particles  */
        r6 = r+c*6;
        /*  misalignment at entrance  */
        if(useT1) ATaddvv(r6,T1);
        if(useR1) ATmultmv(r6,R1);
        /* Check physical apertures at the entrance of the magnet */
        if (RApertures) checkiflostRectangularAp(r6,RApertures);
        if (EApertures) checkiflostEllipticalAp(r6,EApertures);
        /* edge focus */
        if(useFringe1)
            edge_fringe(r6, irho, entrance_angle,fint1,gap);
        else
            edge(r6, irho, entrance_angle);
        /* quadrupole gradient fringe */
        QuadFringePassP(r6,B[1]);
        /* integrator  */
        for(m=0; m < num_int_steps; m++){ /* Loop over slices */
            drift6(r6,L1);
            bndthinkickrad(r6, A, B, K1, irho, E0, max_order);
            drift6(r6,L2);
            bndthinkickrad(r6, A, B, K2, irho, E0, max_order);
            drift6(r6,L2);
            bndthinkickrad(r6, A, B,  K1, irho, E0, max_order);
            drift6(r6,L1);
        }
        /* quadrupole gradient fringe */
        QuadFringePassN(r6,B[1]);
        /* edge focus */
        if(useFringe2)
            edge_fringe(r6, irho, exit_angle,fint2,gap);
        else
            edge(r6, irho, exit_angle);
        /* Check physical apertures at the exit of the magnet */
        if (RApertures) checkiflostRectangularAp(r6,RApertures);
        if (EApertures) checkiflostEllipticalAp(r6,EApertures);
        /* Misalignment at exit */
        if(useR2) ATmultmv(r6,R2);
        if(useT2) ATaddvv(r6,T2);
    }
}

#if defined(MATLAB_MEX_FILE) || defined(PYAT)
ExportMode struct elem *trackFunction(const atElem *ElemData,struct elem *Elem,
			      double *r_in, int num_particles, struct parameters *Param)
{
    double irho;
    if (!Elem) {
        double Length, BendingAngle, EntranceAngle, ExitAngle, FullGap, FringeInt1, FringeInt2, Energy;
        int MaxOrder, NumIntSteps;
        double *PolynomA, *PolynomB, *R1, *R2, *T1, *T2, *EApertures, *RApertures;
        Length=atGetDouble(ElemData,"Length"); check_error();
        PolynomA=atGetDoubleArray(ElemData,"PolynomA"); check_error();
        PolynomB=atGetDoubleArray(ElemData,"PolynomB"); check_error();
        MaxOrder=atGetLong(ElemData,"MaxOrder"); check_error();
        NumIntSteps=atGetLong(ElemData,"NumIntSteps"); check_error();
        BendingAngle=atGetDouble(ElemData,"BendingAngle"); check_error();
        EntranceAngle=atGetDouble(ElemData,"EntranceAngle"); check_error();
        ExitAngle=atGetDouble(ElemData,"ExitAngle"); check_error();
        Energy=atGetDouble(ElemData,"Energy"); check_error();
        /*optional fields*/
        FringeInt1=atGetOptionalDouble(ElemData,"FringeInt1",0); check_error();
        FringeInt2=atGetOptionalDouble(ElemData,"FringeInt2",0); check_error();
        FullGap=atGetOptionalDouble(ElemData,"FullGap",0); check_error();
        R1=atGetOptionalDoubleArray(ElemData,"R1"); check_error();
        R2=atGetOptionalDoubleArray(ElemData,"R2"); check_error();
        T1=atGetOptionalDoubleArray(ElemData,"T1"); check_error();
        T2=atGetOptionalDoubleArray(ElemData,"T2"); check_error();
        EApertures=atGetOptionalDoubleArray(ElemData,"EApertures"); check_error();
        RApertures=atGetOptionalDoubleArray(ElemData,"RApertures"); check_error();
        Elem = (struct elem*)atMalloc(sizeof(struct elem));
        Elem->Length=Length;
        Elem->PolynomA=PolynomA;
        Elem->PolynomB=PolynomB;
        Elem->MaxOrder=MaxOrder;
        Elem->NumIntSteps=NumIntSteps;
        Elem->BendingAngle=BendingAngle;
        Elem->EntranceAngle=EntranceAngle;
        Elem->ExitAngle=ExitAngle;
        Elem->Energy=Energy;
        /*optional fields*/
        Elem->FringeInt1=FringeInt1;
        Elem->FringeInt2=FringeInt2;
        Elem->FullGap=FullGap;
        Elem->R1=R1;
        Elem->R2=R2;
        Elem->T1=T1;
        Elem->T2=T2;
        Elem->EApertures=EApertures;
        Elem->RApertures=RApertures;
    }
    irho = Elem->BendingAngle/Elem->Length;
    BndMPoleSymplectic4FrgFRadPass(r_in,Elem->Length,irho,Elem->PolynomA,Elem->PolynomB,
            Elem->MaxOrder,Elem->NumIntSteps,Elem->EntranceAngle,Elem->ExitAngle,
            Elem->FringeInt1,Elem->FringeInt2,Elem->FullGap,Elem->T1,Elem->T2,
            Elem->R1,Elem->R2,Elem->RApertures,Elem->EApertures,Elem->Energy,num_particles);
    return Elem;
}

MODULE_DEF(BndMPoleSymplectic4FrgFRadPass)        /* Dummy module initialisation */

#endif /*defined(MATLAB_MEX_FILE) || defined(PYAT)*/

#if defined(MATLAB_MEX_FILE)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs == 2) {
        double Length, BendingAngle, EntranceAngle, ExitAngle, FullGap, FringeInt1, FringeInt2, Energy;
        int MaxOrder, NumIntSteps;
        double *PolynomA, *PolynomB, *R1, *R2, *T1, *T2, *EApertures, *RApertures;
        double irho;
        double *r_in;
        const mxArray *ElemData = prhs[0];
        int num_particles = mxGetN(prhs[1]);
        Length=atGetDouble(ElemData,"Length"); check_error();
        PolynomA=atGetDoubleArray(ElemData,"PolynomA"); check_error();
        PolynomB=atGetDoubleArray(ElemData,"PolynomB"); check_error();
        MaxOrder=atGetLong(ElemData,"MaxOrder"); check_error();
        NumIntSteps=atGetLong(ElemData,"NumIntSteps"); check_error();
        BendingAngle=atGetDouble(ElemData,"BendingAngle"); check_error();
        EntranceAngle=atGetDouble(ElemData,"EntranceAngle"); check_error();
        ExitAngle=atGetDouble(ElemData,"ExitAngle"); check_error();
        Energy=atGetDouble(ElemData,"Energy"); check_error();
        /*optional fields*/
        FringeInt1=atGetOptionalDouble(ElemData,"FringeInt1",0); check_error();
        FringeInt2=atGetOptionalDouble(ElemData,"FringeInt2",0); check_error();
        FullGap=atGetOptionalDouble(ElemData,"FullGap",0); check_error();
        R1=atGetOptionalDoubleArray(ElemData,"R1"); check_error();
        R2=atGetOptionalDoubleArray(ElemData,"R2"); check_error();
        T1=atGetOptionalDoubleArray(ElemData,"T1"); check_error();
        T2=atGetOptionalDoubleArray(ElemData,"T2"); check_error();
        EApertures=atGetOptionalDoubleArray(ElemData,"EApertures"); check_error();
        RApertures=atGetOptionalDoubleArray(ElemData,"RApertures"); check_error();
        irho = BendingAngle/Length;
        /* ALLOCATE memory for the output array of the same size as the input  */
        plhs[0] = mxDuplicateArray(prhs[1]);
        r_in = mxGetPr(plhs[0]);
        BndMPoleSymplectic4FrgFRadPass(r_in, Length, irho, PolynomA, PolynomB,
                MaxOrder,NumIntSteps,EntranceAngle,ExitAngle,FringeInt1,FringeInt2,
                FullGap,T1,T2,R1,R2,RApertures,EApertures,Energy,num_particles);
    }
    else if (nrhs == 0) {
        /* list of required fields */
        plhs[0] = mxCreateCellMatrix(9,1);
        mxSetCell(plhs[0],0,mxCreateString("Length"));
        mxSetCell(plhs[0],1,mxCreateString("BendingAngle"));
        mxSetCell(plhs[0],2,mxCreateString("EntranceAngle"));
        mxSetCell(plhs[0],3,mxCreateString("ExitAngle"));
        mxSetCell(plhs[0],4,mxCreateString("PolynomA"));
        mxSetCell(plhs[0],5,mxCreateString("PolynomB"));
        mxSetCell(plhs[0],6,mxCreateString("MaxOrder"));
        mxSetCell(plhs[0],7,mxCreateString("NumIntSteps"));
        mxSetCell(plhs[0],8,mxCreateString("Energy"));
        if (nlhs>1) {
            /* list of optional fields */
            plhs[1] = mxCreateCellMatrix(9,1);
            mxSetCell(plhs[1],0,mxCreateString("FullGap"));
            mxSetCell(plhs[1],1,mxCreateString("FringeInt1"));
            mxSetCell(plhs[1],2,mxCreateString("FringeInt2"));
            mxSetCell(plhs[1],3,mxCreateString("T1"));
            mxSetCell(plhs[1],4,mxCreateString("T2"));
            mxSetCell(plhs[1],5,mxCreateString("R1"));
            mxSetCell(plhs[1],6,mxCreateString("R2"));
            mxSetCell(plhs[1],7,mxCreateString("RApertures"));
            mxSetCell(plhs[1],8,mxCreateString("EApertures"));
        }
    }
    else {
        mexErrMsgIdAndTxt("AT:WrongArg","Needs 0 or 2 arguments");
    }
}
#endif /* MATLAB_MEX_FILE */