#include "atelem.c"
#include "atlalib.c"

struct elem 
{
  double Length;
  double *R1;
  double *R2;
  double *T1;
  double *T2;
  double *EApertures;
  double *RApertures;
};

void DriftPass(double *r_in, double le,
	       const double *T1, const double *T2,
	       const double *R1, const double *R2,
	       double *RApertures, double *EApertures,
	       int num_particles)
/* le - physical length
   r_in - 6-by-N matrix of initial conditions reshaped into 
   1-d array of 6*N elements 
*/
{
  double *r6;
  int c;

  for (c = 0; c<num_particles; c++) { /*Loop over particles  */
    r6 = r_in+c*6;
    if(!atIsNaN(r6[0])) {
      /*  misalignment at entrance  */
      if (T1) ATaddvv(r6, T1);
      if (R1) ATmultmv(r6, R1);
      /* Check physical apertures at the entrance of the magnet */
      if (RApertures) checkiflostRectangularAp(r6,RApertures);
      if (EApertures) checkiflostEllipticalAp(r6,EApertures);
      ATdrift6(r6, le);
      /* Check physical apertures at the exit of the magnet */
      if (RApertures) checkiflostRectangularAp(r6,RApertures);
      if (EApertures) checkiflostEllipticalAp(r6,EApertures);
      /* Misalignment at exit */
      if (R2) ATmultmv(r6, R2);
      if (T2) ATaddvv(r6, T2);
    }
  }
}

ExportMode struct elem *trackFunction2(const atElem *ElemData,struct elem *Elem,
			      double *r_in, int num_particles, struct parameters *Param)
{
/*  if (ElemData) {*/
        if (!Elem) {
            double Length=atGetDouble(ElemData,"Length",0);
            double *R1=atGetDoubleArray(ElemData,"R1",1);
            double *R2=atGetDoubleArray(ElemData,"R2",1);
            double *T1=atGetDoubleArray(ElemData,"T1",1);
            double *T2=atGetDoubleArray(ElemData,"T2",1);
            double *EApertures=atGetDoubleArray(ElemData,"EApertures",1);
            double *RApertures=atGetDoubleArray(ElemData,"RApertures",1);
            if (err_occurred()) return NULL;
            Elem = (struct elem*)atMalloc(sizeof(struct elem));
            Elem->Length=Length;
            Elem->R1=R1;
            Elem->R2=R2;
            Elem->T1=T1;
            Elem->T2=T2;
            Elem->EApertures=EApertures;
            Elem->RApertures=RApertures;
        }
        DriftPass(r_in, Elem->Length, Elem->T1, Elem->T2, Elem->R1, Elem->R2, Elem->RApertures, Elem->EApertures, num_particles);
/*  }
    else {
         atFree(Elem->T1);
         atFree(Elem->T2);
         atFree(Elem->R1);
         atFree(Elem->R2);
         atFree(Elem->EApertures);
         atFree(Elem->RApertures);
     }*/
    return Elem;
}
