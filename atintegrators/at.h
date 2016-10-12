#ifndef AT_H
#define AT_H

#include <math.h>
#include "attypes.h"

#if !defined(MATLAB_MEX_FILE) /*Linux*/
#include <stdbool.h>
#ifndef INFINITY
static const double pinf = 1.0 / 0.0;
#define INFINITY pinf
#endif
#ifndef NAN
static const double dnan = 0.0 / 0.0;
#define NAN dnan
#endif
#endif /* defined(_WIN32) */

#ifdef MATLAB_MEX_FILE

#include "mex.h"
#include <matrix.h>

#else

#include <stdlib.h>
#define mxIsFinite isfinite
#define mxIsNaN isnan
#define mxGetNaN() (NAN)
#define mxGetInf() (INFINITY)
#define mxMalloc malloc
#define mxCalloc calloc
#define mxFree free

#if defined __SUNPRO_C
#include <ieeefp.h>
#define isfinite finite
#endif

#ifdef __MWERKS__
#endif

#endif /*MATLAB_MEX_FILE*/

#endif /*AT_H*/


