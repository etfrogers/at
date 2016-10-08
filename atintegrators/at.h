#ifndef AT_H
#define AT_H

#include <math.h>

#if defined(_WIN32)
#include <Windows.h>
#include <math.h>
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
DECLSPEC_SELECTANY extern const float FLOAT_NaN = ((float)((1e308 * 10)*0.));
#define NAN FLOAT_NaN
DECLSPEC_SELECTANY extern const float FLOAT_POSITIVE_INFINITY = ((float)(1e308 * 10));
#define INFINITY FLOAT_POSITIVE_INFINITY
typedef int bool;
#define false 0
#define true 1
#elif !defined(MATLAB_MEX_FILE)
#include <stdbool.h>
static const double pinf = 1.0 / 0.0;
#define INFINITY pinf
static const double dnan = 0.0 / 0.0;
#define NAN dnan
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


