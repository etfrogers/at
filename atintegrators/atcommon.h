#ifndef ATCOMMON_H
#define ATCOMMON_H

#ifdef PYAT
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_MAJOR_VERSION >= 3
#define NUMPY_IMPORT_ARRAY_RETVAL NULL
#define NUMPY_IMPORT_ARRAY_TYPE void *
#else
#define NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_TYPE void
#define PyLong_AsLong PyInt_AsLong
#endif

#endif /*PYAT*/

#include <stdlib.h>
#include <math.h>

#ifdef MATLAB_MEX_FILE

#include "mex.h"
#include <matrix.h>

#else

#if defined(_WIN32)

#include <Windows.h>
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define isfinite(x) _finite(x)
/* See https://blogs.msdn.microsoft.com/oldnewthing/20100305-00/?p=14713 */
DECLSPEC_SELECTANY extern const float FLOAT_NaN = ((float)((1e308 * 10)*0.));
#define NAN FLOAT_NaN
DECLSPEC_SELECTANY extern const float FLOAT_POSITIVE_INFINITY = ((float)(1e308 * 10));
#define INFINITY FLOAT_POSITIVE_INFINITY
typedef int bool;
#define false 0
#define true 1

#else /* !defined(_WIN32) */

#if defined __SUNPRO_C
#include <ieeefp.h>
#define isfinite finite
#endif
#ifndef NAN
static const double dnan = 0.0 / 0.0;
#define NAN dnan
#endif
#ifndef INFINITY
static const double pinf = 1.0 / 0.0;
#define INFINITY pinf
#endif
#include <stdbool.h>

#endif /* defined(_WIN32) */

#define mxIsFinite isfinite
#define mxIsNaN isnan
#define mxGetNaN() (NAN)
#define mxGetInf() (INFINITY)
#define mxMalloc malloc
#define mxCalloc calloc
#define mxFree free

#endif /*MATLAB_MEX_FILE*/

#if defined(PCWIN) || defined(_WIN32)
#define ExportMode __declspec(dllexport)
#else
#define ExportMode
#endif

struct elem;

struct parameters
{
  int nturn;
  double RingLength;
  double T0;
};

#endif /*ATCOMMON_H*/
