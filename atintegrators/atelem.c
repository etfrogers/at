#if defined(PCWIN)
#define ExportMode __declspec(dllexport)
#else
#define ExportMode
#endif

#if defined(MATLAB_MEX_FILE)

#include <mex.h>
#include <matrix.h>

typedef mxArray atElem;
#define err_occurred() (0)
#define atIsFinite mxIsFinite
#define atIsNaN mxIsNaN
#define atGetNaN mxGetNaN
#define atGetInf mxGetInf
#define atFree mxFree

static int atGetLong(const mxArray *ElemData, const char *fieldname, int defaultValue)
{
    mxArray *field;
    if(field=mxGetField(ElemData,0,fieldname))
        return (int)mxGetScalar(field);
    else
        return defaultValue;
}

static double atGetDouble(const mxArray *ElemData, const char *fieldname, double defaultValue)
{
    mxArray *field;
    if (field=mxGetField(ElemData,0,fieldname))
        return mxGetScalar(field);
    else
        return defaultValue;
}

static double* atGetDoubleArray(const mxArray *ElemData, const char *fieldname, int optional)
{
    mxArray *field;
    if (field=mxGetField(ElemData,0,fieldname))
        return mxGetPr(field);
    else
        if (optional)
            return NULL;
        else
            mexErrMsgTxt("The required attribute %s is missing.", fieldname);
}

static void *atMalloc(size_t size)
{
    void *ptr = mxMalloc(size);
    mexMakeMemoryPersistent(ptr);
    return ptr;
}

static void *atCalloc(size_t count, size_t size)
{
    void *ptr = mxCalloc(count, size);
    mexMakeMemoryPersistent(ptr);
    return ptr;
}

#elif defined(PYAT)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <stdbool.h>

#if PY_MAJOR_VERSION >= 3
#define NUMPY_IMPORT_ARRAY_RETVAL NULL
#define NUMPY_IMPORT_ARRAY_TYPE void *
#else
#define NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_TYPE void
#define PyLong_AsLong PyInt_AsLong
#endif
#include <stdlib.h>

#ifndef NAN
static const double dnan = 0.0 / 0.0;
#define NAN dnan
#endif
#ifndef INFINITY
static const double pinf = 1.0 / 0.0;
#define INFINITY pinf
#endif

typedef PyObject atElem;
#define err_occurred PyErr_Occurred
#define atIsFinite isfinite
#define atIsNaN isnan
#define atGetNaN() (NAN)
#define atGetInf() (INFINITY)
#define atMalloc malloc
#define atCalloc calloc
#define atFree free

#if defined __SUNPRO_C
#include <ieeefp.h>
#define isfinite finite
#endif

static int array_imported = 0;

static NUMPY_IMPORT_ARRAY_TYPE init_numpy(void) {
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

static long atGetLong(const PyObject *element, char *name, long default_value) {
    long l = PyLong_AsLong(PyObject_GetAttrString((PyObject *)element, name));
    if (PyErr_Occurred()) {
        PyErr_Clear();
        l = default_value;
    }
    return l;
}

static double atGetDouble(const PyObject *element, char *name, double default_value) {
    double d = PyFloat_AsDouble(PyObject_GetAttrString((PyObject *)element, name));
    if (PyErr_Occurred()) {
        PyErr_Clear();
        d = default_value;
    }
    return d;
}

static double *atGetDoubleArray(const PyObject *element, char *name, bool optional) {
    char errmessage[60];
    if (!array_imported) {
        init_numpy();
        array_imported = 1;
    }
    PyArrayObject *array = (PyArrayObject *) PyObject_GetAttrString((PyObject *)element, name);
    if (array == NULL) {
        if (optional) {
            PyErr_Clear();
        }
        return NULL;
    }
    if (!PyArray_Check(array)) {
        snprintf(errmessage, 60, "The attribute %s is not an array.", name);
        PyErr_SetString(PyExc_RuntimeError, errmessage);
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_DOUBLE) {
        snprintf(errmessage, 60, "The attribute %s is not a double array.", name);
        PyErr_SetString(PyExc_RuntimeError, errmessage);
        return NULL;
    }
    if ((PyArray_FLAGS(array) & NPY_ARRAY_CARRAY_RO) != NPY_ARRAY_CARRAY_RO) {
        snprintf(errmessage, 60, "The attribute %s is not C-aligned.", name);
        PyErr_SetString(PyExc_RuntimeError, errmessage);
        return NULL;
    }
    return PyArray_DATA(array);
}

#endif /*PYAT*/

struct elem;

struct parameters
{
  int mode;
  int nturn;
  double RingLength;
  double T0;
};

ExportMode struct elem *trackFunction2(const atElem *ElemData, struct elem *Elem,
			      double *r_in, int num_particles, struct parameters *Param);
