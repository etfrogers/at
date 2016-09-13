#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

static int array_imported = 0;

#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
init_numpy(void) {
    import_array();
}

static long py_get_long(PyObject *element, char *name) {
    return PyInt_AsLong(PyObject_GetAttrString(element, name));
}

static double py_get_double(PyObject *element, char *name) {
    return PyFloat_AsDouble(PyObject_GetAttrString(element, name));
}

static double *numpy_get_double_array(PyObject *element, char *name) {
    if (!array_imported) {
        init_numpy();
        array_imported = 1;
    }
    PyObject *array;
    if (PyObject_HasAttrString(element, name)) {
        array = PyObject_GetAttrString(element, name);
    } else {
        return NULL;
    }
    double *arin;
    if (!PyArray_Check(array)) {
        printf("%s not an array!\n", name);
        return NULL;
    }
    npy_intp dims[3];
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (!PyArray_AsCArray((PyObject **)&array, (void *)&arin, dims, 1, descr) < 0) {
        printf("Conversion failed.\n");
        return NULL;
    }
    return arin;
}
