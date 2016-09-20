#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <stdbool.h>

static int array_imported = 0;

#if PY_MAJOR_VERSION >= 3
int
#else
void
#endif
init_numpy(void) {
    import_array();
}

static long py_get_long(PyObject *element, char *name, bool optional) {
    if (PyErr_Occurred()) {
        return -1;
    }
    long l = PyInt_AsLong(PyObject_GetAttrString(element, name));
    if (PyErr_Occurred()) {
        if (optional) {
            PyErr_Clear();
            return 0;
        } else {
            PyErr_SetString(PyExc_AttributeError, "Element missing required integer attribute.");
        }
    }
    return l;
}

static double py_get_double(PyObject *element, char *name, bool optional) {
    if (PyErr_Occurred()) {
        return -1;
    }
    double d = PyFloat_AsDouble(PyObject_GetAttrString(element, name));
    if (PyErr_Occurred()) {
        if (optional) {
            PyErr_Clear();
            return 0;
        } else {
            PyErr_SetString(PyExc_AttributeError, "Element missing required double attribute.");
        }
    }
    return d;
}

static double *numpy_get_double_array(PyObject *element, char *name, bool optional) {
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (!array_imported) {
        init_numpy();
        array_imported = 1;
    }
    PyArrayObject *array = (PyArrayObject *) PyObject_GetAttrString(element, name);
    if (array == NULL) {
        if (optional) {
            PyErr_Clear();
        }
        return NULL;
    }

    if (!PyArray_Check(array)) {
        printf("%s not an array\n", name);
        PyErr_SetString(PyExc_RuntimeError, "Attribute not a numpy array.");
        array = NULL;
    }
    if (PyArray_TYPE(array) != NPY_DOUBLE) {
        printf("%s not double\n", name);
        PyErr_SetString(PyExc_RuntimeError, "Attribute not a double array.");
        array = NULL;
    }
    if ((PyArray_FLAGS(array) & NPY_ARRAY_CARRAY_RO) != NPY_ARRAY_CARRAY_RO) {
        printf("%s not C aligned\n", name);
        PyErr_SetString(PyExc_RuntimeError, "Attribute array not C-aligned.");
        array = NULL;
    }
    return PyArray_DATA(array);
}
