#include <Python.h>
#include <numpy/ndarrayobject.h>

long py_get_long(PyObject *element, char *name) {
	return PyInt_AsLong(PyObject_GetAttrString(element, name));
}

double py_get_double(PyObject *element, char *name) {
	return PyFloat_AsDouble(PyObject_GetAttrString(element, name));
}

double *get_t1(PyObject *element) {
	return NULL;
}

double *get_t2(PyObject *element) {
	return NULL;
}

double *get_r1(PyObject *element) {
	return NULL;
}

double *get_r2(PyObject *element) {
	return NULL;
}
