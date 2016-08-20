/*
 * 1. Register single function atpass
 * 2. Interpret list of objects
 * 3. Go round ring
 * 4. Call functions
 * 5. Return numpy array.
 */

#include <Python.h>
#include <stdio.h>
#include <numpy/ndarrayobject.h>

// Linux only
#include <dlfcn.h>

#define MAX_ORDER 3
#define MAX_INT_STEPS 5
#define INTEGRATOR_PATH "../atintegrators/"
#define ATPY_PASS "atpyPass"

/* Directly copied from atpass.c */
static struct LibraryListElement {
    char *LibraryFileName;
    char *MethodName;
    void *FunctionHandle;
    struct LibraryListElement *Next;
} *LibraryList = NULL;

struct LibraryListElement* SearchLibraryList(struct LibraryListElement *head, const char *method_name)
{
    /* recusively search the list to check if the library containing method_name is
     * already loaded. If it is - retutn the pointer to the list element. If not -
     * return NULL */
    if (head)
        return (strcmp(head->MethodName, method_name)==0) ? head :
            SearchLibraryList(head->Next, method_name);
    else
        return NULL;
}

typedef void (*pass_function)(double *rin, int num_particles, PyObject *element);

int call_pass_method(double *rin, PyObject *element, char *fn_name) {
	char *lib_file = malloc(sizeof(char) * 1000);
	struct LibraryListElement *LibraryListPtr;
	void *fn_handle;
	strcpy(lib_file, INTEGRATOR_PATH);
	strcat(lib_file, fn_name);
	strcat(lib_file, ".so");
	LibraryListPtr = SearchLibraryList(LibraryList, fn_name);
	if (LibraryListPtr) {
		fn_handle = LibraryListPtr->FunctionHandle;
	} else {
		printf("Trying to load %s\n", lib_file);
		void *dl_handle = dlopen(lib_file, RTLD_LAZY);
		char *error = dlerror();
		if (error) {
			printf("%s\n", error);
		}
		fn_handle = dlsym(dl_handle, ATPY_PASS);
		error = dlerror();
		if (error) {
			printf("Error loading %s from %s: %s\n", ATPY_PASS, lib_file, error);
		}
		printf("Loaded %p\n", fn_handle);
		LibraryListPtr = (struct LibraryListElement *)malloc(sizeof(struct LibraryListElement));
		LibraryListPtr->Next = LibraryList;
		LibraryListPtr->MethodName = fn_name;
		LibraryListPtr->FunctionHandle = fn_handle;
		LibraryList = LibraryListPtr;
	}
	pass_function pfn;
	pfn = (pass_function) fn_handle;
	pfn(rin, 1, element);
	return 0;
}


int pass_element(double *rin, PyObject *element) {
	if (!PyObject_HasAttrString(element, "pass_method")) {
		printf("No pass method.\n");
		return 1;
	}
	PyObject *fn_name_object = PyObject_GetAttrString(element, "pass_method");
	char *fn_name = PyString_AsString(fn_name_object);
	call_pass_method(rin, element, fn_name);
	return 0;
}


/*
 * Arguments:
 *  - the_ring: sequence of elements
 *  - rin: numpy 6-vector of initial conditions
 *  - num_turns: int number of turns to simulate
 */
static PyObject *at_atpass(PyObject *self, PyObject *args) {
	PyObject *element_list;
	PyArrayObject *rin;
	double *arin;
	int num_turns;
	int i, j;

	if (!PyArg_ParseTuple(args, "O!O!i", &PyList_Type, &element_list, &PyArray_Type, &rin, &num_turns)) {
		return NULL;
	}
	if (!PyArray_Check(rin)) {
		printf("not an array!\n");
		return NULL;
	}

	npy_intp dims[1];
	PyArray_Descr *descr;
	descr = PyArray_DescrFromType(NPY_DOUBLE);
	if (!PyArray_AsCArray((PyObject **)&rin, (void *)&arin, dims, 1, descr) < 0) {
		printf("Conversion failed.\n");
		return NULL;
	}
	int num_elements = PyList_Size(element_list);
	printf("There are %d elements in the list\n", num_elements);
	printf("Going for %d turns\n", num_turns);
	for (i = 0; i < num_turns; i++) {
		for (j = 0; j < num_elements; j++) {
			PyObject *element = PyList_GetItem(element_list, j);
			pass_element(arin, element);
		}
	}
	return Py_BuildValue("i", 1);
}

/* Boilerplate to register methods. */

static PyMethodDef AtMethods[] = {
	{"atpass",  at_atpass, METH_VARARGS,
	"Python clone of atpass"},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initat(void) {
	import_array();
	(void) Py_InitModule("at", AtMethods);
}


int main(int argc, char *argv[]) {
	/* Pass argv[0] to the Python interpreter */
	Py_SetProgramName(argv[0]);

	/* Initialize the Python interpreter.  Required. */
	Py_Initialize();

	/* Add a static module */
	initat();
	return 0;
}

