MEX=/dls_sw/prod/tools/RHEL6-x86_64/matlab/R2014a/prefix/bin/mex

FLAGS=-DGLNXA64 CFLAGS="-std=c99 -fPIC -shared" -DATPY
PYTHON=/dls_sw/prod/tools/RHEL6-x86_64/Python/2-7-3/prefix/include/python2.7/
LPYTHON=/tmp/python
NUMPY=/dls_sw/prod/tools/RHEL6-x86_64/numpy/1-11-1/prefix/lib/python2.7/site-packages/numpy-1.11.1-py2.7-linux-x86_64.egg/numpy/core/include/
LINKEXPORT='LINKEXPORT=-Wl,--version-script,$(CURDIR)/mexFunctionGLNX86.map'

SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.mexa64)

main: $(OBJ)

%.mexa64: %.c
	$(MEX) -I$(NUMPY) -I$(PYTHON) -L$(LPYTHON) $(FLAGS) $(LINKEXPORT) $^ mxutils.c -lpython2.7

clean:
	rm *.mexa64
