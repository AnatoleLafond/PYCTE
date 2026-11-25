/* pycte._phreeqcRow: one row of the selected output of an IPhreeqc instance, read in a single call from Python.

   phreeqc.selectedOutputRow calls the IPhreeqc function GetSelectedOutputValue once per column (about 170 calls per
   node with ctypes, ~1 us each). read() makes the same calls, in the same order, into the same VAR structures, from
   C: the values are the same, read faster. after() makes, in one call, every call that follows the run of a node
   (warnings, row and column counts, last row). */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* IPQ_RESULT GetSelectedOutputValue(int id, int row, int col, VAR* pVAR) of IPhreeqc */
typedef int (*GetValue)(int id, int row, int col, void *var);

/* read(function, id, row, ncols, vars, stride) -> code
   function: address of GetSelectedOutputValue; vars: address of ncols VAR structures, `stride` bytes apart.
   Calls function(id, row, col, vars + col * stride) for col = 0 .. ncols - 1 and returns 0, or stops at the first
   call that returns a non-zero code and returns that code (the next columns are not read), as the Python loop did. */
static PyObject *readRow(PyObject *self, PyObject *args)
{
    unsigned long long function, vars;
    int id, row, ncols, col, code = 0;
    Py_ssize_t stride;
    GetValue get;
    char *base;

    (void)self;
    if (!PyArg_ParseTuple(args, "KiiiKn", &function, &id, &row, &ncols, &vars, &stride))
        return NULL;
    if (!function || !vars || stride <= 0 || ncols < 0) {
        PyErr_SetString(PyExc_ValueError, "_phreeqcRow.read : null function or buffer, or negative size");
        return NULL;
    }
    get = (GetValue)(uintptr_t)function;
    base = (char *)(uintptr_t)vars;
    /* the GIL is released, as ctypes releases it around each call of a foreign function */
    Py_BEGIN_ALLOW_THREADS
    for (col = 0; col < ncols; ++col) {
        code = get(id, row, col, base + (Py_ssize_t)col * stride);
        if (code != 0)
            break;
    }
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(code);
}

/* const char* GetWarningString(int id), int GetSelectedOutputRowCount(int id) / GetSelectedOutputColumnCount(int id) */
typedef const char *(*GetWarning)(int id);
typedef int (*GetCount)(int id);

/* after(warning, rows, cols, function, id, expected, vars, stride, typeOffset, valueOffset, dest[, columns, count])
       -> (warning text or None, nrows, ncols, status)
   What phreeqc.speciationPhreeqC asks the instance after the run of a node, in the same order, in one call:
   GetWarningString(id) (warning: its address, 0 without it: None), GetSelectedOutputRowCount(id), then, when there is a
   row, GetSelectedOutputColumnCount(id) (ncols 0 otherwise). Then, when nrows > 0 and ncols == expected (>= 0) and
   vars and dest are given, the last row is read as read() reads it (GetSelectedOutputValue column by column into the
   VAR structures at vars, `stride` bytes apart, stopping at a non-zero code) and, when every VAR holds a double (type
   TT_DOUBLE = 3, an int at typeOffset), the doubles (at valueOffset) are copied as they are into dest (ncols
   doubles): status 0. Otherwise status 1 (row not read), 2 (a value that is not a double) or 3 (a non-zero code):
   the caller then reads the row as before. The warning text is None when the string is empty.
   columns (address of `count` ints, 0: every column): only these columns of the last row are read, in this order,
   column columns[i] into the i-th VAR, its double into dest[i] (count doubles); status 1 when one of them is not a
   column of the row (0 <= columns[i] < ncols). */
static PyObject *afterRun(PyObject *self, PyObject *args)
{
    unsigned long long warningFn, rowsFn, colsFn, valueFn, vars, dest, columns = 0;
    int id, expected, nrows, ncols = 0, col, code = 0, status = 1;
    Py_ssize_t stride, typeOffset, valueOffset, count = -1, i, n;
    const char *warning = NULL;
    const int *picked;
    PyObject *text;

    (void)self;
    if (!PyArg_ParseTuple(args, "KKKKiiKnnnK|Kn", &warningFn, &rowsFn, &colsFn, &valueFn, &id, &expected, &vars,
                          &stride, &typeOffset, &valueOffset, &dest, &columns, &count))
        return NULL;
    if (columns && count < 0) {
        PyErr_SetString(PyExc_ValueError, "_phreeqcRow.after : columns given without their count");
        return NULL;
    }
    if (!rowsFn || !colsFn || !valueFn) {
        PyErr_SetString(PyExc_ValueError, "_phreeqcRow.after : null function");
        return NULL;
    }
    if (warningFn)
        warning = ((GetWarning)(uintptr_t)warningFn)(id);
    if (warning && warning[0]) {
        text = PyBytes_FromString(warning);
        if (!text)
            return NULL;
    } else {
        text = Py_None;
        Py_INCREF(text);
    }
    nrows = ((GetCount)(uintptr_t)rowsFn)(id);
    if (nrows)
        ncols = ((GetCount)(uintptr_t)colsFn)(id);
    picked = (const int *)(uintptr_t)columns;
    n = picked ? count : ncols;
    if (nrows > 0 && expected >= 0 && ncols == expected && vars && dest && stride > 0 && typeOffset >= 0
            && valueOffset >= 0) {
        char *base = (char *)(uintptr_t)vars;
        char *out = (char *)(uintptr_t)dest;
        GetValue get = (GetValue)(uintptr_t)valueFn;
        int type, inside = 1;
        if (picked)
            for (i = 0; i < n; ++i)
                if (picked[i] < 0 || picked[i] >= ncols) {
                    inside = 0;
                    break;
                }
        if (inside) {
            /* the GIL is released, as ctypes releases it around each call of a foreign function */
            Py_BEGIN_ALLOW_THREADS
            for (i = 0; i < n; ++i) {
                col = picked ? picked[i] : (int)i;
                code = get(id, nrows - 1, col, base + i * stride);
                if (code != 0)
                    break;
            }
            if (code != 0) {
                status = 3;
            } else {
                status = 0;
                for (i = 0; i < n; ++i) {
                    memcpy(&type, base + i * stride + typeOffset, sizeof(int));
                    if (type != 3) {
                        status = 2;
                        break;
                    }
                }
                if (status == 0)
                    for (i = 0; i < n; ++i)
                        memcpy(out + i * (Py_ssize_t)sizeof(double), base + i * stride + valueOffset, sizeof(double));
            }
            Py_END_ALLOW_THREADS
        }
    }
    return Py_BuildValue("(Niii)", text, nrows, ncols, status);
}

/* primary(values, positions, coeffs, out) -> touched (bytes, one per primary species: 1 when it has a nonzero
   coefficient)
   The sums of phreeqc.primaryColumns: for each primary species k and each node i, from out[k, i] (0.0, np.zeros), the
   products coeffs[j, k] * values[i, positions[j]] of the species j with a nonzero coefficient (c != 0.0: NaN counts, as
   for np.nonzero) are added one by one, j ascending: the operations of the Python loop out[k] += coeffs[j, k] * conc.
   Each product is rounded before it is added (stored in a volatile: never contracted into a fused multiply-add,
   whatever the compiler and its options), so the sums are the same doubles.
   values: 2-D float64 (nodes x columns, any strides); positions: 1-D int64 (column of each species); coeffs: 2-D
   float64 (species x primary species, any strides); out: 2-D float64, C-contiguous, writable (primary species x
   nodes). */
static int buffer(PyObject *obj, Py_buffer *view, int ndim, int writable, const char *what)
{
    int flags = PyBUF_STRIDES | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(obj, view, flags) < 0)
        return -1;
    if (view->ndim != ndim || view->itemsize != 8 || !view->format || view->format[1] != '\0') {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "_phreeqcRow.primary : %s has the wrong type or shape", what);
        return -1;
    }
    return 0;
}

static PyObject *primary(PyObject *self, PyObject *args)
{
    PyObject *valuesObj, *positionsObj, *coeffsObj, *outObj, *touched = NULL;
    Py_buffer values, positions, coeffs, out;
    Py_ssize_t nodes, nspecies, nprim, ncols, i, j, k;
    int bad = 0;
    char *flags;

    (void)self;
    if (!PyArg_ParseTuple(args, "OOOO", &valuesObj, &positionsObj, &coeffsObj, &outObj))
        return NULL;
    if (buffer(valuesObj, &values, 2, 0, "values") < 0)
        return NULL;
    if (buffer(positionsObj, &positions, 1, 0, "positions") < 0) {
        PyBuffer_Release(&values);
        return NULL;
    }
    if (buffer(coeffsObj, &coeffs, 2, 0, "coeffs") < 0) {
        PyBuffer_Release(&values);
        PyBuffer_Release(&positions);
        return NULL;
    }
    if (buffer(outObj, &out, 2, 1, "out") < 0) {
        PyBuffer_Release(&values);
        PyBuffer_Release(&positions);
        PyBuffer_Release(&coeffs);
        return NULL;
    }
    nodes = values.shape[0];
    ncols = values.shape[1];
    nspecies = coeffs.shape[0];
    nprim = coeffs.shape[1];
    if (values.format[0] != 'd' || coeffs.format[0] != 'd' || out.format[0] != 'd'
            || strchr("qlnQLN", positions.format[0]) == NULL || positions.shape[0] != nspecies
            || out.shape[0] != nprim || out.shape[1] != nodes || out.strides[1] != 8
            || out.strides[0] != nodes * 8)
        bad = 1;
    for (j = 0; j < nspecies && !bad; ++j) {
        int64_t p;
        memcpy(&p, (const char *)positions.buf + j * positions.strides[0], sizeof p);
        if (p < 0 || p >= ncols)
            bad = 1;
    }
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "_phreeqcRow.primary : arrays of inconsistent types or shapes");
        goto done;
    }
    touched = PyBytes_FromStringAndSize(NULL, nprim);
    if (!touched)
        goto done;
    flags = PyBytes_AS_STRING(touched);
    Py_BEGIN_ALLOW_THREADS
    for (k = 0; k < nprim; ++k) {
        double *sum = (double *)((char *)out.buf + k * out.strides[0]);
        flags[k] = 0;
        for (j = 0; j < nspecies; ++j) {
            double c;
            int64_t p;
            const char *column;
            memcpy(&c, (const char *)coeffs.buf + j * coeffs.strides[0] + k * coeffs.strides[1], sizeof c);
            if (!(c != 0.0))
                continue;
            flags[k] = 1;
            memcpy(&p, (const char *)positions.buf + j * positions.strides[0], sizeof p);
            column = (const char *)values.buf + p * values.strides[1];
            for (i = 0; i < nodes; ++i) {
                double v;
                volatile double product;
                memcpy(&v, column + i * values.strides[0], sizeof v);
                product = c * v;
                sum[i] = sum[i] + product;
            }
        }
    }
    Py_END_ALLOW_THREADS
done:
    PyBuffer_Release(&values);
    PyBuffer_Release(&positions);
    PyBuffer_Release(&coeffs);
    PyBuffer_Release(&out);
    return touched;
}

static PyMethodDef methods[] = {
    {"read", readRow, METH_VARARGS,
     "read(function, id, row, ncols, vars, stride) -> 0, or the first non-zero code of GetSelectedOutputValue"},
    {"primary", primary, METH_VARARGS,
     "primary(values, positions, coeffs, out) -> touched: the sums of phreeqc.primaryColumns, the same operations"},
    {"after", afterRun, METH_VARARGS,
     "after(warning, rows, cols, function, id, expected, vars, stride, typeOffset, valueOffset, dest[, columns, count])"
     " -> (warning text or None, nrows, ncols, status): what follows the run of a node, in one call"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_phreeqcRow",
    "One row of the IPhreeqc selected output in a single call (pycte.Source.phreeqc.selectedOutputRow).",
    -1, methods
};

/* VERSION 2: after() takes the columns to read (VERSION missing: an older build, without them) */
PyMODINIT_FUNC PyInit__phreeqcRow(void)
{
    PyObject *m = PyModule_Create(&module);
    if (m && PyModule_AddIntConstant(m, "VERSION", 2) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
