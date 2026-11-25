/* pycte._floatText: float64 values written as Python writes them (repr(float), the text of f"{x}", str(), and of
   numpy's astype(str)), faster.

   repr(x) of a float is the shortest decimal string that reads back as x (the closest one, ties to even), laid out as
   'r' formatting does (fixed notation when the decimal point position is in (-4, 16], else d.ddde+XX, '.0' after an
   integer, 'inf', '-inf', 'nan'). CPython finds the digits with David Gay's dtoa (~0.5 us per value here); the digits
   are found here with Ryu (U. Adams, "Ryu: fast float-to-string conversion", PLDI 2018), which gives the same digits
   (shortest, closest, ties to even), then laid out the same way. Same text, character for character: checked against
   repr() and numpy on random bit patterns of every exponent and on the edge cases (tests/test_float_text.py).

   texts(values) -> list of str          (1-D float64 buffer)
   rows(values, sep, newline) -> str     (2-D float64 buffer: each row joined by sep and ended by newline) */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "floatText_tables.h"

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#define FT_MSVC_X64 1
#endif

/* ---- Ryu d2s (shortest digits), after the reference implementation of U. Adams (Apache 2.0 / Boost 1.0) ---- */

#define DOUBLE_MANTISSA_BITS 52
#define DOUBLE_EXPONENT_BITS 11
#define DOUBLE_BIAS 1023

/* high 64 bits and low 64 bits of a * b */
static inline uint64_t umul128(uint64_t a, uint64_t b, uint64_t *high)
{
#if defined(FT_MSVC_X64)
    return _umul128(a, b, high);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 p = (unsigned __int128)a * b;
    *high = (uint64_t)(p >> 64);
    return (uint64_t)p;
#else
    uint32_t aLo = (uint32_t)a, aHi = (uint32_t)(a >> 32), bLo = (uint32_t)b, bHi = (uint32_t)(b >> 32);
    uint64_t b00 = (uint64_t)aLo * bLo, b01 = (uint64_t)aLo * bHi, b10 = (uint64_t)aHi * bLo,
             b11 = (uint64_t)aHi * bHi;
    uint32_t b00Lo = (uint32_t)b00, b00Hi = (uint32_t)(b00 >> 32);
    uint64_t mid1 = b10 + b00Hi;
    uint32_t mid1Lo = (uint32_t)mid1, mid1Hi = (uint32_t)(mid1 >> 32);
    uint64_t mid2 = b01 + mid1Lo;
    uint32_t mid2Lo = (uint32_t)mid2, mid2Hi = (uint32_t)(mid2 >> 32);
    *high = b11 + mid1Hi + mid2Hi;
    return ((uint64_t)mid2Lo << 32) | b00Lo;
#endif
}

/* (hi:lo) >> dist, 0 < dist < 64 */
static inline uint64_t shiftright128(uint64_t lo, uint64_t hi, uint32_t dist)
{
#if defined(FT_MSVC_X64)
    return __shiftright128(lo, hi, (unsigned char)dist);
#else
    return (hi << (64 - dist)) | (lo >> dist);
#endif
}

static inline uint32_t pow5bits(int32_t e) { return (uint32_t)(((uint32_t)e * 1217359) >> 19) + 1; }
static inline uint32_t log10Pow2(int32_t e) { return ((uint32_t)e * 78913) >> 18; }
static inline uint32_t log10Pow5(int32_t e) { return ((uint32_t)e * 732923) >> 20; }

static inline uint32_t pow5Factor(uint64_t value)
{
    uint32_t count = 0;
    for (;;) {
        uint64_t q = value / 5;
        uint32_t r = (uint32_t)(value - 5 * q);
        if (r != 0)
            break;
        value = q;
        ++count;
    }
    return count;
}

static inline int multipleOfPowerOf5(uint64_t value, uint32_t p) { return pow5Factor(value) >= p; }
static inline int multipleOfPowerOf2(uint64_t value, uint32_t p) { return (value & ((1ull << p) - 1)) == 0; }

static inline uint64_t mulShift64(uint64_t m, const uint64_t *mul, int32_t j)
{
    uint64_t high1, high0, sum;
    uint64_t low1 = umul128(m, mul[1], &high1);
    umul128(m, mul[0], &high0);
    sum = high0 + low1;
    if (sum < high0)
        ++high1;
    return shiftright128(sum, high1, (uint32_t)(j - 64));
}

static inline uint64_t mulShiftAll64(uint64_t m, const uint64_t *mul, int32_t j, uint64_t *vp, uint64_t *vm,
                                     uint32_t mmShift)
{
    *vp = mulShift64(4 * m + 2, mul, j);
    *vm = mulShift64(4 * m - 1 - mmShift, mul, j);
    return mulShift64(4 * m, mul, j);
}

/* shortest decimal m * 10^e of the finite, non-zero double (ieeeMantissa, ieeeExponent) */
static void d2d(uint64_t ieeeMantissa, uint32_t ieeeExponent, uint64_t *mantissa, int32_t *exponent)
{
    int32_t e2, e10, removed = 0;
    uint64_t m2, mv, vr, vp, vm, output;
    uint32_t mmShift;
    int even, acceptBounds, vmIsTrailingZeros = 0, vrIsTrailingZeros = 0;
    uint8_t lastRemovedDigit = 0;

    if (ieeeExponent == 0) {
        e2 = 1 - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
        m2 = ieeeMantissa;
    } else {
        e2 = (int32_t)ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS - 2;
        m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
    }
    even = (m2 & 1) == 0;
    acceptBounds = even;
    mv = 4 * m2;
    mmShift = ieeeMantissa != 0 || ieeeExponent <= 1;

    if (e2 >= 0) {
        uint32_t q = log10Pow2(e2) - (e2 > 3);
        int32_t k = POW5_INV_BITCOUNT + (int32_t)pow5bits((int32_t)q) - 1;
        int32_t i = -e2 + (int32_t)q + k;
        e10 = (int32_t)q;
        vr = mulShiftAll64(m2, POW5_INV_SPLIT[q], i, &vp, &vm, mmShift);
        if (q <= 21) {
            uint32_t mvMod5 = (uint32_t)(mv - 5 * (mv / 5));
            if (mvMod5 == 0)
                vrIsTrailingZeros = multipleOfPowerOf5(mv, q);
            else if (acceptBounds)
                vmIsTrailingZeros = multipleOfPowerOf5(mv - 1 - mmShift, q);
            else
                vp -= multipleOfPowerOf5(mv + 2, q);
        }
    } else {
        uint32_t q = log10Pow5(-e2) - (-e2 > 1);
        int32_t i = -e2 - (int32_t)q;
        int32_t k = (int32_t)pow5bits(i) - POW5_BITCOUNT;
        int32_t j = (int32_t)q - k;
        e10 = (int32_t)q + e2;
        vr = mulShiftAll64(m2, POW5_SPLIT[i], j, &vp, &vm, mmShift);
        if (q <= 1) {
            vrIsTrailingZeros = 1;
            if (acceptBounds)
                vmIsTrailingZeros = mmShift == 1;
            else
                --vp;
        } else if (q < 63) {
            vrIsTrailingZeros = multipleOfPowerOf2(mv, q);
        }
    }

    if (vmIsTrailingZeros || vrIsTrailingZeros) {
        for (;;) {
            uint64_t vpDiv10 = vp / 10, vmDiv10 = vm / 10, vrDiv10;
            uint32_t vmMod10, vrMod10;
            if (vpDiv10 <= vmDiv10)
                break;
            vmMod10 = (uint32_t)(vm - 10 * vmDiv10);
            vrDiv10 = vr / 10;
            vrMod10 = (uint32_t)(vr - 10 * vrDiv10);
            vmIsTrailingZeros &= vmMod10 == 0;
            vrIsTrailingZeros &= lastRemovedDigit == 0;
            lastRemovedDigit = (uint8_t)vrMod10;
            vr = vrDiv10;
            vp = vpDiv10;
            vm = vmDiv10;
            ++removed;
        }
        if (vmIsTrailingZeros) {
            for (;;) {
                uint64_t vmDiv10 = vm / 10, vpDiv10, vrDiv10;
                uint32_t vmMod10 = (uint32_t)(vm - 10 * vmDiv10), vrMod10;
                if (vmMod10 != 0)
                    break;
                vpDiv10 = vp / 10;
                vrDiv10 = vr / 10;
                vrMod10 = (uint32_t)(vr - 10 * vrDiv10);
                vrIsTrailingZeros &= lastRemovedDigit == 0;
                lastRemovedDigit = (uint8_t)vrMod10;
                vr = vrDiv10;
                vp = vpDiv10;
                vm = vmDiv10;
                ++removed;
            }
        }
        if (vrIsTrailingZeros && lastRemovedDigit == 5 && vr % 2 == 0)
            lastRemovedDigit = 4;           /* round to even when the exact value is ...50..0 */
        output = vr + ((vr == vm && (!acceptBounds || !vmIsTrailingZeros)) || lastRemovedDigit >= 5);
    } else {
        int roundUp = 0;
        uint64_t vpDiv100 = vp / 100, vmDiv100 = vm / 100;
        if (vpDiv100 > vmDiv100) {
            uint64_t vrDiv100 = vr / 100;
            uint32_t vrMod100 = (uint32_t)(vr - 100 * vrDiv100);
            roundUp = vrMod100 >= 50;
            vr = vrDiv100;
            vp = vpDiv100;
            vm = vmDiv100;
            removed += 2;
        }
        for (;;) {
            uint64_t vpDiv10 = vp / 10, vmDiv10 = vm / 10, vrDiv10;
            uint32_t vrMod10;
            if (vpDiv10 <= vmDiv10)
                break;
            vrDiv10 = vr / 10;
            vrMod10 = (uint32_t)(vr - 10 * vrDiv10);
            roundUp = vrMod10 >= 5;
            vr = vrDiv10;
            vp = vpDiv10;
            vm = vmDiv10;
            ++removed;
        }
        output = vr + (vr == vm || roundUp);
    }
    *mantissa = output;
    *exponent = e10 + removed;
}

/* integers in [1, 2^53): their own digits */
static int smallInt(uint64_t ieeeMantissa, uint32_t ieeeExponent, uint64_t *mantissa, int32_t *exponent)
{
    uint64_t m2 = (1ull << DOUBLE_MANTISSA_BITS) | ieeeMantissa;
    int32_t e2 = (int32_t)ieeeExponent - DOUBLE_BIAS - DOUBLE_MANTISSA_BITS;
    uint64_t mask;
    if (e2 > 0 || e2 < -52)
        return 0;
    mask = (1ull << -e2) - 1;
    if (m2 & mask)
        return 0;
    *mantissa = m2 >> -e2;
    *exponent = 0;
    return 1;
}

/* ---- layout of repr(float): CPython format_float_short, mode 'r', Py_DTSF_ADD_DOT_0 ---- */

#define FT_MAX_LENGTH 32        /* longest text: 24 characters (-1.2345678901234567e-308) */

static int formatDouble(double value, char *out)
{
    uint64_t bits, ieeeMantissa, mantissa;
    uint32_t ieeeExponent;
    int32_t exponent, decpt, n = 0, i;
    char digits[24], *p = out;

    memcpy(&bits, &value, sizeof bits);
    ieeeMantissa = bits & ((1ull << DOUBLE_MANTISSA_BITS) - 1);
    ieeeExponent = (uint32_t)((bits >> DOUBLE_MANTISSA_BITS) & ((1u << DOUBLE_EXPONENT_BITS) - 1));
    if (ieeeExponent == (1u << DOUBLE_EXPONENT_BITS) - 1) {
        if (ieeeMantissa) {
            memcpy(p, "nan", 3);
            return 3;
        }
        if (bits >> 63)
            *p++ = '-';
        memcpy(p, "inf", 3);
        return (int)(p - out) + 3;
    }
    if (bits >> 63)
        *p++ = '-';
    if (ieeeExponent == 0 && ieeeMantissa == 0) {
        memcpy(p, "0.0", 3);
        return (int)(p - out) + 3;
    }
    if (!smallInt(ieeeMantissa, ieeeExponent, &mantissa, &exponent))
        d2d(ieeeMantissa, ieeeExponent, &mantissa, &exponent);
    while (mantissa % 10 == 0) {        /* digits without trailing zeros, as dtoa gives them */
        mantissa /= 10;
        ++exponent;
    }
    {
        char reversed[24];
        while (mantissa) {
            reversed[n++] = (char)('0' + mantissa % 10);
            mantissa /= 10;
        }
        for (i = 0; i < n; ++i)
            digits[i] = reversed[n - 1 - i];
    }
    decpt = n + exponent;               /* value = 0.d1d2...dn * 10^decpt */
    if (decpt <= -4 || decpt > 16) {
        int32_t e = decpt - 1;
        uint32_t a;
        *p++ = digits[0];
        if (n > 1) {
            *p++ = '.';
            memcpy(p, digits + 1, (size_t)(n - 1));
            p += n - 1;
        }
        *p++ = 'e';
        *p++ = e < 0 ? '-' : '+';
        a = (uint32_t)(e < 0 ? -e : e);
        if (a >= 100) {
            *p++ = (char)('0' + a / 100);
            a %= 100;
        }
        *p++ = (char)('0' + a / 10);
        *p++ = (char)('0' + a % 10);
    } else if (decpt <= 0) {
        *p++ = '0';
        *p++ = '.';
        for (i = 0; i < -decpt; ++i)
            *p++ = '0';
        memcpy(p, digits, (size_t)n);
        p += n;
    } else if (decpt >= n) {
        memcpy(p, digits, (size_t)n);
        p += n;
        for (i = 0; i < decpt - n; ++i)
            *p++ = '0';
        *p++ = '.';
        *p++ = '0';
    } else {
        memcpy(p, digits, (size_t)decpt);
        p += decpt;
        *p++ = '.';
        memcpy(p, digits + decpt, (size_t)(n - decpt));
        p += n - decpt;
    }
    return (int)(p - out);
}

/* ---- Python functions ---- */

static int doubleBuffer(PyObject *obj, Py_buffer *view, int ndim)
{
    const char *f;
    if (PyObject_GetBuffer(obj, view, PyBUF_STRIDES | PyBUF_FORMAT) < 0)
        return -1;
    f = view->format ? view->format : "B";
    if (*f == '@' || *f == '=' || *f == '<')
        ++f;
    if (view->ndim != ndim || view->itemsize != 8 || strcmp(f, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "_floatText : a %d-D float64 array is expected", ndim);
        return -1;
    }
    return 0;
}

static PyObject *asciiText(const char *text, Py_ssize_t length)
{
    PyObject *s = PyUnicode_New(length, 127);
    if (s != NULL)
        memcpy(PyUnicode_DATA(s), text, (size_t)length);
    return s;
}

/* texts(values) -> [repr(float(v)) for v in values] */
static PyObject *texts(PyObject *self, PyObject *arg)
{
    Py_buffer view;
    Py_ssize_t n, i;
    PyObject *list;
    char buffer[FT_MAX_LENGTH];

    (void)self;
    if (doubleBuffer(arg, &view, 1) < 0)
        return NULL;
    n = view.shape[0];
    list = PyList_New(n);
    if (list != NULL) {
        for (i = 0; i < n; ++i) {
            double v;
            PyObject *s;
            memcpy(&v, (const char *)view.buf + i * view.strides[0], sizeof v);
            s = asciiText(buffer, formatDouble(v, buffer));
            if (s == NULL) {
                Py_CLEAR(list);
                break;
            }
            PyList_SET_ITEM(list, i, s);
        }
    }
    PyBuffer_Release(&view);
    return list;
}

/* rows(values, sep, newline) -> "".join(sep.join(map(repr, row)) + newline for row in values) */
static PyObject *rows(PyObject *self, PyObject *args)
{
    PyObject *values, *result = NULL;
    const char *sep, *newline;
    Py_ssize_t sepLength, newlineLength, nRows, nCols, size, r, c;
    Py_buffer view;
    char *text, *p;

    (void)self;
    if (!PyArg_ParseTuple(args, "Os#s#", &values, &sep, &sepLength, &newline, &newlineLength))
        return NULL;
    if (doubleBuffer(values, &view, 2) < 0)
        return NULL;
    nRows = view.shape[0];
    nCols = view.shape[1];
    if (nCols > 0 && nRows > (PY_SSIZE_T_MAX / 2) / (nCols * (FT_MAX_LENGTH + sepLength) + newlineLength + 1)) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    size = nRows * (nCols * (FT_MAX_LENGTH + sepLength) + newlineLength) + 1;
    text = (char *)PyMem_RawMalloc((size_t)size);
    if (text == NULL) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    p = text;
    /* no Python object is touched while the text is written: the other threads (pycte computing while its output
       thread writes the files) run meanwhile */
    Py_BEGIN_ALLOW_THREADS
    for (r = 0; r < nRows; ++r) {
        const char *row = (const char *)view.buf + r * view.strides[0];
        for (c = 0; c < nCols; ++c) {
            double v;
            if (c) {
                memcpy(p, sep, (size_t)sepLength);
                p += sepLength;
            }
            memcpy(&v, row + c * view.strides[1], sizeof v);
            p += formatDouble(v, p);
        }
        memcpy(p, newline, (size_t)newlineLength);
        p += newlineLength;
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    {
        /* sep and newline are ASCII in pycte ('\t', os.linesep); otherwise decoded as UTF-8 */
        int ascii = 1;
        Py_ssize_t k;
        for (k = 0; k < sepLength && ascii; ++k)
            ascii = (unsigned char)sep[k] < 128;
        for (k = 0; k < newlineLength && ascii; ++k)
            ascii = (unsigned char)newline[k] < 128;
        result = ascii ? asciiText(text, (Py_ssize_t)(p - text))
                       : PyUnicode_DecodeUTF8(text, (Py_ssize_t)(p - text), "strict");
    }
    PyMem_RawFree(text);
    return result;
}

static PyMethodDef methods[] = {
    {"texts", texts, METH_O, "texts(values) -> [repr(float(v)) for v in values] (1-D float64 array)"},
    {"rows", rows, METH_VARARGS,
     "rows(values, sep, newline) -> ''.join(sep.join(map(repr, row)) + newline for row in values) (2-D float64 "
     "array, the values as Python floats)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_floatText",
    "float64 values written as repr(float) writes them, faster (Ryu shortest digits, repr layout).",
    -1, methods
};

PyMODINIT_FUNC PyInit__floatText(void)
{
    return PyModule_Create(&module);
}
