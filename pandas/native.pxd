# Copyright 2014 Cloudera, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++

from libc.stdint cimport *
from libcpp cimport bool as c_bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

from cpython cimport PyObject

# This must be included for cerr and other things to work
cdef extern from "<iostream>":
    pass

cdef extern from "pandas/numpy_interop.h":
    pass

cdef extern from "numpy/arrayobject.h":
    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef:
            # Only taking a few of the most commonly used and stable fields.
            # One should use PyArray_* macros instead to access the C fields.
            # char *data
            # int ndim "nd"
            # npy_intp *shape "dimensions"
            # npy_intp *strides
            # # dtype descr
            PyObject* base


cdef extern from "pandas/catch_exception.h" namespace "pandas" nogil:
    void catch_exception()

cdef extern from "pandas/common.h" namespace "pandas" nogil:
    pass

cdef extern from "pandas/api.h" namespace "pandas":

    enum TypeId:
        TypeId_NA " pandas::TypeId::NA"
        TypeId_UINT8 " pandas::TypeId::UINT8"
        TypeId_UINT16 " pandas::TypeId::UINT16"
        TypeId_UINT32 " pandas::TypeId::UINT32"
        TypeId_UINT64 " pandas::TypeId::UINT64"
        TypeId_INT8 " pandas::TypeId::INT8"
        TypeId_INT16 " pandas::TypeId::INT16"
        TypeId_INT32 " pandas::TypeId::INT32"
        TypeId_INT64 " pandas::TypeId::INT64"
        TypeId_BOOL " pandas::TypeId::BOOL"
        TypeId_FLOAT32 " pandas::TypeId::FLOAT32"
        TypeId_FLOAT64 " pandas::TypeId::FLOAT64"
        TypeId_PYOBJECT " pandas::TypeId::PYOBJECT"
        TypeId_CATEGORY " pandas::TypeId::CATEGORY"
        TypeId_TIMESTAMP " pandas::TypeId::TIMESTAMP"
        TypeId_TIMESTAMP_TZ " pandas::TypeId::TIMESTAMP_TZ"

    cdef cppclass DataType:
        TypeId type()

        DataType()

        c_bool Equals(const DataType& other)
        string ToString()

    ctypedef shared_ptr[const DataType] TypePtr

    cdef cppclass Int8Type(DataType):
        pass

    cdef cppclass Int16Type(DataType):
        pass

    cdef cppclass Int32Type(DataType):
        pass

    cdef cppclass Int64Type(DataType):
        pass

    cdef cppclass UInt8Type(DataType):
        pass

    cdef cppclass UInt16Type(DataType):
        pass

    cdef cppclass UInt32Type(DataType):
        pass

    cdef cppclass UInt64Type(DataType):
        pass

    cdef cppclass FloatType(DataType):
        pass

    cdef cppclass DoubleType(DataType):
        pass

    cdef cppclass PyObjectType(DataType):
        pass

    cdef cppclass CategoryType(DataType):
        pass

    cdef cppclass CArray" pandas::Array":

        const TypePtr& type()
        TypeId type_id()
        size_t length()

        object GetItem(size_t i)
        void SetItem(size_t i, object val)

    cdef cppclass CCategoryArray" pandas::CategoryArray"(CArray):
        pass

    cdef cppclass CBooleanArray" pandas::BooleanArray"(CArray):
        pass

    ctypedef shared_ptr[CArray] ArrayPtr

    shared_ptr[DataType] primitive_type_from_enum(TypeId tp_enum) \
        except +catch_exception

    shared_ptr[CArray] CreateArrayFromNumPy(ndarray arr) \
        except +catch_exception
    shared_ptr[CArray] CreateArrayFromNumPy(ndarray arr) \
        except +catch_exception


cdef extern from "pandas/pytypes.h" namespace "pandas::py":
    void init_natype(object type_obj, object inst_obj)
    c_bool is_na(object type_obj)
