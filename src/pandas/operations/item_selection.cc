// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// TODO: Temporary resting place

template <typename TYPE>
PyObject* FloatingArray<TYPE>::GetItem(int64_t i) {
  return NULL;
}

template <typename TYPE>
Status FloatingArray<TYPE>::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}

static Status PyObjectToInt64(PyObject* obj, int64_t* out) {
  PyObject* num = PyNumber_Long(obj);

  RETURN_IF_PYERROR();
  *out = PyLong_AsLongLong(num);
  Py_DECREF(num);
  return Status::OK();
}

template <typename TYPE>
PyObject* IntegerArray<TYPE>::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  return PyLong_FromLongLong(this->data()[i]);
}

template <typename TYPE>
Status IntegerArray<TYPE>::SetItem(int64_t i, PyObject* val) {
  if (!this->data_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Underlying buffer is immutable");
  }

  if (!valid_bits_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Valid bits buffer is immutable");
  }

  if (py::is_na(val)) {
    if (!valid_bits_) {
      // TODO: raise Python exception on error status
      RETURN_NOT_OK(AllocateValidityBitmap(this->length_, &valid_bits_));
    }
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    BitUtil::ClearBit(mutable_bits, i);
  } else {
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    if (valid_bits_) { BitUtil::SetBit(mutable_bits, i); }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    this->mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}

PyObject* BooleanArray::GetItem(int64_t i) {
  if (valid_bits_ && BitUtil::BitNotSet(valid_bits_->data(), i)) {
    Py_INCREF(py::NA);
    return py::NA;
  }
  if (data()[i] > 0) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

Status BooleanArray::SetItem(int64_t i, PyObject* val) {
  if (!data_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Underlying buffer is immutable");
  }

  if (!valid_bits_->is_mutable()) {
    // TODO(wesm): copy-on-write?
    return Status::Invalid("Valid bits buffer is immutable");
  }

  if (py::is_na(val)) {
    if (!valid_bits_) {
      // TODO: raise Python exception on error status
      RETURN_NOT_OK(AllocateValidityBitmap(length_, &valid_bits_));
    }
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    BitUtil::ClearBit(mutable_bits, i);
  } else {
    auto mutable_bits = static_cast<MutableBuffer*>(valid_bits_.get())->mutable_data();
    if (valid_bits_) { BitUtil::SetBit(mutable_bits, i); }
    int64_t cval;
    RETURN_NOT_OK(PyObjectToInt64(val, &cval));

    // Overflow issues
    mutable_data()[i] = cval;
  }
  RETURN_IF_PYERROR();
  return Status::OK();
}

PyObject* PyObjectArray::GetItem(int64_t i) {
  return nullptr;
}

Status PyObjectArray::SetItem(int64_t i, PyObject* val) {
  return Status::OK();
}
