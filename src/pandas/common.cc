// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#include "pandas/common.h"

#include <exception>
#include <sstream>
#include <string>

namespace pandas {

void PandasException::NYI(const std::string& msg) {
  std::stringstream ss;
  ss << "Not yet implemented: " << msg << ".";
  throw PandasException(ss.str());
}

void PandasException::Throw(const std::string& msg) {
  throw PandasException(msg);
}

PandasException::PandasException(const char* msg) : msg_(msg) {}

PandasException::PandasException(const std::string& msg) : msg_(msg) {}

PandasException::PandasException(const char* msg, std::exception& e) : msg_(msg) {}

PandasException::~PandasException() throw() {}

const char* PandasException::what() const throw() {
  return msg_.c_str();
}

}  // namespace pandas
