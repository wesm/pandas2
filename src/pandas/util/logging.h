// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

// Contains some code derived from Apache Arrow

#pragma once

#include <cstdlib>
#include <iostream>

namespace pandas {

// Stubbed versions of macros defined in glog/logging.h, intended for
// environments where glog headers aren't available.
//
// Add more as needed.

// Log levels. LOG ignores them, so their values are abitrary.

#define PANDAS_INFO 0
#define PANDAS_WARNING 1
#define PANDAS_ERROR 2
#define PANDAS_FATAL 3

#define PANDAS_LOG_INTERNAL(level) ::pandas::internal::CerrLog(level)
#define PANDAS_LOG(level) PANDAS_LOG_INTERNAL(PANDAS_##level)

#define PANDAS_CHECK(condition)                               \
  (condition) ? 0 : ::pandas::internal::FatalLog(PANDAS_FATAL) \
                        << __FILE__ << __LINE__ << " Check failed: " #condition " "

#ifdef NDEBUG
#define PANDAS_DFATAL PANDAS_WARNING

#define PANDAS_DCHECK(condition)                \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_EQ(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_NE(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_LE(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_LT(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_GE(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()                 \

#define PANDAS_DCHECK_GT(val1, val2)            \
  while (false)                                 \
  ::pandas::internal::NullLog()

#else
#define PANDAS_DFATAL PANDAS_FATAL

#define PANDAS_DCHECK(condition) PANDAS_CHECK(condition)
#define PANDAS_DCHECK_EQ(val1, val2) PANDAS_CHECK((val1) == (val2))
#define PANDAS_DCHECK_NE(val1, val2) PANDAS_CHECK((val1) != (val2))
#define PANDAS_DCHECK_LE(val1, val2) PANDAS_CHECK((val1) <= (val2))
#define PANDAS_DCHECK_LT(val1, val2) PANDAS_CHECK((val1) < (val2))
#define PANDAS_DCHECK_GE(val1, val2) PANDAS_CHECK((val1) >= (val2))
#define PANDAS_DCHECK_GT(val1, val2) PANDAS_CHECK((val1) > (val2))

#endif  // NDEBUG

namespace internal {

class NullLog {
 public:
  template <class T>
  NullLog& operator<<(const T& t) {
    return *this;
  }
};

class CerrLog {
 public:
  CerrLog(int severity)  // NOLINT(runtime/explicit)
      : severity_(severity),
        has_logged_(false) {}

  virtual ~CerrLog() {
    if (has_logged_) { std::cerr << std::endl; }
    if (severity_ == PANDAS_FATAL) { std::exit(1); }
  }

  template <class T>
  CerrLog& operator<<(const T& t) {
    has_logged_ = true;
    std::cerr << t;
    return *this;
  }

 protected:
  const int severity_;
  bool has_logged_;
};

// Clang-tidy isn't smart enough to determine that PANDAS_DCHECK using CerrLog doesn't
// return so we create a new class to give it a hint.
class FatalLog : public CerrLog {
 public:
  explicit FatalLog(int /* severity */)  // NOLINT
      : CerrLog(PANDAS_FATAL) {}          // NOLINT

  [[noreturn]] ~FatalLog() {
    if (has_logged_) { std::cerr << std::endl; }
    std::exit(1);
  }
};

}  // namespace internal
}  // namespace pandas
