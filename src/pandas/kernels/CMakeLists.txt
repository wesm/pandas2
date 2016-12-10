# This file is a part of pandas. See LICENSE for details about reuse and
# copyright holders

set(KERNELS_SRCS
  take.cc
)

add_library(pandas_kernels STATIC
  ${KERNELS_SRCS}
)
