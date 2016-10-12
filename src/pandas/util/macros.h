// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#pragma once

// From Google gutil
#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete
#endif
