#pragma once
// This file is a part of pandas. See LICENSE for details about reuse and
// copyright holders

#if defined(_WIN32) || defined(__CYGWIN__)
#define PANDAS_EXPORT __declspec(dllexport)
#else  // Not Windows
#ifndef PANDAS_EXPORT
#define PANDAS_EXPORT __attribute__((visibility("default")))
#endif
#ifndef PANDAS_NO_EXPORT
#define PANDAS_NO_EXPORT __attribute__((visibility("hidden")))
#endif
#endif  // Non-Windows
