// MSVC/Windows shim for <px4_platform_common/defines.h>.
// Shadows the real PX4 header (which pulls in <sys/ioctl.h> / <px4_boardconfig.h>
// that don't exist here) — provides only the math macros + helpers the EKF uses.
#pragma once
#include <cmath>
#include <cstdint>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif
#ifndef M_PI_2_F
#define M_PI_2_F 1.57079632679489661923f
#endif
#ifndef M_PI_4_F
#define M_PI_4_F 0.78539816339744830962f
#endif
#ifndef M_TWOPI_F
#define M_TWOPI_F 6.28318530717958647692f
#endif
#ifndef M_DEG_TO_RAD
#define M_DEG_TO_RAD 0.017453292519943295
#endif
#ifndef M_DEG_TO_RAD_F
#define M_DEG_TO_RAD_F 0.01745329251994329577f
#endif
#ifndef M_RAD_TO_DEG
#define M_RAD_TO_DEG 57.295779513082320876
#endif
#ifndef M_RAD_TO_DEG_F
#define M_RAD_TO_DEG_F 57.29577951308232087721f
#endif
#ifndef M_SQRT2_F
#define M_SQRT2_F 1.41421356237309504880f
#endif

#ifndef PX4_OK
#define PX4_OK 0
#endif
#ifndef PX4_ERROR
#define PX4_ERROR (-1)
#endif
#ifndef PX4_ISFINITE
#define PX4_ISFINITE(x) std::isfinite(x)
#endif

#ifdef _MSC_VER
#include <cstddef>
#ifndef _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif
