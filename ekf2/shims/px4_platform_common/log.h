// MSVC/Windows shim for <px4_platform_common/log.h> — no-op PX4 logging macros.
#pragma once
#define PX4_INFO(...) ((void)0)
#define PX4_INFO_RAW(...) ((void)0)
#define PX4_WARN(...) ((void)0)
#define PX4_ERR(...) ((void)0)
#define PX4_DEBUG(...) ((void)0)
#define PX4_PANIC(...) ((void)0)
#define PX4_LOG_NAMED(...) ((void)0)
#define PX4_LOG_NAMED_COND(...) ((void)0)
