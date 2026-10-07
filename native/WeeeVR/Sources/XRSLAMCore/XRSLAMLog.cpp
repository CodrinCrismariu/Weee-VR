// Replaces XRSLAM's spdlog-based utility/debug.cpp: messages go to the unified log, readable with
// scripts/logs.sh (idevicesyslog --match WeeeVR). Debug-level messages are dropped.
#include <os/log.h>
#include <cstdarg>
#include <cstdio>
#include <xrslam/common.h>

namespace xrslam {

void log_message(LogLevel level, const char *format, ...) {
    if (level < XRSLAM_LOG_INFO) return;
    thread_local char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    static os_log_t log = os_log_create("com.weeesports.vr", "XRSLAM");
    os_log_with_type(log, level >= XRSLAM_LOG_ERR ? OS_LOG_TYPE_ERROR : OS_LOG_TYPE_DEFAULT,
                     "WeeeVR xrslam: %{public}s", buffer);
}

} // namespace xrslam
