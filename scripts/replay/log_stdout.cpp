// Linux stand-in for Sources/XRSLAMCore/XRSLAMLog.cpp (os_log): XRSLAM messages to stderr.
#include <cstdarg>
#include <cstdio>
#include <xrslam/common.h>

namespace xrslam {

void log_message(LogLevel level, const char *format, ...) {
    if (level < XRSLAM_LOG_INFO) return;
    va_list args;
    va_start(args, format);
    std::fputs("xrslam: ", stderr);
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    va_end(args);
}

} // namespace xrslam
