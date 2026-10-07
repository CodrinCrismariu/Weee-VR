import os

/// App log lines, readable from the PC with `scripts/logs.sh` (idevicesyslog). NSLog/print text is
/// redacted as <private> in the system log for sideloaded apps, so everything is logged as public.
enum Log {
    private static let logger = Logger(subsystem: "com.weeesports.vr", category: "WeeeVR")

    static func info(_ message: String) {
        logger.notice("WeeeVR \(message, privacy: .public)")
    }
}
