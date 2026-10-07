#!/usr/bin/env bash
# Stream the app's log lines (latency stats every 2 s, camera/ARKit status) from the iPhone. Run inside WSL.
exec idevicesyslog --match WeeeVR "$@"
