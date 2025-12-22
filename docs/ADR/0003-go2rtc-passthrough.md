# ADR 0003: go2rtc Raw H.264 Passthrough

## Status
Accepted

## Context
MetaBonk produces GPU-encoded H.264 streams per worker. We use go2rtc to
distribute those streams to WebRTC/MSE/RTSP without adding CPU load or latency.

## Decision
Use FIFO + raw H.264 passthrough (`#video=h264#raw`) as the default go2rtc source.

## Consequences
- No transcoding cost in go2rtc.
- Requires downstream clients to support H.264.
- Hardware acceleration only needed if go2rtc is configured to transcode.
