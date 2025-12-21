FROM alexxit/go2rtc:latest

SHELL ["/bin/sh", "-c"]

RUN set -eux; \
  if command -v apk >/dev/null 2>&1; then \
    apk add --no-cache python3 ffmpeg bash; \
  elif command -v apt-get >/dev/null 2>&1; then \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 ffmpeg bash ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
  else \
    echo "no supported package manager found" >&2; \
    exit 1; \
  fi

WORKDIR /app
