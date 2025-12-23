import { useEffect, useState } from "react";

export default function Go2rtcEmbed({
  url,
  className,
  title,
  onReady,
}: {
  url: string;
  className?: string;
  title?: string;
  onReady?: (ready: boolean) => void;
}) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!onReady) return;
    onReady(ready);
  }, [onReady, ready]);

  return (
    <div className={`go2rtc-embed ${className ?? ""} ${ready ? "is-ready" : ""}`}>
      <iframe
        src={url}
        title={title || "go2rtc stream"}
        className="go2rtc-embed-frame"
        allow="autoplay; fullscreen; picture-in-picture"
        referrerPolicy="no-referrer"
        onLoad={() => setReady(true)}
      />
    </div>
  );
}
