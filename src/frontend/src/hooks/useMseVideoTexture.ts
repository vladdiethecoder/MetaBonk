import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

const DEFAULT_MIME = 'video/mp4; codecs="avc1.42E01E"';

export function useMseVideoTexture(url: string | undefined, mime: string = DEFAULT_MIME) {
  const [texture, setTexture] = useState<THREE.VideoTexture | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (!url) {
      setTexture(null);
      return;
    }

    let disposed = false;
    let objectUrl: string | null = null;
    let tex: THREE.VideoTexture | null = null;

    const video = document.createElement("video");
    video.crossOrigin = "Anonymous";
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    videoRef.current = video;

    const setupStream = async () => {
      if (!window.MediaSource || !MediaSource.isTypeSupported(mime)) {
        video.src = url;
        return;
      }

      const ms = new MediaSource();
      objectUrl = URL.createObjectURL(ms);
      video.src = objectUrl;

      await new Promise<void>((resolve) => ms.addEventListener("sourceopen", () => resolve(), { once: true }));
      if (disposed || ms.readyState !== "open") return;

      try {
        const sb = ms.addSourceBuffer(mime);
        sb.mode = "segments";
        const queue: Uint8Array[] = [];
        let isSourceOpen = true;

        const processQueue = () => {
          if (sb.updating || queue.length === 0 || !isSourceOpen) return;
          try {
            sb.appendBuffer(queue.shift()!);
          } catch {
            queue.length = 0;
          }
        };

        sb.addEventListener("updateend", processQueue);
        ms.addEventListener("sourceclose", () => {
          isSourceOpen = false;
        });

        const response = await fetch(url, { cache: "no-store" });
        if (!response.body) {
          video.src = url;
          return;
        }
        const reader = response.body.getReader();
        while (!disposed) {
          const { value, done } = await reader.read();
          if (done || disposed) break;
          if (value) {
            queue.push(value);
            processQueue();
          }
        }
      } catch {
        video.src = url;
      }
    };

    (async () => {
      await setupStream();
      try {
        await video.play();
      } catch {
        // ignore autoplay restrictions
      }

      tex = new THREE.VideoTexture(video);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      if (!disposed) setTexture(tex);
    })();

    return () => {
      disposed = true;
      if (videoRef.current) {
        try {
          videoRef.current.pause();
          videoRef.current.src = "";
        } catch {}
        videoRef.current = null;
      }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (tex) tex.dispose();
    };
  }, [url, mime]);

  return texture;
}
