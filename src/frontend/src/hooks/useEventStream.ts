import { useEffect, useState } from "react";
import type { Event } from "../api";

export function useEventStream(limit = 200) {
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    const es = new EventSource(`/api/events/stream`);
    es.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data) as Event;
        setEvents((prev) => {
          const next = [...prev, ev];
          return next.length > limit ? next.slice(next.length - limit) : next;
        });
      } catch {
        // ignore
      }
    };
    es.onerror = () => {
      es.close();
    };
    return () => es.close();
  }, [limit]);

  return events;
}

