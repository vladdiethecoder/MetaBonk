import { useState } from "react";
import useTauriEvent from "./useTauriEvent";

export type TrainingProgressData = {
  epoch: number;
  loss: number;
  accuracy: number;
};

export default function useTrainingProgress() {
  const [progress, setProgress] = useState<TrainingProgressData | null>(null);
  useTauriEvent<TrainingProgressData>("training-progress", (payload) => setProgress(payload));
  return progress;
}
