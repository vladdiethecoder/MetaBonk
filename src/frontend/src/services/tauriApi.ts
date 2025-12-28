import { tauriInvoke } from "../lib/tauri_api";

export interface TrainingConfig {
  model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
}

export async function startTraining(config: TrainingConfig): Promise<string> {
  return tauriInvoke<string>("start_training", { config });
}

export async function getTrainingStatus(training_id: string): Promise<string> {
  return tauriInvoke<string>("get_training_status", { training_id });
}

export async function videoProcessingRunning(): Promise<boolean> {
  return tauriInvoke<boolean>("video_processing_running");
}

export async function processVideos(args: { video_dir: string; fps: number; resize: [number, number] }): Promise<string> {
  return tauriInvoke<string>("process_videos", args);
}
