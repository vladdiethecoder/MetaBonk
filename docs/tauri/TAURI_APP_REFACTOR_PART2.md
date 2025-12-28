# MetaBonk Tauri App Refactor - Part 2
## Analytics, Settings, State Management & Backend

**Continued from Part 1**

---

## 📊 **4. Analytics Tab - Performance & Metrics**

**Consolidates**: Old "Metrics", "Stats", "Performance" tabs into one comprehensive analytics view.

```tsx
// src/pages/AnalyticsPage.tsx

import { useState } from 'react';
import {
  TrainingCurves,
  SystemHealth,
  ComparisonView,
  ExportData,
} from '@/components/analytics';

export function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('1h');
  const [selectedMetrics, setSelectedMetrics] = useState(['reward', 'loss']);

  return (
    <div className="p-6 space-y-6">
      {/* Header with Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Analytics</h2>
          <p className="text-secondary">Training performance and system metrics</p>
        </div>

        <div className="flex items-center gap-3">
          {/* Time Range Selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-bg-secondary border border-glass-border rounded-lg px-3 py-2"
          >
            <option value="5m">Last 5 minutes</option>
            <option value="15m">Last 15 minutes</option>
            <option value="1h">Last hour</option>
            <option value="6h">Last 6 hours</option>
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="all">All time</option>
          </select>

          {/* Export Button */}
          <Button onClick={() => exportData()}>
            <DownloadIcon className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics Summary */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Avg Reward"
          value={metrics.avgReward}
          change={metrics.rewardChange}
          trend="up"
        />
        <MetricCard
          label="Success Rate"
          value={`${(metrics.successRate * 100).toFixed(1)}%`}
          change={metrics.successRateChange}
          trend="up"
        />
        <MetricCard
          label="Training FPS"
          value={metrics.fps}
          change={metrics.fpsChange}
          trend="stable"
        />
        <MetricCard
          label="GPU Util"
          value={`${metrics.gpuUtil}%`}
          change={metrics.gpuUtilChange}
          trend="stable"
        />
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Training Curves */}
        <Card className="col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Training Curves</h3>
            <MetricSelector
              selected={selectedMetrics}
              onChange={setSelectedMetrics}
            />
          </div>
          <TrainingCurves
            metrics={selectedMetrics}
            timeRange={timeRange}
          />
        </Card>

        {/* Reward Distribution */}
        <Card>
          <h3 className="text-lg font-semibold mb-4">Reward Distribution</h3>
          <RewardHistogram data={metrics.rewardDistribution} />
        </Card>

        {/* Action Distribution */}
        <Card>
          <h3 className="text-lg font-semibold mb-4">Action Distribution</h3>
          <ActionDistributionChart data={metrics.actionDistribution} />
        </Card>
      </div>

      {/* System Health */}
      <SystemHealth timeRange={timeRange} />

      {/* Agent Comparison (if multiple agents) */}
      {agents.length > 1 && (
        <ComparisonView agents={agents} timeRange={timeRange} />
      )}
    </div>
  );
}
```

### **Metric Card Component**

```tsx
// src/components/analytics/MetricCard.tsx

export function MetricCard({ label, value, change, trend }) {
  const trendIcon = {
    up: '↗',
    down: '↘',
    stable: '→',
  }[trend];

  const trendColor = {
    up: 'text-success',
    down: 'text-error',
    stable: 'text-secondary',
  }[trend];

  return (
    <Card>
      <div className="space-y-2">
        <p className="text-sm text-secondary">{label}</p>
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold">{value}</span>
          {change !== undefined && (
            <span className={cn('text-sm flex items-center gap-1', trendColor)}>
              <span>{trendIcon}</span>
              <span>{Math.abs(change).toFixed(1)}%</span>
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}
```

### **Training Curves Component**

```tsx
// src/components/analytics/TrainingCurves.tsx

import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export function TrainingCurves({ metrics, timeRange }) {
  const data = useMetricsData(metrics, timeRange);

  const chartData = {
    labels: data.timestamps,
    datasets: metrics.map((metric, i) => ({
      label: metric.toUpperCase(),
      data: data[metric],
      borderColor: METRIC_COLORS[metric],
      backgroundColor: `${METRIC_COLORS[metric]}20`,
      fill: true,
      tension: 0.4,
    })),
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: '#8b9dc3',
          usePointStyle: true,
        },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: '#12192e',
        borderColor: '#1a2542',
        borderWidth: 1,
        titleColor: '#ffffff',
        bodyColor: '#8b9dc3',
      },
    },
    scales: {
      x: {
        grid: {
          color: '#1a2542',
        },
        ticks: {
          color: '#4a5f7f',
        },
      },
      y: {
        grid: {
          color: '#1a2542',
        },
        ticks: {
          color: '#4a5f7f',
        },
      },
    },
  };

  return (
    <div className="h-80">
      <Line data={chartData} options={options} />
    </div>
  );
}

const METRIC_COLORS = {
  reward: '#00ff9f',
  loss: '#ff0080',
  value_loss: '#ff4757',
  policy_loss: '#ffb900',
  entropy: '#00d4ff',
  fps: '#00ff9f',
};
```

### **System Health Component**

```tsx
// src/components/analytics/SystemHealth.tsx

export function SystemHealth({ timeRange }) {
  const health = useSystemHealth(timeRange);

  return (
    <Card>
      <h3 className="text-lg font-semibold mb-4">System Health</h3>

      <div className="grid grid-cols-3 gap-6">
        {/* GPU */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-secondary">GPU</h4>
          <div className="space-y-2">
            <HealthMetric
              label="Utilization"
              value={health.gpu.utilization}
              unit="%"
              threshold={85}
            />
            <HealthMetric
              label="Temperature"
              value={health.gpu.temperature}
              unit="°C"
              threshold={80}
            />
            <HealthMetric
              label="VRAM Used"
              value={health.gpu.vramUsed}
              max={health.gpu.vramTotal}
              unit="GB"
            />
            <HealthMetric
              label="Power Draw"
              value={health.gpu.powerDraw}
              unit="W"
              threshold={300}
            />
          </div>
        </div>

        {/* CPU */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-secondary">CPU</h4>
          <div className="space-y-2">
            <HealthMetric
              label="Utilization"
              value={health.cpu.utilization}
              unit="%"
              threshold={90}
            />
            <HealthMetric
              label="Temperature"
              value={health.cpu.temperature}
              unit="°C"
              threshold={85}
            />
            <HealthMetric
              label="RAM Used"
              value={health.cpu.ramUsed}
              max={health.cpu.ramTotal}
              unit="GB"
            />
          </div>
        </div>

        {/* Synthetic Eye */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-secondary">Synthetic Eye</h4>
          <div className="space-y-2">
            <HealthMetric
              label="Frame Rate"
              value={health.syntheticEye.fps}
              unit="FPS"
              threshold={400}
              inverted
            />
            <HealthMetric
              label="Latency"
              value={health.syntheticEye.latency}
              unit="ms"
              threshold={2}
            />
            <HealthMetric
              label="PCIe BW"
              value={health.syntheticEye.pcieBandwidth}
              unit="GB/s"
              threshold={1}
            />
            <div className="flex items-center justify-between text-sm">
              <span className="text-secondary">Status:</span>
              <StatusBadge status={health.syntheticEye.status} size="sm" />
            </div>
          </div>
        </div>
      </div>

      {/* Health Score */}
      <Card className="mt-4 bg-success/5 border-success/20">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold text-success">System Health Score</h4>
            <p className="text-xs text-secondary mt-1">
              All systems operational • No bottlenecks detected
            </p>
          </div>
          <div className="text-4xl font-bold text-success">
            {health.overallScore}/100
          </div>
        </div>
      </Card>
    </Card>
  );
}

function HealthMetric({ label, value, unit, threshold, max, inverted = false }) {
  const percentage = max ? (value / max) * 100 : value;
  const isWarning = inverted
    ? threshold && value < threshold
    : threshold && value > threshold;

  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-secondary">{label}:</span>
      <div className="flex items-center gap-2">
        <span className={cn('font-mono', isWarning && 'text-warning')}>
          {value.toFixed(1)} {unit}
        </span>
        {isWarning && <span className="text-warning">⚠</span>}
      </div>
    </div>
  );
}
```

---

## ⚙️ **5. Settings Tab - Configuration & Advanced**

```tsx
// src/pages/SettingsPage.tsx

export function SettingsPage() {
  const [activeSection, setActiveSection] = useState('general');

  const sections = [
    { id: 'general', label: 'General', icon: '⚙️' },
    { id: 'synthetic-eye', label: 'Synthetic Eye', icon: '🔬' },
    { id: 'discovery', label: 'Discovery', icon: '🔍' },
    { id: 'training', label: 'Training', icon: '🎓' },
    { id: 'advanced', label: 'Advanced', icon: '🛠️' },
  ];

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-64 border-r border-glass-border p-4">
        <h2 className="text-xl font-bold mb-4">Settings</h2>
        <nav className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all',
                activeSection === section.id
                  ? 'bg-primary/10 text-primary'
                  : 'hover:bg-bg-secondary text-secondary'
              )}
            >
              <span className="text-xl">{section.icon}</span>
              <span>{section.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 overflow-auto">
        {activeSection === 'general' && <GeneralSettings />}
        {activeSection === 'synthetic-eye' && <SyntheticEyeSettings />}
        {activeSection === 'discovery' && <DiscoverySettings />}
        {activeSection === 'training' && <TrainingSettings />}
        {activeSection === 'advanced' && <AdvancedSettings />}
      </div>
    </div>
  );
}
```

### **Synthetic Eye Settings**

```tsx
// src/components/settings/SyntheticEyeSettings.tsx

export function SyntheticEyeSettings() {
  const { config, updateConfig } = useSyntheticEyeConfig();

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h3 className="text-2xl font-bold mb-2">Synthetic Eye Configuration</h3>
        <p className="text-secondary">
          Configure zero-copy observation pipeline
        </p>
      </div>

      {/* Status */}
      <Card className="bg-info/5 border-info/20">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold mb-1">Synthetic Eye Status</h4>
            <p className="text-sm text-secondary">
              Compositor running at {config.fps} FPS
            </p>
          </div>
          <StatusBadge status={config.status} />
        </div>
      </Card>

      {/* Resolution */}
      <SettingSection
        title="Resolution"
        description="Observation resolution (higher = more detail, lower = faster)"
      >
        <select
          value={config.resolution}
          onChange={(e) => updateConfig({ resolution: e.target.value })}
          className="input"
        >
          <option value="1280x720">1280x720 (HD)</option>
          <option value="1920x1080">1920x1080 (Full HD)</option>
          <option value="2560x1440">2560x1440 (2K)</option>
          <option value="3840x2160">3840x2160 (4K)</option>
        </select>
      </SettingSection>

      {/* Lockstep Mode */}
      <SettingSection
        title="Lockstep Mode"
        description="Agent controls frame timing for deterministic training"
      >
        <Toggle
          enabled={config.lockstep}
          onChange={(enabled) => updateConfig({ lockstep: enabled })}
        />
      </SettingSection>

      {/* DMA-BUF Buffer Count */}
      <SettingSection
        title="Buffer Pool Size"
        description="Number of frame buffers for triple buffering"
      >
        <input
          type="number"
          min={2}
          max={5}
          value={config.bufferCount}
          onChange={(e) => updateConfig({ bufferCount: parseInt(e.target.value) })}
          className="input w-24"
        />
        <p className="text-xs text-muted mt-1">
          Default: 3 (triple buffering)
        </p>
      </SettingSection>

      {/* Socket Path */}
      <SettingSection
        title="Socket Path"
        description="Unix domain socket for IPC"
      >
        <input
          type="text"
          value={config.socketPath}
          onChange={(e) => updateConfig({ socketPath: e.target.value })}
          className="input"
        />
      </SettingSection>

      {/* Advanced Options */}
      <details className="group">
        <summary className="cursor-pointer font-semibold text-secondary hover:text-primary">
          Advanced Options ▼
        </summary>
        <div className="mt-4 space-y-4 pl-4 border-l-2 border-glass-border">
          <SettingSection
            title="GPU Device"
            description="DRM device path"
          >
            <input
              type="text"
              value={config.drmDevice}
              className="input font-mono"
              readOnly
            />
          </SettingSection>

          <SettingSection
            title="Pixel Format"
            description="GPU buffer format"
          >
            <select value={config.format} className="input" disabled>
              <option>ARGB8888</option>
            </select>
          </SettingSection>
        </div>
      </details>

      {/* Actions */}
      <div className="flex gap-3 pt-6 border-t border-glass-border">
        <Button variant="primary" onClick={() => restartSyntheticEye()}>
          Apply & Restart
        </Button>
        <Button variant="outline" onClick={() => testConnection()}>
          Test Connection
        </Button>
      </div>
    </div>
  );
}

function SettingSection({ title, description, children }) {
  return (
    <Card>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <h4 className="font-semibold mb-1">{title}</h4>
          <p className="text-sm text-secondary">{description}</p>
        </div>
        <div className="flex-shrink-0">{children}</div>
      </div>
    </Card>
  );
}
```

### **Training Hyperparameters**

```tsx
// src/components/settings/TrainingSettings.tsx

export function TrainingSettings() {
  const { hyperparams, updateHyperparams } = useTrainingConfig();

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h3 className="text-2xl font-bold mb-2">Training Configuration</h3>
        <p className="text-secondary">
          PPO hyperparameters and training settings
        </p>
      </div>

      {/* Preset Selector */}
      <Card className="bg-primary/5 border-primary/20">
        <h4 className="font-semibold mb-3">Presets</h4>
        <div className="grid grid-cols-3 gap-2">
          <PresetButton preset="conservative" />
          <PresetButton preset="balanced" />
          <PresetButton preset="aggressive" />
        </div>
      </Card>

      {/* PPO Settings */}
      <div className="space-y-4">
        <h4 className="font-semibold">PPO Hyperparameters</h4>

        <HyperparamInput
          label="Learning Rate"
          value={hyperparams.learningRate}
          onChange={(v) => updateHyperparams({ learningRate: v })}
          min={1e-6}
          max={1e-2}
          step={1e-6}
          format="scientific"
        />

        <HyperparamInput
          label="Gamma (Discount)"
          value={hyperparams.gamma}
          onChange={(v) => updateHyperparams({ gamma: v })}
          min={0.9}
          max={0.999}
          step={0.001}
        />

        <HyperparamInput
          label="GAE Lambda"
          value={hyperparams.gaeLambda}
          onChange={(v) => updateHyperparams({ gaeLambda: v })}
          min={0.9}
          max={0.99}
          step={0.01}
        />

        <HyperparamInput
          label="Clip Range"
          value={hyperparams.clipRange}
          onChange={(v) => updateHyperparams({ clipRange: v })}
          min={0.1}
          max={0.3}
          step={0.05}
        />

        <HyperparamInput
          label="Entropy Coefficient"
          value={hyperparams.entropyCoef}
          onChange={(v) => updateHyperparams({ entropyCoef: v })}
          min={0.0}
          max={0.1}
          step={0.001}
          format="scientific"
        />
      </div>

      {/* Training Loop */}
      <div className="space-y-4">
        <h4 className="font-semibold">Training Loop</h4>

        <HyperparamInput
          label="Batch Size"
          value={hyperparams.batchSize}
          onChange={(v) => updateHyperparams({ batchSize: v })}
          min={32}
          max={2048}
          step={32}
          format="integer"
        />

        <HyperparamInput
          label="Mini-batch Size"
          value={hyperparams.miniBatchSize}
          onChange={(v) => updateHyperparams({ miniBatchSize: v })}
          min={16}
          max={512}
          step={16}
          format="integer"
        />

        <HyperparamInput
          label="Epochs per Update"
          value={hyperparams.epochs}
          onChange={(v) => updateHyperparams({ epochs: v })}
          min={1}
          max={20}
          step={1}
          format="integer"
        />
      </div>

      {/* Save/Reset */}
      <div className="flex gap-3 pt-6 border-t border-glass-border">
        <Button variant="primary" onClick={() => saveHyperparams()}>
          Save Changes
        </Button>
        <Button variant="outline" onClick={() => resetToDefaults()}>
          Reset to Defaults
        </Button>
      </div>
    </div>
  );
}

function HyperparamInput({ label, value, onChange, min, max, step, format = 'float' }) {
  const displayValue = format === 'scientific'
    ? value.toExponential(2)
    : format === 'integer'
    ? value.toString()
    : value.toFixed(3);

  return (
    <Card>
      <div className="flex items-center justify-between gap-4">
        <label className="font-medium">{label}</label>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="slider"
          />
          <input
            type="number"
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            step={step}
            className="input w-32 font-mono text-sm"
          />
        </div>
      </div>
    </Card>
  );
}
```

---

## 🏪 **State Management (Zustand)**

### **App Store** (Global State)

```tsx
// src/stores/appStore.ts

import create from 'zustand';
import { persist } from 'zustand/middleware';

interface AppState {
  // User state
  isFirstRun: boolean;
  onboardingComplete: boolean;
  
  // Current environment
  environment: string;
  
  // View state
  currentTab: string;
  sidebarCollapsed: boolean;
  
  // Settings
  theme: 'dark' | 'light';
  
  // Actions
  setFirstRun: (value: boolean) => void;
  completeOnboarding: () => void;
  setEnvironment: (env: string) => void;
  setCurrentTab: (tab: string) => void;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Initial state
      isFirstRun: true,
      onboardingComplete: false,
      environment: 'megabonk',
      currentTab: 'home',
      sidebarCollapsed: false,
      theme: 'dark',
      
      // Actions
      setFirstRun: (value) => set({ isFirstRun: value }),
      completeOnboarding: () => set({
        onboardingComplete: true,
        isFirstRun: false,
      }),
      setEnvironment: (env) => set({ environment: env }),
      setCurrentTab: (tab) => set({ currentTab: tab }),
      toggleSidebar: () => set((state) => ({
        sidebarCollapsed: !state.sidebarCollapsed,
      })),
    }),
    {
      name: 'metabonk-app-store',
    }
  )
);
```

### **Agent Store**

```tsx
// src/stores/agentStore.ts

import create from 'zustand';

interface Agent {
  id: string;
  name: string;
  status: 'idle' | 'training' | 'paused' | 'stopped';
  environment: string;
  step: number;
  metrics: {
    reward: number;
    fps: number;
    successRate: number;
    lastAction: string;
  };
  mind: {
    system1: {
      intent: string;
      confidence: number;
    };
    system2: {
      plan: string;
      reasoning: string;
      skill: string;
    };
  };
}

interface AgentState {
  agents: Agent[];
  focusedAgentId: string | null;
  
  // Actions
  addAgent: (agent: Agent) => void;
  updateAgent: (id: string, updates: Partial<Agent>) => void;
  removeAgent: (id: string) => void;
  setFocusedAgent: (id: string) => void;
}

export const useAgentStore = create<AgentState>((set) => ({
  agents: [],
  focusedAgentId: null,
  
  addAgent: (agent) => set((state) => ({
    agents: [...state.agents, agent],
  })),
  
  updateAgent: (id, updates) => set((state) => ({
    agents: state.agents.map((agent) =>
      agent.id === id ? { ...agent, ...updates } : agent
    ),
  })),
  
  removeAgent: (id) => set((state) => ({
    agents: state.agents.filter((agent) => agent.id !== id),
    focusedAgentId: state.focusedAgentId === id ? null : state.focusedAgentId,
  })),
  
  setFocusedAgent: (id) => set({ focusedAgentId: id }),
}));
```

---

## 🔌 **Backend Integration (Tauri)**

### **Tauri Commands** (Rust)

```rust
// src-tauri/src/main.rs

use tauri::State;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Default)]
struct AppState {
    synthetic_eye: Mutex<Option<SyntheticEyeClient>>,
    discovery: Mutex<DiscoveryState>,
}

#[derive(Serialize)]
struct GpuInfo {
    gpu_name: String,
    cuda_version: String,
    kernel_version: String,
    dmabuf_support: bool,
}

#[tauri::command]
async fn check_synthetic_eye_requirements() -> Result<GpuInfo, String> {
    // Check NVIDIA GPU
    let gpu_info = check_gpu_info()?;
    
    // Check CUDA
    let cuda_version = check_cuda_version()?;
    
    // Check kernel
    let kernel_version = check_kernel_version()?;
    
    // Check DMA-BUF support
    let dmabuf_support = check_dmabuf_support()?;
    
    Ok(GpuInfo {
        gpu_name: gpu_info,
        cuda_version,
        kernel_version,
        dmabuf_support,
    })
}

#[tauri::command]
async fn start_synthetic_eye(
    config: SyntheticEyeConfig,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut eye = state.synthetic_eye.lock().unwrap();
    
    // Start compositor
    let client = SyntheticEyeClient::connect(
        &config.socket_path,
        config.width,
        config.height,
        config.lockstep,
    ).map_err(|e| e.to_string())?;
    
    *eye = Some(client);
    
    Ok(())
}

#[tauri::command]
async fn start_autonomous_discovery(
    environment: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut discovery = state.discovery.lock().unwrap();
    
    // Start discovery pipeline
    discovery.start(environment)?;
    
    Ok(())
}

#[tauri::command]
async fn get_discovery_status(
    state: State<'_, AppState>,
) -> Result<DiscoveryStatus, String> {
    let discovery = state.discovery.lock().unwrap();
    Ok(discovery.get_status())
}

#[tauri::command]
async fn start_training(
    config: TrainingConfig,
) -> Result<(), String> {
    // Launch training
    launch_omega_protocol(config)?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            check_synthetic_eye_requirements,
            start_synthetic_eye,
            start_autonomous_discovery,
            get_discovery_status,
            start_training,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### **WebSocket Manager** (TypeScript)

```tsx
// src/services/websocket.ts

class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private listeners: Map<string, Set<Function>> = new Map();

  connect(url: string = 'ws://localhost:8765') {
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      console.log('✓ WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connection', { status: 'connected' });
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.ws.onclose = () => {
      console.warn('WebSocket disconnected');
      this.emit('connection', { status: 'disconnected' });
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', { error });
    };
  }

  private handleMessage(data: any) {
    const { type, payload } = data;

    // Route to appropriate handler
    switch (type) {
      case 'agent_update':
        useAgentStore.getState().updateAgent(payload.id, payload);
        break;
      
      case 'discovery_progress':
        useDiscoveryStore.getState().updateProgress(payload);
        break;
      
      case 'metrics_update':
        useMetricsStore.getState().updateMetrics(payload);
        break;
      
      case 'system_health':
        // Handle system health update
        break;
      
      default:
        // Emit to custom listeners
        this.emit(type, payload);
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach((callback) => {
      callback(data);
    });
  }

  send(type: string, payload: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    }
  }

  disconnect() {
    this.ws?.close();
  }
}

export const wsManager = new WebSocketManager();
```

---

This is Part 2. Should I continue with Part 3 (Production Polish, Error Handling, Testing)?
