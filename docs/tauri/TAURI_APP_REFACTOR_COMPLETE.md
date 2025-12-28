# MetaBonk Tauri App: Complete Production Refactor
## SOTA Architecture for Autonomous AGI Training Platform

**Goal**: Transform MetaBonk into a professional, production-ready autonomous AGI training platform with clean UX, proper onboarding, and zero redundancy.

---

## 🎯 **Current Issues & Solutions**

### **Problems Identified**:
1. ❌ **Tab Redundancy** - Multiple tabs showing similar/overlapping information
2. ❌ **Clutter** - Too much information displayed at once
3. ❌ **No Onboarding** - First-time users are confused
4. ❌ **Missing Features** - Synthetic Eye, autonomous discovery not integrated
5. ❌ **Poor Information Hierarchy** - Hard to understand what's important

### **Solutions**:
1. ✅ **Clean Tab Architecture** - 5 focused tabs, each with clear purpose
2. ✅ **Progressive Disclosure** - Show simple view by default, advanced on demand
3. ✅ **Guided Setup** - Step-by-step onboarding flow
4. ✅ **Full Integration** - Synthetic Eye + Discovery + Training all wired
5. ✅ **Clear Hierarchy** - Most important info front and center

---

## 📐 **New Architecture**

### **Tab Structure** (Clean, No Redundancy)

```
┌─────────────────────────────────────────────────────────────┐
│  MetaBonk - Autonomous AGI Training Platform                │
├─────────────────────────────────────────────────────────────┤
│  🏠 Home  │  🎮 Agents  │  🔬 Discovery  │  📊 Analytics  │  ⚙️ Settings  │
└─────────────────────────────────────────────────────────────┘
```

#### **1. 🏠 Home** - Dashboard & Quick Actions
- **Purpose**: High-level status, quick actions, getting started
- **For New Users**: Guided setup wizard
- **For Power Users**: System health, quick launch, recent activity

#### **2. 🎮 Agents** - Training & Live View
- **Purpose**: Watch agents train, inspect their reasoning, control training
- **Replaces**: Old "Training", "Stream", "Live" tabs (consolidates)
- **Features**: Live video feed, agent mind (System 1+2), performance graphs

#### **3. 🔬 Discovery** - Autonomous Discovery Pipeline
- **Purpose**: Monitor/control autonomous discovery (Phases 0-3)
- **Replaces**: Manual configuration screens
- **Features**: Discovery progress, learned action space, effect visualization

#### **4. 📊 Analytics** - Performance & Metrics
- **Purpose**: Deep dive into training metrics, system performance
- **Replaces**: Old "Metrics", "Stats" tabs (consolidates)
- **Features**: Training curves, system health, comparative analysis

#### **5. ⚙️ Settings** - Configuration & Advanced
- **Purpose**: System configuration, advanced settings, troubleshooting
- **Features**: Synthetic Eye config, hyperparameters, debug tools

---

## 🎨 **UI/UX Design System**

### **Design Principles**

1. **Progressive Disclosure** - Simple by default, powerful when needed
2. **Guided Discovery** - Help users find features naturally
3. **Status-First** - Always show current state clearly
4. **Action-Oriented** - Clear CTAs, no confusion about what to do next
5. **Professional** - Clean, modern, trustworthy aesthetic

### **Color Scheme** (Cyberpunk AGI Theme)

```css
/* Primary Colors */
--primary: #00ff9f;        /* Neon green - success, active */
--secondary: #ff0080;      /* Hot pink - highlights, important */
--accent: #00d4ff;         /* Cyan - info, links */

/* Background Layers */
--bg-primary: #0a0e1a;     /* Deep space blue - main bg */
--bg-secondary: #12192e;   /* Elevated surfaces */
--bg-tertiary: #1a2542;    /* Cards, panels */

/* Text */
--text-primary: #ffffff;   /* Primary text */
--text-secondary: #8b9dc3; /* Secondary text */
--text-muted: #4a5f7f;     /* Muted/disabled */

/* Status Colors */
--success: #00ff9f;
--warning: #ffb900;
--error: #ff4757;
--info: #00d4ff;

/* Glassmorphism */
--glass-bg: rgba(18, 25, 46, 0.7);
--glass-border: rgba(139, 157, 195, 0.1);
```

### **Typography**

```css
/* Font Stack */
--font-display: 'Inter', -apple-system, system-ui, sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', monospace;

/* Type Scale */
--text-xs: 0.75rem;   /* 12px - labels, captions */
--text-sm: 0.875rem;  /* 14px - secondary text */
--text-base: 1rem;    /* 16px - body text */
--text-lg: 1.125rem;  /* 18px - subheadings */
--text-xl: 1.25rem;   /* 20px - headings */
--text-2xl: 1.5rem;   /* 24px - section titles */
--text-3xl: 1.875rem; /* 30px - page titles */
--text-4xl: 2.25rem;  /* 36px - hero text */
```

---

## 🏗️ **Component Architecture**

### **File Structure**

```
src/
├── components/
│   ├── common/                 # Shared components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Modal.tsx
│   │   ├── StatusBadge.tsx
│   │   ├── ProgressBar.tsx
│   │   ├── Tooltip.tsx
│   │   └── LoadingState.tsx
│   │
│   ├── layout/                 # Layout components
│   │   ├── AppShell.tsx       # Main app wrapper
│   │   ├── Navigation.tsx     # Tab navigation
│   │   ├── Sidebar.tsx        # Collapsible sidebar
│   │   └── Header.tsx         # Status bar
│   │
│   ├── onboarding/            # First-time user experience
│   │   ├── WelcomeWizard.tsx
│   │   ├── SetupSteps.tsx
│   │   ├── QuickTour.tsx
│   │   └── FirstRun.tsx
│   │
│   ├── home/                  # Home tab components
│   │   ├── Dashboard.tsx
│   │   ├── QuickActions.tsx
│   │   ├── SystemStatus.tsx
│   │   └── RecentActivity.tsx
│   │
│   ├── agents/                # Agents tab
│   │   ├── AgentGrid.tsx      # Multi-agent view
│   │   ├── AgentCard.tsx      # Single agent card
│   │   ├── LiveFeed.tsx       # Video stream
│   │   ├── MindPanel.tsx      # System 1+2 reasoning
│   │   ├── ActionHistory.tsx  # Recent actions
│   │   └── PerformanceGraph.tsx
│   │
│   ├── discovery/             # Discovery tab
│   │   ├── DiscoveryPipeline.tsx
│   │   ├── PhaseProgress.tsx
│   │   ├── ActionSpaceViz.tsx
│   │   ├── EffectHeatmap.tsx
│   │   └── ClusterView.tsx
│   │
│   ├── analytics/             # Analytics tab
│   │   ├── MetricsDashboard.tsx
│   │   ├── TrainingCurves.tsx
│   │   ├── SystemHealth.tsx
│   │   └── ComparisonView.tsx
│   │
│   └── settings/              # Settings tab
│       ├── GeneralSettings.tsx
│       ├── SyntheticEyeConfig.tsx
│       ├── HyperparametersPanel.tsx
│       └── DebugTools.tsx
│
├── hooks/                     # Custom React hooks
│   ├── useWebSocket.ts       # Real-time updates
│   ├── useDiscoveryStatus.ts
│   ├── useAgentMetrics.ts
│   └── useSystemHealth.ts
│
├── stores/                    # State management (Zustand)
│   ├── appStore.ts           # Global app state
│   ├── agentStore.ts         # Agent data
│   ├── discoveryStore.ts     # Discovery state
│   └── metricsStore.ts       # Metrics data
│
├── services/                  # Backend communication
│   ├── api.ts                # REST API client
│   ├── websocket.ts          # WebSocket manager
│   └── tauri.ts              # Tauri commands
│
├── types/                     # TypeScript types
│   ├── agent.ts
│   ├── discovery.ts
│   ├── metrics.ts
│   └── config.ts
│
└── pages/                     # Main tab pages
    ├── HomePage.tsx
    ├── AgentsPage.tsx
    ├── DiscoveryPage.tsx
    ├── AnalyticsPage.tsx
    └── SettingsPage.tsx
```

---

## 🚀 **1. Home Tab - Dashboard & Onboarding**

### **First-Time User Flow**

```tsx
// src/components/onboarding/WelcomeWizard.tsx

import React, { useState } from 'react';
import { Button, Card, ProgressBar } from '@/components/common';

const SETUP_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to MetaBonk',
    description: 'The autonomous AGI training platform',
    component: WelcomeStep,
  },
  {
    id: 'synthetic-eye',
    title: 'Configure Synthetic Eye',
    description: 'Zero-copy observations (500+ FPS)',
    component: SyntheticEyeStep,
  },
  {
    id: 'select-game',
    title: 'Select Environment',
    description: 'Choose a game or application',
    component: GameSelectionStep,
  },
  {
    id: 'discovery',
    title: 'Autonomous Discovery',
    description: 'Let agents discover controls',
    component: DiscoveryStep,
  },
  {
    id: 'ready',
    title: 'Ready to Train!',
    description: 'Start your first training run',
    component: ReadyStep,
  },
];

export function WelcomeWizard() {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState({});

  const step = SETUP_STEPS[currentStep];
  const StepComponent = step.component;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <Card className="max-w-3xl w-full p-8">
        {/* Progress */}
        <div className="mb-8">
          <ProgressBar 
            value={((currentStep + 1) / SETUP_STEPS.length) * 100}
            className="mb-2"
          />
          <p className="text-sm text-muted">
            Step {currentStep + 1} of {SETUP_STEPS.length}
          </p>
        </div>

        {/* Title */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">{step.title}</h2>
          <p className="text-secondary">{step.description}</p>
        </div>

        {/* Step Content */}
        <StepComponent
          config={config}
          setConfig={setConfig}
          onNext={() => setCurrentStep(current => current + 1)}
          onBack={() => setCurrentStep(current => current - 1)}
        />
      </Card>
    </div>
  );
}
```

### **WelcomeStep Component**

```tsx
// src/components/onboarding/steps/WelcomeStep.tsx

export function WelcomeStep({ onNext }) {
  return (
    <div className="space-y-6">
      {/* Hero Visual */}
      <div className="aspect-video bg-gradient-to-br from-primary/20 to-secondary/20 rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">🧠</div>
          <h3 className="text-2xl font-bold">Autonomous AGI Training</h3>
          <p className="text-secondary mt-2">
            Zero configuration • Zero-copy observations • Autonomous discovery
          </p>
        </div>
      </div>

      {/* Key Features */}
      <div className="grid grid-cols-3 gap-4">
        <FeatureCard
          icon="🔬"
          title="Synthetic Eye"
          description="500+ FPS zero-copy observations"
        />
        <FeatureCard
          icon="🎯"
          title="Auto-Discovery"
          description="Agents discover controls themselves"
        />
        <FeatureCard
          icon="🧠"
          title="Dual Cognition"
          description="System 1 (fast) + System 2 (reasoning)"
        />
      </div>

      {/* What You'll Set Up */}
      <Card className="bg-secondary/5 border-secondary/20">
        <h4 className="font-semibold mb-3">What we'll set up:</h4>
        <ul className="space-y-2 text-sm text-secondary">
          <li className="flex items-start gap-2">
            <span className="text-primary">✓</span>
            <span>Synthetic Eye compositor for ultra-fast observations</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">✓</span>
            <span>Select your game/application environment</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">✓</span>
            <span>Run autonomous discovery (agents learn controls)</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary">✓</span>
            <span>Start your first training run</span>
          </li>
        </ul>
      </Card>

      {/* CTA */}
      <div className="flex justify-between items-center pt-6 border-t border-glass-border">
        <Button variant="ghost" onClick={() => window.close()}>
          Skip Setup (Advanced)
        </Button>
        <Button onClick={onNext} size="lg">
          Let's Get Started →
        </Button>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <Card className="text-center p-4">
      <div className="text-3xl mb-2">{icon}</div>
      <h4 className="font-semibold text-sm mb-1">{title}</h4>
      <p className="text-xs text-secondary">{description}</p>
    </Card>
  );
}
```

### **SyntheticEyeStep Component**

```tsx
// src/components/onboarding/steps/SyntheticEyeStep.tsx

import { invoke } from '@tauri-apps/api/tauri';
import { useState, useEffect } from 'react';

export function SyntheticEyeStep({ config, setConfig, onNext, onBack }) {
  const [status, setStatus] = useState<'checking' | 'ready' | 'error'>('checking');
  const [gpuInfo, setGpuInfo] = useState(null);

  useEffect(() => {
    checkSyntheticEye();
  }, []);

  async function checkSyntheticEye() {
    try {
      const info = await invoke('check_synthetic_eye_requirements');
      setGpuInfo(info);
      setStatus('ready');
    } catch (error) {
      setStatus('error');
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-info/5 border-info/20">
        <h4 className="font-semibold mb-2 flex items-center gap-2">
          <span className="text-2xl">🔬</span>
          What is Synthetic Eye?
        </h4>
        <p className="text-sm text-secondary mb-4">
          Synthetic Eye is a custom Wayland compositor that provides zero-copy observations
          directly from GPU memory. This eliminates the PCIe bottleneck and enables
          training at <span className="text-primary font-semibold">500+ FPS</span> with
          <span className="text-primary font-semibold">&lt;1ms latency</span>.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <span className="text-success">✓</span>
            <span>Zero PCIe bandwidth usage</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-success">✓</span>
            <span>Sub-millisecond latency</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-success">✓</span>
            <span>GPU-resident tensors</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-success">✓</span>
            <span>Lockstep synchronization</span>
          </div>
        </div>
      </Card>

      {/* System Check */}
      <Card>
        <h4 className="font-semibold mb-4">System Requirements</h4>
        
        {status === 'checking' && (
          <div className="text-center py-8">
            <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" />
            <p className="text-sm text-secondary">Checking system...</p>
          </div>
        )}

        {status === 'ready' && (
          <div className="space-y-3">
            <CheckItem
              label="NVIDIA GPU (RTX/GTX)"
              status="pass"
              detail={gpuInfo?.gpu_name}
            />
            <CheckItem
              label="CUDA 11.0+"
              status="pass"
              detail={`CUDA ${gpuInfo?.cuda_version}`}
            />
            <CheckItem
              label="Linux Kernel 5.15+"
              status="pass"
              detail={`Kernel ${gpuInfo?.kernel_version}`}
            />
            <CheckItem
              label="DMA-BUF support"
              status="pass"
              detail="zwp_linux_dmabuf_v1"
            />
          </div>
        )}

        {status === 'error' && (
          <div className="space-y-4">
            <div className="bg-error/10 border border-error/20 rounded-lg p-4">
              <p className="text-error font-semibold mb-2">⚠️ Requirements Not Met</p>
              <p className="text-sm text-secondary">
                Synthetic Eye requires an NVIDIA GPU with CUDA support on Linux.
              </p>
            </div>
            <div className="bg-secondary/5 rounded-lg p-4">
              <p className="text-sm font-semibold mb-2">Fallback Mode Available:</p>
              <p className="text-xs text-secondary">
                You can still use MetaBonk with traditional screen capture (60 FPS).
                Performance will be reduced but everything will work.
              </p>
            </div>
          </div>
        )}
      </Card>

      {/* Configuration */}
      {status === 'ready' && (
        <Card>
          <h4 className="font-semibold mb-4">Configuration</h4>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Resolution
              </label>
              <select
                className="w-full bg-bg-secondary border border-glass-border rounded-lg px-3 py-2"
                value={config.resolution || '1920x1080'}
                onChange={(e) => setConfig({ ...config, resolution: e.target.value })}
              >
                <option value="1920x1080">1920x1080 (Full HD)</option>
                <option value="2560x1440">2560x1440 (2K)</option>
                <option value="3840x2160">3840x2160 (4K)</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Lockstep Mode
                </label>
                <p className="text-xs text-secondary">
                  Agent controls frame timing (recommended)
                </p>
              </div>
              <input
                type="checkbox"
                checked={config.lockstep !== false}
                onChange={(e) => setConfig({ ...config, lockstep: e.target.checked })}
                className="toggle"
              />
            </div>
          </div>
        </Card>
      )}

      {/* Navigation */}
      <div className="flex justify-between pt-6 border-t border-glass-border">
        <Button variant="ghost" onClick={onBack}>
          ← Back
        </Button>
        <Button
          onClick={onNext}
          disabled={status === 'checking'}
        >
          Continue →
        </Button>
      </div>
    </div>
  );
}

function CheckItem({ label, status, detail }) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-3">
        <span className={status === 'pass' ? 'text-success' : 'text-error'}>
          {status === 'pass' ? '✓' : '✗'}
        </span>
        <span className="text-sm">{label}</span>
      </div>
      {detail && <span className="text-xs text-secondary">{detail}</span>}
    </div>
  );
}
```

### **Returning User Dashboard**

```tsx
// src/pages/HomePage.tsx

export function HomePage() {
  const { isFirstRun } = useAppStore();
  const { systemStatus } = useSystemHealth();
  const { recentRuns } = useRecentActivity();

  if (isFirstRun) {
    return <WelcomeWizard />;
  }

  return (
    <div className="p-6 space-y-6">
      {/* Status Bar */}
      <SystemStatusBar status={systemStatus} />

      {/* Quick Actions */}
      <section>
        <h2 className="text-2xl font-bold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-3 gap-4">
          <QuickActionCard
            icon="🚀"
            title="Start Training"
            description="Resume or start new run"
            onClick={() => startTraining()}
            variant="primary"
          />
          <QuickActionCard
            icon="🔬"
            title="Run Discovery"
            description="Discover new game controls"
            onClick={() => runDiscovery()}
            variant="secondary"
          />
          <QuickActionCard
            icon="📊"
            title="View Analytics"
            description="See training performance"
            onClick={() => navigate('/analytics')}
          />
        </div>
      </section>

      {/* System Status */}
      <section className="grid grid-cols-3 gap-4">
        <StatusCard
          label="Synthetic Eye"
          status={systemStatus.syntheticEye}
          metric="542 FPS"
        />
        <StatusCard
          label="Active Agents"
          status="running"
          metric="8 / 8"
        />
        <StatusCard
          label="GPU Utilization"
          status="optimal"
          metric="87%"
        />
      </section>

      {/* Recent Activity */}
      <section>
        <h3 className="text-xl font-semibold mb-4">Recent Runs</h3>
        <div className="space-y-2">
          {recentRuns.map(run => (
            <RecentRunCard key={run.id} run={run} />
          ))}
        </div>
      </section>
    </div>
  );
}
```

---

## 🎮 **2. Agents Tab - Live Training View**

**Consolidates**: Old "Training", "Stream", "Live" tabs into one clean interface.

```tsx
// src/pages/AgentsPage.tsx

import { useState } from 'react';
import { AgentGrid, AgentCard, LiveFeed, MindPanel } from '@/components/agents';

export function AgentsPage() {
  const { agents } = useAgentStore();
  const [focusedAgent, setFocusedAgent] = useState(null);
  const [viewMode, setViewMode] = useState<'grid' | 'focus'>('grid');

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-glass-border">
        <div>
          <h2 className="text-2xl font-bold">Agents</h2>
          <p className="text-sm text-secondary">
            {agents.filter(a => a.status === 'training').length} agents training
          </p>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === 'grid' ? 'primary' : 'ghost'}
            onClick={() => setViewMode('grid')}
          >
            Grid View
          </Button>
          <Button
            variant={viewMode === 'focus' ? 'primary' : 'ghost'}
            onClick={() => setViewMode('focus')}
            disabled={!focusedAgent}
          >
            Focus Mode
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {viewMode === 'grid' ? (
          <AgentGrid
            agents={agents}
            onAgentClick={setFocusedAgent}
          />
        ) : (
          <FocusView agent={focusedAgent} />
        )}
      </div>
    </div>
  );
}
```

### **Agent Grid View**

```tsx
// src/components/agents/AgentGrid.tsx

export function AgentGrid({ agents, onAgentClick }) {
  return (
    <div className="grid grid-cols-4 gap-4 p-4">
      {agents.map(agent => (
        <AgentCard
          key={agent.id}
          agent={agent}
          onClick={() => onAgentClick(agent)}
        />
      ))}
    </div>
  );
}

export function AgentCard({ agent, onClick }) {
  const { stream, metrics } = useAgentMetrics(agent.id);

  return (
    <Card
      className="cursor-pointer hover:border-primary transition-all"
      onClick={onClick}
    >
      {/* Video Stream */}
      <div className="aspect-video bg-black rounded-t-lg overflow-hidden relative">
        <LiveFeed stream={stream} size="thumbnail" />
        
        {/* Overlay Info */}
        <div className="absolute top-2 left-2 flex gap-1">
          <StatusBadge status={agent.status} size="sm" />
          <Badge variant="dark">{metrics.fps} FPS</Badge>
        </div>
      </div>

      {/* Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h4 className="font-semibold">{agent.name}</h4>
          <span className="text-xs text-secondary">
            Step {metrics.step.toLocaleString()}
          </span>
        </div>

        {/* Mini Metrics */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-secondary">Reward:</span>
            <span className="ml-1 text-primary font-mono">
              {metrics.reward.toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-secondary">Success:</span>
            <span className="ml-1 text-success font-mono">
              {(metrics.successRate * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Current Action */}
        <div className="text-xs text-secondary truncate">
          <span>Action:</span> <code className="ml-1">{metrics.lastAction}</code>
        </div>
      </div>
    </Card>
  );
}
```

### **Focus View (Single Agent Detail)**

```tsx
// src/components/agents/FocusView.tsx

export function FocusView({ agent }) {
  const { stream, metrics, mind } = useAgentMetrics(agent.id);

  return (
    <div className="grid grid-cols-3 gap-4 p-4 h-full">
      {/* Left: Video Feed (2 cols) */}
      <div className="col-span-2 space-y-4">
        {/* Main Video */}
        <Card className="aspect-video bg-black overflow-hidden">
          <LiveFeed stream={stream} size="full" />
        </Card>

        {/* Performance Graph */}
        <Card className="h-64">
          <h4 className="font-semibold mb-3">Reward Over Time</h4>
          <PerformanceGraph data={metrics.rewardHistory} />
        </Card>

        {/* Action History */}
        <Card>
          <h4 className="font-semibold mb-3">Recent Actions</h4>
          <ActionHistory actions={metrics.actionHistory} />
        </Card>
      </div>

      {/* Right: Mind Panel */}
      <div className="space-y-4">
        {/* Agent Info */}
        <Card>
          <h3 className="text-xl font-bold mb-2">{agent.name}</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-secondary">Status:</span>
              <StatusBadge status={agent.status} />
            </div>
            <div className="flex justify-between">
              <span className="text-secondary">Step:</span>
              <span className="font-mono">{metrics.step.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-secondary">FPS:</span>
              <span className="font-mono text-primary">{metrics.fps}</span>
            </div>
          </div>
        </Card>

        {/* Mind Panel (System 1 + 2) */}
        <MindPanel mind={mind} />

        {/* Quick Actions */}
        <Card>
          <h4 className="font-semibold mb-3">Controls</h4>
          <div className="space-y-2">
            <Button variant="outline" className="w-full">
              Pause Training
            </Button>
            <Button variant="outline" className="w-full">
              Save Checkpoint
            </Button>
            <Button variant="danger" className="w-full">
              Stop Agent
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
```

### **Mind Panel (System 1 + System 2 Reasoning)**

```tsx
// src/components/agents/MindPanel.tsx

export function MindPanel({ mind }) {
  return (
    <Card className="bg-gradient-to-br from-primary/5 to-secondary/5">
      <h4 className="font-semibold mb-4 flex items-center gap-2">
        <span className="text-2xl">🧠</span>
        Agent Mind
      </h4>

      {/* System 1 (Fast Thinking) */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h5 className="text-sm font-semibold text-primary">
            System 1 (Fast)
          </h5>
          <span className="text-xs text-secondary">60 Hz</span>
        </div>
        <Card className="bg-bg-primary/50 text-sm">
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-secondary">Intent:</span>
              <span className="font-mono">{mind.system1.intent}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-secondary">Confidence:</span>
              <ProgressBar value={mind.system1.confidence * 100} size="sm" />
            </div>
          </div>
        </Card>
      </div>

      {/* System 2 (Slow Thinking) */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h5 className="text-sm font-semibold text-secondary">
            System 2 (Reasoning)
          </h5>
          <span className="text-xs text-secondary">0.5 Hz</span>
        </div>
        <Card className="bg-bg-primary/50 text-sm">
          <div className="space-y-2">
            <div>
              <span className="text-secondary text-xs">Current Plan:</span>
              <p className="text-xs mt-1">{mind.system2.plan}</p>
            </div>
            <div>
              <span className="text-secondary text-xs">Reasoning:</span>
              <p className="text-xs mt-1 text-muted italic">
                "{mind.system2.reasoning}"
              </p>
            </div>
            <div className="flex justify-between items-center text-xs">
              <span className="text-secondary">Skill:</span>
              <span className="font-mono">{mind.system2.skill}</span>
            </div>
          </div>
        </Card>
      </div>
    </Card>
  );
}
```

---

## 🔬 **3. Discovery Tab - Autonomous Discovery Pipeline**

**Replaces**: Manual configuration screens, button mapping UIs.

```tsx
// src/pages/DiscoveryPage.tsx

export function DiscoveryPage() {
  const { environment, setEnvironment } = useAppStore();
  const { status, phase, progress } = useDiscoveryStatus(environment);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Autonomous Discovery</h2>
          <p className="text-secondary">
            Let agents discover game controls automatically
          </p>
        </div>

        {/* Environment Selector */}
        <select
          value={environment}
          onChange={(e) => setEnvironment(e.target.value)}
          className="bg-bg-secondary border border-glass-border rounded-lg px-4 py-2"
        >
          <option value="megabonk">MegaBonk</option>
          <option value="doom">DOOM</option>
          <option value="minecraft">Minecraft</option>
          <option value="custom">Custom...</option>
        </select>
      </div>

      {/* Discovery Pipeline */}
      <DiscoveryPipeline
        environment={environment}
        status={status}
        phase={phase}
        progress={progress}
      />

      {/* Results (if complete) */}
      {status === 'complete' && (
        <>
          <ActionSpaceVisualization environment={environment} />
          <EffectHeatmap environment={environment} />
          <ClusterView environment={environment} />
        </>
      )}
    </div>
  );
}
```

### **Discovery Pipeline Component**

```tsx
// src/components/discovery/DiscoveryPipeline.tsx

const PHASES = [
  {
    id: 0,
    name: 'Input Enumeration',
    description: 'Discover available keyboard & mouse inputs',
    icon: '⌨️',
  },
  {
    id: 1,
    name: 'Effect Detection',
    description: 'Test each input and measure effects',
    icon: '🔍',
  },
  {
    id: 2,
    name: 'Semantic Clustering',
    description: 'Group similar actions together',
    icon: '🧬',
  },
  {
    id: 3,
    name: 'Action Space Construction',
    description: 'Build optimal action space for training',
    icon: '🎯',
  },
];

export function DiscoveryPipeline({ environment, status, phase, progress }) {
  const [isRunning, setIsRunning] = useState(false);

  async function startDiscovery() {
    setIsRunning(true);
    await invoke('start_autonomous_discovery', { environment });
  }

  return (
    <Card>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold">Discovery Pipeline</h3>
        
        {status === 'idle' && (
          <Button onClick={startDiscovery} disabled={isRunning}>
            {isRunning ? 'Starting...' : 'Start Discovery'}
          </Button>
        )}
        
        {status === 'running' && (
          <Badge variant="primary">Running Phase {phase}</Badge>
        )}
        
        {status === 'complete' && (
          <Badge variant="success">✓ Complete</Badge>
        )}
      </div>

      {/* Phase Progress */}
      <div className="space-y-4">
        {PHASES.map((p) => (
          <PhaseCard
            key={p.id}
            phase={p}
            current={phase === p.id}
            complete={phase > p.id || status === 'complete'}
            progress={phase === p.id ? progress : phase > p.id ? 100 : 0}
          />
        ))}
      </div>

      {/* Status Messages */}
      {status === 'running' && (
        <Card className="mt-4 bg-info/5 border-info/20">
          <p className="text-sm">
            <strong>Phase {phase}:</strong> {PHASES[phase].description}
          </p>
          <ProgressBar value={progress} className="mt-2" />
        </Card>
      )}
    </Card>
  );
}

function PhaseCard({ phase, current, complete, progress }) {
  return (
    <Card
      className={cn(
        'transition-all',
        current && 'border-primary bg-primary/5',
        complete && 'border-success/30'
      )}
    >
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className={cn(
          'text-3xl p-3 rounded-lg',
          complete ? 'bg-success/20' : current ? 'bg-primary/20' : 'bg-bg-secondary'
        )}>
          {complete ? '✓' : phase.icon}
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex items-center justify-between mb-1">
            <h4 className="font-semibold">Phase {phase.id}: {phase.name}</h4>
            {complete && (
              <span className="text-xs text-success">Complete</span>
            )}
          </div>
          <p className="text-sm text-secondary">{phase.description}</p>
          
          {current && progress > 0 && (
            <ProgressBar value={progress} className="mt-2" size="sm" />
          )}
        </div>
      </div>
    </Card>
  );
}
```

### **Action Space Visualization**

```tsx
// src/components/discovery/ActionSpaceVisualization.tsx

export function ActionSpaceVisualization({ environment }) {
  const { actionSpace } = useDiscoveryResults(environment);

  if (!actionSpace) return null;

  return (
    <Card>
      <h3 className="text-xl font-semibold mb-4">Learned Action Space</h3>

      <div className="grid grid-cols-2 gap-6">
        {/* Discrete Actions */}
        <div>
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <span className="text-primary">⌨️</span>
            Discrete Actions ({actionSpace.discrete.length})
          </h4>
          <div className="space-y-1 max-h-96 overflow-auto">
            {actionSpace.discrete.map((action, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-2 bg-bg-secondary rounded text-sm"
              >
                <div className="flex items-center gap-2">
                  <Badge size="sm">{action.binding.name}</Badge>
                  <span className="text-secondary">{action.semantic_label}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted">
                    Utility: {(action.utility * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Continuous Actions */}
        <div>
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <span className="text-secondary">🖱️</span>
            Continuous Actions ({Object.keys(actionSpace.continuous).length})
          </h4>
          <div className="space-y-2">
            {Object.entries(actionSpace.continuous).map(([name, config]) => (
              <Card key={name} className="bg-bg-secondary">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-sm">{name}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-secondary">
                  <span>Range:</span>
                  <code>[{config.min}, {config.max}]</code>
                  <span>Scale:</span>
                  <code>{config.scale}</code>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <Card className="mt-4 bg-success/5 border-success/20">
        <div className="grid grid-cols-4 gap-4 text-center text-sm">
          <div>
            <div className="text-2xl font-bold text-success">
              {actionSpace.metadata.total_discovered}
            </div>
            <div className="text-secondary">Total Discovered</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-success">
              {actionSpace.metadata.selected_discrete}
            </div>
            <div className="text-secondary">Selected</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-success">
              {actionSpace.metadata.cluster_coverage}/{actionSpace.metadata.total_clusters}
            </div>
            <div className="text-secondary">Cluster Coverage</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-success">
              {((actionSpace.metadata.cluster_coverage / actionSpace.metadata.total_clusters) * 100).toFixed(0)}%
            </div>
            <div className="text-secondary">Diversity</div>
          </div>
        </div>
      </Card>
    </Card>
  );
}
```

---

This is Part 1 of the complete refactor. Should I continue with:
- **Part 2**: Analytics Tab, Settings Tab, State Management
- **Part 3**: Backend Integration (Tauri Commands, WebSocket)
- **Part 4**: Deployment & Production Hardening

Would you like me to continue with the complete implementation?
