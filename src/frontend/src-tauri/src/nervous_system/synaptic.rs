use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub struct SynapticState {
  pub gate: Mutex<SynapticCommandGate>,
}

#[derive(Clone)]
struct Synapse {
  base_cooldown: Duration,
  last_fire: Option<Instant>,
  strength: f32,
}

impl Synapse {
  fn new(base_cooldown: Duration) -> Self {
    Self {
      base_cooldown,
      last_fire: None,
      strength: 1.0,
    }
  }

  fn decay_strength(&mut self, now: Instant) {
    if let Some(last) = self.last_fire {
      let dt = now.saturating_duration_since(last).as_secs_f32();
      // Exponential decay back toward 1.0 over ~60s.
      let decay = (-dt / 60.0).exp();
      self.strength = 1.0 + (self.strength - 1.0) * decay;
    }
  }

  fn current_cooldown(&self) -> Duration {
    // Stronger pathways have shorter cooldowns (bounded).
    let s = self.strength.clamp(0.5, 2.0);
    let secs = self.base_cooldown.as_secs_f32() / s;
    Duration::from_secs_f32(secs.max(0.0))
  }

  fn can_fire(&self, now: Instant) -> bool {
    match self.last_fire {
      None => true,
      Some(last) => now.saturating_duration_since(last) >= self.current_cooldown(),
    }
  }

  fn fire(&mut self, now: Instant) {
    self.last_fire = Some(now);
    // Potentiation: strengthen with use, bounded.
    self.strength = (self.strength * 1.05).min(2.0);
  }
}

pub struct SynapticCommandGate {
  synapses: HashMap<&'static str, Synapse>,
}

impl SynapticCommandGate {
  pub fn new() -> Self {
    let mut synapses = HashMap::new();
    // Heavy commands get a refractory period to avoid flooding.
    synapses.insert("start_omega", Synapse::new(Duration::from_secs_f32(2.0)));
    synapses.insert("stop_omega", Synapse::new(Duration::from_secs_f32(2.0)));
    synapses.insert("start_discovery", Synapse::new(Duration::from_secs_f32(1.0)));
    synapses.insert("stop_discovery", Synapse::new(Duration::from_secs_f32(1.0)));
    synapses.insert("start_training", Synapse::new(Duration::from_secs_f32(2.0)));
    synapses.insert("process_videos", Synapse::new(Duration::from_secs_f32(2.0)));
    synapses.insert(
      "run_synthetic_eye_bench",
      Synapse::new(Duration::from_secs_f32(1.0)),
    );
    Self { synapses }
  }

  pub fn fire(&mut self, cmd: &'static str) -> Result<(), String> {
    let now = Instant::now();
    let Some(s) = self.synapses.get_mut(cmd) else {
      return Ok(());
    };
    s.decay_strength(now);
    if !s.can_fire(now) {
      let wait = s.current_cooldown().saturating_sub(now.saturating_duration_since(s.last_fire.unwrap_or(now)));
      return Err(format!(
        "command in refractory period: {cmd} (retry in ~{:.1}s)",
        wait.as_secs_f32()
      ));
    }
    s.fire(now);
    Ok(())
  }
}
