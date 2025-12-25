use std::{
    process::{Child, Command, Stdio},
    sync::mpsc::Sender,
    thread,
    time::Duration,
};

use anyhow::Context;
use tracing::{info, warn};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::ConnectionExt;

#[derive(Debug, Clone, Copy)]
pub enum ResetReason {
    XWaylandStall = 1,
}

#[derive(Debug)]
pub struct WatchdogEvent {
    pub reason: ResetReason,
}

pub struct XWaylandWatchdog {
    display: String,
    cmd: Vec<String>,
    heartbeat: Duration,
    fail_threshold: u32,
    tx: Sender<WatchdogEvent>,
}

impl XWaylandWatchdog {
    pub fn new(
        display: String,
        cmd: Vec<String>,
        heartbeat: Duration,
        fail_threshold: u32,
        tx: Sender<WatchdogEvent>,
    ) -> Self {
        Self {
            display,
            cmd,
            heartbeat,
            fail_threshold: fail_threshold.max(1),
            tx,
        }
    }

    pub fn spawn(self) -> thread::JoinHandle<()> {
        thread::spawn(move || self.run())
    }

    fn run(self) {
        let mut child: Option<Child> = None;
        let mut fails: u32 = 0;
        loop {
            if child.is_none() {
                match spawn_cmd(&self.cmd) {
                    Ok(p) => {
                        info!("xwayland watchdog spawned process (display={})", self.display);
                        child = Some(p);
                        fails = 0;
                    }
                    Err(e) => {
                        warn!("xwayland watchdog failed to spawn: {e}");
                        thread::sleep(Duration::from_secs(1));
                        continue;
                    }
                }
            }

            // If process exited, restart and notify.
            if let Some(p) = child.as_mut() {
                if let Ok(Some(status)) = p.try_wait() {
                    warn!("xwayland exited ({status}); restarting");
                    child = None;
                    let _ = self.tx.send(WatchdogEvent {
                        reason: ResetReason::XWaylandStall,
                    });
                    continue;
                }
            }

            let ok = heartbeat_x11(&self.display, Duration::from_millis(250));
            if ok {
                fails = 0;
            } else {
                fails += 1;
                if fails >= self.fail_threshold {
                    warn!(
                        "xwayland heartbeat failed {} times; restarting (display={})",
                        fails, self.display
                    );
                    if let Some(mut p) = child.take() {
                        let _ = p.kill();
                        let _ = p.wait();
                    }
                    let _ = self.tx.send(WatchdogEvent {
                        reason: ResetReason::XWaylandStall,
                    });
                    fails = 0;
                }
            }

            thread::sleep(self.heartbeat);
        }
    }
}

fn spawn_cmd(cmd: &[String]) -> anyhow::Result<Child> {
    if cmd.is_empty() {
        anyhow::bail!("empty xwayland_cmd");
    }
    let mut c = Command::new(&cmd[0]);
    for a in &cmd[1..] {
        c.arg(a);
    }
    c.stdin(Stdio::null());
    c.stdout(Stdio::null());
    c.stderr(Stdio::null());
    c.spawn().context("spawn")
}

fn heartbeat_x11(display: &str, timeout: Duration) -> bool {
    // x11rb does not provide a built-in blocking timeout per request; run the
    // heartbeat in a helper thread and join with a timeout.
    let display = display.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = thread::spawn(move || {
        let ok = (|| -> anyhow::Result<()> {
            let (conn, _screen_num) = x11rb::connect(Some(&display)).context("connect")?;
            let cookie = conn.get_input_focus().context("get_input_focus")?;
            let _ = cookie.reply().context("reply")?;
            conn.flush().ok();
            Ok(())
        })()
        .is_ok();
        let _ = tx.send(ok);
    });
    match rx.recv_timeout(timeout) {
        Ok(v) => v,
        Err(_) => false,
    }
}
