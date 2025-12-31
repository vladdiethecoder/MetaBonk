use std::{
    fs,
    io::{Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    os::unix::prelude::AsRawFd,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::Context;
use bytes::BytesMut;
use metabonk_frame_abi::{HeaderV1, MsgType, MAGIC, VERSION};
use std::io::IoSlice;

use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags};
use nix::{errno::Errno, sys::socket::recv};
use tracing::{debug, info, warn};

pub struct FrameServer {
    listener: UnixListener,
    stream: Option<UnixStream>,
}

impl FrameServer {
    pub fn bind(sock_path: &str) -> anyhow::Result<Self> {
        if let Some(parent) = Path::new(sock_path).parent() {
            fs::create_dir_all(parent).with_context(|| format!("create_dir_all({parent:?})"))?;
        }
        // Remove stale socket.
        let _ = fs::remove_file(sock_path);
        let listener = UnixListener::bind(sock_path).with_context(|| format!("bind({sock_path})"))?;
        listener
            .set_nonblocking(false)
            .context("set_nonblocking(false)")?;
        Ok(Self {
            listener,
            stream: None,
        })
    }

    pub fn accept_if_needed(&mut self) -> anyhow::Result<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        info!("waiting for worker connection...");
        let (mut stream, _) = self.listener.accept().context("accept")?;
        stream.set_nonblocking(false).context("set_nonblocking(false)")?;
        self.handle_optional_hello(&mut stream)?;
        self.send_hello_ack(&mut stream)?;
        info!("worker connected");
        self.stream = Some(stream);
        Ok(())
    }

    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    fn handle_optional_hello(&self, stream: &mut UnixStream) -> anyhow::Result<()> {
        let raw_fd = stream.as_raw_fd();
        let mut hdr = [0u8; HeaderV1::LEN];
        // Best-effort: peek without blocking. If no HELLO is queued, proceed.
        match recv(raw_fd, &mut hdr, MsgFlags::MSG_PEEK | MsgFlags::MSG_DONTWAIT) {
            Ok(n) => {
                if n < HeaderV1::LEN {
                    return Ok(());
                }
            }
            Err(Errno::EAGAIN) => return Ok(()),
            Err(err) => return Err(anyhow::anyhow!("recv(MSG_PEEK) failed: {err}")),
        }

        if &hdr[0..8] != &MAGIC[..] || u16::from_le_bytes([hdr[8], hdr[9]]) != VERSION {
            return Ok(());
        }
        let header = match HeaderV1::decode(&hdr) {
            Ok(h) => h,
            Err(_) => return Ok(()),
        };
        if header.msg_type != MsgType::Hello {
            return Ok(());
        }
        debug!("received HELLO");

        // Consume the HELLO so subsequent reads see the next message (e.g., PING).
        stream.read_exact(&mut hdr).context("read hello header")?;
        if header.payload_len > 0 {
            let mut payload = vec![0u8; header.payload_len as usize];
            stream.read_exact(&mut payload).context("read hello payload")?;
        }
        Ok(())
    }

    fn send_hello_ack(&self, stream: &mut UnixStream) -> anyhow::Result<()> {
        let mut buf = BytesMut::with_capacity(HeaderV1::LEN);
        HeaderV1 {
            msg_type: MsgType::HelloAck,
            payload_len: 0,
            fd_count: 0,
        }
        .encode(&mut buf);
        stream.write_all(&buf).context("write hello_ack")?;
        stream.flush().ok();
        Ok(())
    }

    pub fn send_message(&mut self, msg_type: MsgType, payload: &[u8], fds: &[i32]) -> anyhow::Result<()> {
        self.accept_if_needed()?;
        let Some(stream) = &self.stream else {
            anyhow::bail!("no worker connection");
        };

        let mut buf = BytesMut::with_capacity(HeaderV1::LEN + payload.len());
        HeaderV1 {
            msg_type,
            payload_len: payload.len() as u32,
            fd_count: fds.len() as u32,
        }
        .encode(&mut buf);
        buf.extend_from_slice(payload);

        let raw_fd = stream.as_raw_fd();
        let started = Instant::now();
        let mut offset: usize = 0;
        let mut first = true;
        while offset < buf.len() {
            let iov = [IoSlice::new(&buf[offset..])];
            let cmsg_holder = if first && !fds.is_empty() {
                Some(ControlMessage::ScmRights(fds))
            } else {
                None
            };
            let cmsg: &[ControlMessage] = match cmsg_holder.as_ref() {
                Some(msg) => std::slice::from_ref(msg),
                None => &[],
            };
            match sendmsg::<()>(raw_fd, &iov, cmsg, MsgFlags::MSG_NOSIGNAL, None) {
                Ok(n) => {
                    if n == 0 {
                        warn!("sendmsg returned 0 bytes sent (msg_type={msg_type:?}, len={})", buf.len());
                        self.stream = None;
                        return Err(anyhow::anyhow!("sendmsg returned 0 bytes sent"));
                    }
                    if first && n < buf.len() {
                        warn!(
                            "sendmsg partial write (msg_type={msg_type:?}): sent={} total={} fds={}",
                            n,
                            buf.len(),
                            fds.len()
                        );
                    }
                    offset = offset.saturating_add(n);
                    first = false;
                }
                Err(err) => {
                    warn!("sendmsg failed: {err}");
                    self.stream = None;
                    return Err(anyhow::anyhow!("sendmsg failed: {err}"));
                }
            }
        }
        let elapsed = started.elapsed();
        if elapsed > Duration::from_millis(50) {
            warn!(
                "sendmsg slow (msg_type={msg_type:?}): elapsed_ms={:.1} len={} fds={}",
                elapsed.as_secs_f64() * 1000.0,
                buf.len(),
                fds.len()
            );
        }
        Ok(())
    }

    pub fn recv_message_type(&mut self) -> anyhow::Result<MsgType> {
        self.accept_if_needed()?;
        let Some(stream) = self.stream.as_mut() else {
            anyhow::bail!("no worker connection");
        };

        let mut hdr_bytes = [0u8; HeaderV1::LEN];
        if let Err(err) = stream.read_exact(&mut hdr_bytes) {
            self.stream = None;
            return Err(anyhow::anyhow!("read header failed: {err}"));
        }
        let header = HeaderV1::decode(&hdr_bytes).map_err(|e| anyhow::anyhow!("decode header failed: {e}"))?;
        if header.fd_count != 0 {
            warn!(
                msg_type = ?header.msg_type,
                fd_count = header.fd_count,
                "received control message with unexpected fd_count; fds will be dropped"
            );
        }
        let payload_len = header.payload_len as usize;
        if payload_len > 0 {
            // Payload is currently unused for control messages; drain it for framing correctness.
            let mut payload = vec![0u8; payload_len];
            if let Err(err) = stream.read_exact(&mut payload) {
                self.stream = None;
                return Err(anyhow::anyhow!("read payload failed: {err}"));
            }
        }
        Ok(header.msg_type)
    }

    pub fn wait_for_ping(&mut self) -> anyhow::Result<()> {
        loop {
            let msg = self.recv_message_type()?;
            match msg {
                MsgType::Ping => return Ok(()),
                // Ignore other control messages for now.
                _ => continue,
            }
        }
    }
}
