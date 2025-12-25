use std::{
    fs,
    io::{Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    os::unix::prelude::AsRawFd,
    path::Path,
};

use anyhow::Context;
use bytes::BytesMut;
use metabonk_frame_abi::{HeaderV1, MsgType, MAGIC, VERSION};
use std::io::IoSlice;

use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags};
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
        let mut hdr = [0u8; HeaderV1::LEN];
        // Best-effort: peek a header without blocking too long. If no HELLO is sent, proceed.
        stream
            .set_read_timeout(Some(std::time::Duration::from_millis(10)))
            .ok();
        let n = match stream.read(&mut hdr) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(_) => {
                stream.set_read_timeout(None).ok();
                return Ok(());
            }
        };
        stream.set_read_timeout(None).ok();
        if n < HeaderV1::LEN {
            return Ok(());
        }
        if &hdr[0..8] != &MAGIC[..] || u16::from_le_bytes([hdr[8], hdr[9]]) != VERSION {
            return Ok(());
        }
        let msg_type = u16::from_le_bytes([hdr[10], hdr[11]]);
        if msg_type == MsgType::Hello as u16 {
            debug!("received HELLO");
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

        let iov = [IoSlice::new(&buf)];
        let cmsg = if fds.is_empty() {
            vec![]
        } else {
            vec![ControlMessage::ScmRights(fds)]
        };

        let raw_fd = stream.as_raw_fd();
        match sendmsg::<()>(raw_fd, &iov, &cmsg, MsgFlags::MSG_NOSIGNAL, None) {
            Ok(_) => Ok(()),
            Err(err) => {
                warn!("sendmsg failed: {err}");
                self.stream = None;
                Err(anyhow::anyhow!("sendmsg failed: {err}"))
            }
        }
    }
}
