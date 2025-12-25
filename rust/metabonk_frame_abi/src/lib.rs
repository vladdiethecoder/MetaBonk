use bytes::{Buf, BufMut, BytesMut};

pub const MAGIC: &[u8; 8] = b"MBEYEABI";
pub const VERSION: u16 = 1;

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsgType {
    Hello = 1,
    HelloAck = 2,
    Frame = 3,
    Reset = 4,
    Ping = 5,
    Pong = 6,
}

#[derive(thiserror::Error, Debug)]
pub enum AbiError {
    #[error("bad magic")]
    BadMagic,
    #[error("unsupported version {0}")]
    UnsupportedVersion(u16),
    #[error("truncated header")]
    TruncatedHeader,
    #[error("truncated payload")]
    TruncatedPayload,
    #[error("invalid msg type {0}")]
    InvalidMsgType(u16),
    #[error("invalid frame fd_count (expected {expected}, got {got})")]
    FrameFdCount { expected: usize, got: usize },
    #[error("invalid plane fd_index {0}")]
    PlaneFdIndex(u8),
}

#[derive(Clone, Debug)]
pub struct HeaderV1 {
    pub msg_type: MsgType,
    pub payload_len: u32,
    pub fd_count: u32,
}

impl HeaderV1 {
    pub const LEN: usize = 8 + 2 + 2 + 4 + 4;

    pub fn encode(&self, out: &mut BytesMut) {
        out.put_slice(MAGIC);
        out.put_u16_le(VERSION);
        out.put_u16_le(self.msg_type as u16);
        out.put_u32_le(self.payload_len);
        out.put_u32_le(self.fd_count);
    }

    pub fn decode(mut bytes: &[u8]) -> Result<Self, AbiError> {
        if bytes.len() < Self::LEN {
            return Err(AbiError::TruncatedHeader);
        }
        let mut magic = [0u8; 8];
        bytes.copy_to_slice(&mut magic);
        if &magic != MAGIC {
            return Err(AbiError::BadMagic);
        }
        let ver = bytes.get_u16_le();
        if ver != VERSION {
            return Err(AbiError::UnsupportedVersion(ver));
        }
        let msg_type_u = bytes.get_u16_le();
        let msg_type = match msg_type_u {
            1 => MsgType::Hello,
            2 => MsgType::HelloAck,
            3 => MsgType::Frame,
            4 => MsgType::Reset,
            5 => MsgType::Ping,
            6 => MsgType::Pong,
            other => return Err(AbiError::InvalidMsgType(other)),
        };
        let payload_len = bytes.get_u32_le();
        let fd_count = bytes.get_u32_le();
        Ok(Self {
            msg_type,
            payload_len,
            fd_count,
        })
    }
}

#[derive(Clone, Debug)]
pub struct PlaneV1 {
    pub fd_index: u8,
    pub stride: u32,
    pub offset: u32,
    pub size_bytes: u32,
}

#[derive(Clone, Debug)]
pub struct FrameV1 {
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub modifier: u64,
    pub dmabuf_fd_count: u8,
    pub planes: Vec<PlaneV1>,
}

impl FrameV1 {
    pub fn encode_payload(&self) -> BytesMut {
        let mut out = BytesMut::with_capacity(32 + self.planes.len() * 24);
        out.put_u64_le(self.frame_id);
        out.put_u32_le(self.width);
        out.put_u32_le(self.height);
        out.put_u32_le(self.drm_fourcc);
        out.put_u64_le(self.modifier);
        out.put_u8(self.dmabuf_fd_count);
        out.put_u8(self.planes.len() as u8);
        out.put_u16_le(0);
        for p in &self.planes {
            out.put_u8(p.fd_index);
            out.put_u8(0);
            out.put_u16_le(0);
            out.put_u32_le(p.stride);
            out.put_u32_le(p.offset);
            out.put_u32_le(p.size_bytes);
            out.put_u32_le(0);
        }
        out
    }

    pub fn decode_payload(mut payload: &[u8], fd_count: usize) -> Result<Self, AbiError> {
        if payload.len() < 32 {
            return Err(AbiError::TruncatedPayload);
        }
        let frame_id = payload.get_u64_le();
        let width = payload.get_u32_le();
        let height = payload.get_u32_le();
        let drm_fourcc = payload.get_u32_le();
        let modifier = payload.get_u64_le();
        let dmabuf_fd_count = payload.get_u8();
        let plane_count = payload.get_u8();
        let _reserved0 = payload.get_u16_le();

        let min_expected = dmabuf_fd_count as usize;
        let max_expected = dmabuf_fd_count as usize + 2;
        if fd_count < min_expected || fd_count > max_expected {
            return Err(AbiError::FrameFdCount {
                expected: max_expected,
                got: fd_count,
            });
        }

        let mut planes = Vec::with_capacity(plane_count as usize);
        for _ in 0..plane_count {
            if payload.len() < 24 {
                return Err(AbiError::TruncatedPayload);
            }
            let fd_index = payload.get_u8();
            let _r0 = payload.get_u8();
            let _r1 = payload.get_u16_le();
            let stride = payload.get_u32_le();
            let offset = payload.get_u32_le();
            let size_bytes = payload.get_u32_le();
            let _r2 = payload.get_u32_le();
            if fd_index as usize >= dmabuf_fd_count as usize {
                return Err(AbiError::PlaneFdIndex(fd_index));
            }
            planes.push(PlaneV1 {
                fd_index,
                stride,
                offset,
                size_bytes,
            });
        }
        Ok(Self {
            frame_id,
            width,
            height,
            drm_fourcc,
            modifier,
            dmabuf_fd_count,
            planes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResetV1 {
    pub reason: u32,
}

impl ResetV1 {
    pub fn encode_payload(&self) -> BytesMut {
        let mut out = BytesMut::with_capacity(8);
        out.put_u32_le(self.reason);
        out.put_u32_le(0);
        out
    }
}
