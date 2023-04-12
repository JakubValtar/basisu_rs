#![forbid(unsafe_code)]
#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::vec::Vec;

mod basis;
mod basis_lz;
mod bitreader;
mod bitwriter;
mod bytereader;
mod color;
mod target_formats;
mod uastc;

pub use basis::{
    read_to_astc, read_to_bc7, read_to_etc1, read_to_etc2, read_to_rgba, read_to_uastc, Header,
};
use color::Color32;
use uastc::{ASTC_BLOCK_SIZE, BC7_BLOCK_SIZE, ETC1_BLOCK_SIZE, ETC2_BLOCK_SIZE, UASTC_BLOCK_SIZE};

type Error = alloc::string::String;
type Result<T> = core::result::Result<T, Error>;

pub fn unpack_uastc_block_to_rgba(data: [u8; UASTC_BLOCK_SIZE]) -> Result<[u32; 16]> {
    uastc::decode_block_to_rgba(data).map(|b| b.map(|c| c.to_rgba_u32()))
}

pub fn transcode_uastc_block_to_astc(
    data: [u8; UASTC_BLOCK_SIZE],
) -> Result<[u8; ASTC_BLOCK_SIZE]> {
    target_formats::astc::convert_block_from_uastc(data)
}

pub fn transcode_uastc_block_to_bc7(data: [u8; UASTC_BLOCK_SIZE]) -> Result<[u8; BC7_BLOCK_SIZE]> {
    target_formats::bc7::convert_block_from_uastc(data)
}

pub fn transcode_uastc_block_to_etc1(
    data: [u8; UASTC_BLOCK_SIZE],
) -> Result<[u8; ETC1_BLOCK_SIZE]> {
    target_formats::etc::convert_etc1_block_from_uastc(data)
}

pub fn transcode_uastc_block_to_etc2(
    data: [u8; UASTC_BLOCK_SIZE],
) -> Result<[u8; ETC2_BLOCK_SIZE]> {
    target_formats::etc::convert_etc2_block_from_uastc(data)
}

#[doc(hidden)]
#[macro_export]
macro_rules! mask {
    ($size:expr) => {
        !(!($size ^ $size)).checked_shl($size as u32).unwrap_or(0)
    };
}

pub struct Image<T> {
    pub w: u32,
    pub h: u32,
    pub stride: u32,
    pub data: Vec<T>,
}

impl Image<Color32> {
    pub fn into_rgba_bytes(self) -> Image<u8> {
        Image {
            w: self.w,
            h: self.h,
            stride: self.stride * 4,
            data: Color32::into_rgba_bytes(self.data),
        }
    }
}
