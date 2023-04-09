#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

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

type Error = alloc::string::String;
type Result<T> = core::result::Result<T, Error>;

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
