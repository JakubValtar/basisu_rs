#![warn(clippy::all)]

use std::ops::{Index, IndexMut};
use std::path::Path;

mod huffman;
mod bitreader;
mod bytereader;
mod etc1s;
mod basis;

use basis::{
    TexFormat,
};

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    if header.texture_format()? == TexFormat::ETC1S {
        if header.has_alpha() && (header.total_slices % 2) != 0 {
            return Err("File has alpha, but slice count is odd".into());
        }

        let decoder = etc1s::Decoder::from_file_bytes(&header, &buf)?;

        if header.has_alpha() {
            let mut images = Vec::with_capacity(header.total_slices as usize / 2);
            for slice_desc in slice_descs.chunks_exact(2) {
                let image = decoder.decode_rgba_slice(&slice_desc[0], &slice_desc[1], &buf)?;
                images.push(image.into_rgba_bytes());
            }
            return Ok(images);
        } else {
            let mut images = Vec::with_capacity(header.total_slices as usize);
            for slice_desc in &slice_descs {
                let image = decoder.decode_rgb_slice(slice_desc, &buf)?;
                images.push(image.into_rgba_bytes());
            }
            return Ok(images);
        }
    } else {
        unimplemented!();
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! mask {
    ($size:expr) => {
        !(!($size ^ $size)).checked_shl($size as u32).unwrap_or(0)
    }
}


pub struct Image<T> {
    pub w: u32,
    pub h: u32,
    pub stride: u32,
    pub pixel_stride: u32,
    pub y_flipped: bool,
    pub data: Vec<T>,
}

impl<T> Image<T> {
    pub fn rows<'a>(&'a self) -> Box<dyn Iterator<Item=&'a [T]> + 'a> {
        // TODO: Is texture with y flipped aligned to the top or to the bottom? This code assumes to the top.
        let res = self.data
            .chunks_exact(self.stride as usize)
            .take(self.h as usize)
            .map(move |r| &r[0..(self.w * self.pixel_stride) as usize]);

        if self.y_flipped {
            Box::new(res.rev())
        } else {
            Box::new(res)
        }
    }
}

impl Image<Color32> {
    pub fn into_rgba_bytes(self) -> Image<u8> {
        Image {
            w: self.w,
            h: self.h,
            stride: self.stride * 4,
            pixel_stride: 4,
            y_flipped: self.y_flipped,
            data: Color32::as_rgba_bytes(self.data),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Color32([u8; 4]);

impl Color32 {
    pub fn new(r: u8, b: u8, g: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub fn as_rgba_bytes(data: Vec<Self>) -> Vec<u8> {
        let len = data.len();
        unsafe {
            let mut bytes: Vec<u8> = std::mem::transmute(data);
            bytes.set_len(4 * len);
            bytes
        }
    }
}

impl Index<usize> for Color32 {
    type Output = u8;
    fn index<'a>(&'a self, i: usize) -> &'a Self::Output {
        &self.0[i]
    }
}

impl IndexMut<usize> for Color32 {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Self::Output {
        &mut self.0[i]
    }
}
