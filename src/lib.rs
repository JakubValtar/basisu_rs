#![forbid(unsafe_code)]

use std::fmt;
use std::ops::{Index, IndexMut};
use std::path::Path;

mod huffman;
mod bitreader;
mod bitwriter;
mod bytereader;
mod etc1s;
mod uastc;
mod basis;
mod bc7;
mod astc;
mod etc;

use basis::{
    TexFormat,
};

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

pub fn read_to_rgba<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
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
                let image = decoder.decode_to_rgba(&slice_desc[0], Some(&slice_desc[1]), &buf)?;
                images.push(image.into_rgba_bytes());
            }
            Ok(images)
        } else {
            let mut images = Vec::with_capacity(header.total_slices as usize);
            for slice_desc in &slice_descs {
                let image = decoder.decode_to_rgba(slice_desc, None, &buf)?;
                images.push(image.into_rgba_bytes());
            }
            Ok(images)
        }
    } else if header.texture_format()? == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
            for slice_desc in &slice_descs {
                let image = decoder.decode_to_rgba(slice_desc, &buf)?;
                images.push(image.into_rgba_bytes());
            }
            Ok(images)
    } else {
        unimplemented!();
    }
}

pub fn read_to_etc1<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    let format = header.texture_format()?;
    if format == TexFormat::ETC1S {
        if header.has_alpha() && (header.total_slices % 2) != 0 {
            return Err("File has alpha, but slice count is odd".into());
        }

        let decoder = etc1s::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.transcode_to_etc1(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
    } else if format == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.transcode_to_etc1(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
    } else {
        unimplemented!();
    }
}

pub fn read_to_etc2<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    let format = header.texture_format()?;
    if format == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.transcode_to_etc2(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
    } else {
        unimplemented!();
    }
}

pub fn read_to_uastc<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {

        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.read_to_uastc(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
    } else {
        unimplemented!();
    }
}

pub fn read_to_astc<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {

        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.transcode_to_astc(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
    } else {
        unimplemented!();
    }
}

pub fn read_to_bc7<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    let header = basis::read_header(&buf)?;

    if !basis::check_file_checksum(&buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = basis::read_slice_descs(&buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {

        let decoder = uastc::Decoder::from_file_bytes(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let image = decoder.transcode_to_bc7(slice_desc, &buf)?;
            images.push(image);
        }
        Ok(images)
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
    pub y_flipped: bool,
    pub data: Vec<T>,
}

impl Image<Color32> {
    pub fn into_rgba_bytes(self) -> Image<u8> {
        Image {
            w: self.w,
            h: self.h,
            stride: self.stride * 4,
            y_flipped: self.y_flipped,
            data: Color32::into_rgba_bytes(self.data),
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Color32([u8; 4]);

impl Color32 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    pub fn into_rgba_bytes(data: Vec<Self>) -> Vec<u8> {
        let mut result = vec![0u8; data.len() * 4];

        for (chunk, color) in result.chunks_exact_mut(4).zip(data.into_iter()) {
            chunk.copy_from_slice(&color.0);
        }

        result
    }

    pub fn to_rgba_u32(&self) -> u32 {
        u32::from_le_bytes(self.0)
    }

    pub fn from_rgba_u32(rgba: u32) -> Self {
        Color32(rgba.to_le_bytes())
    }
}

impl fmt::Debug for Color32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:08X}", self.to_rgba_u32())
    }
}

impl Index<usize> for Color32 {
    type Output = u8;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl IndexMut<usize> for Color32 {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}
