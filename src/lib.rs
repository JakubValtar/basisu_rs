#![forbid(unsafe_code)]

use std::fmt;
use std::ops::{Index, IndexMut};
use std::path::Path;

mod astc;
mod basis;
mod bc7;
mod bitreader;
mod bitwriter;
mod bytereader;
mod etc;
mod etc1s;
mod huffman;
mod uastc;

use basis::{Header, TexFormat, TextureType};

type Error = Box<dyn std::error::Error>;
type Result<T> = std::result::Result<T, Error>;

pub fn read_to_rgba<P: AsRef<Path>>(path: P) -> Result<(Header, Vec<Image<u8>>)> {
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

        let decoder = make_etc1s_decoder(&header, &buf)?;

        if header.has_alpha() {
            let mut images = Vec::with_capacity(header.total_slices as usize / 2);
            for slice_desc in slice_descs.chunks_exact(2) {
                let rgb_desc = &slice_desc[0];
                let alpha_desc = &slice_desc[1];
                if !alpha_desc.has_alpha() {
                    return Err("Expected slice with alpha".into());
                }
                if alpha_desc.num_blocks_x != rgb_desc.num_blocks_x
                    || alpha_desc.num_blocks_y != rgb_desc.num_blocks_y
                {
                    return Err("RGB slice and Alpha slice have different dimensions".into());
                }
                let data = decoder.decode_to_rgba(
                    rgb_desc.num_blocks_x,
                    rgb_desc.num_blocks_y,
                    rgb_desc.data(&buf),
                    Some(alpha_desc.data(&buf)),
                )?;
                let image = Image {
                    w: rgb_desc.orig_width as u32,
                    h: rgb_desc.orig_height as u32,
                    stride: 4 * rgb_desc.orig_width as u32,
                    data,
                };
                images.push(image.into_rgba_bytes());
            }
            Ok((header, images))
        } else {
            let mut images = Vec::with_capacity(header.total_slices as usize);
            for slice_desc in &slice_descs {
                let data = decoder.decode_to_rgba(
                    slice_desc.num_blocks_x,
                    slice_desc.num_blocks_y,
                    slice_desc.data(&buf),
                    None,
                )?;
                let image = Image {
                    w: slice_desc.orig_width as u32,
                    h: slice_desc.orig_height as u32,
                    stride: 4 * slice_desc.orig_width as u32,
                    data,
                };
                images.push(image.into_rgba_bytes());
            }
            Ok((header, images))
        }
    } else if header.texture_format()? == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data =
                decoder.decode_to_rgba(slice_desc.data(&buf), slice_desc.num_blocks_x as usize)?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: 4 * slice_desc.num_blocks_x as u32,
                data,
            };
            images.push(image.into_rgba_bytes());
        }
        Ok((header, images))
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

        let decoder = make_etc1s_decoder(&header, &buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode_to_etc1(
                slice_desc.num_blocks_x,
                slice_desc.num_blocks_y,
                slice_desc.data(&buf),
            )?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: etc1s::BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
            images.push(image);
        }
        Ok(images)
    } else if format == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data =
                decoder.transcode(uastc::TargetTextureFormat::Etc1, slice_desc.data(&buf))?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: uastc::ETC1_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
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
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data =
                decoder.transcode(uastc::TargetTextureFormat::Etc2, slice_desc.data(&buf))?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: uastc::ETC2_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
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
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.read_to_uastc(slice_desc.data(&buf))?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: uastc::UASTC_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
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
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data =
                decoder.transcode(uastc::TargetTextureFormat::Astc, slice_desc.data(&buf))?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: uastc::ASTC_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
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
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode(uastc::TargetTextureFormat::Bc7, slice_desc.data(&buf))?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: uastc::BC7_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
            images.push(image);
        }
        Ok(images)
    } else {
        unimplemented!();
    }
}

fn make_etc1s_decoder(header: &Header, bytes: &[u8]) -> Result<etc1s::Decoder> {
    let endpoints = {
        let start = header.endpoint_cb_file_ofs as usize;
        let len = header.endpoint_cb_file_size as usize;
        &bytes[start..start + len]
    };

    let selectors = {
        let start = header.selector_cb_file_ofs as usize;
        let len = header.selector_cb_file_size as usize;
        &bytes[start..start + len]
    };

    let tables = {
        let start = header.tables_file_ofs as usize;
        let len = header.tables_file_size as usize;
        &bytes[start..start + len]
    };

    let extended = {
        let start = header.extended_file_ofs as usize;
        let len = header.extended_file_size as usize;
        &bytes[start..start + len]
    };

    let is_video = header.tex_type == TextureType::VideoFrames as u8;

    etc1s::Decoder::new(
        header.total_selectors,
        header.total_selectors,
        endpoints,
        selectors,
        tables,
        extended,
        is_video,
    )
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

    pub fn to_rgba_u32(self) -> u32 {
        u32::from_le_bytes(self.0)
    }

    #[allow(dead_code)]
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
