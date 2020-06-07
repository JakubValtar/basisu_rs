#![warn(clippy::all)]

use byteorder::{ByteOrder, LE};
use std::ops::{Index, IndexMut};
use std::path::Path;

mod huffman;
mod bitreader;
mod etc1s;
mod basis;

use basis::{
    Header,
    HeaderFlags,
    SliceDesc,
    SliceDescFlags,
    TexFormat,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    if !basis::check_file_sig(&buf) {
        return Err("Sig mismatch, not a Basis Universal file".into());
    }

    if !Header::check_size(&buf) {
        return Err(format!(
            "Expected at least {} byte header, got {} bytes",
            Header::FILE_SIZE, buf.len()).into()
        );
    }

    let header = Header::from_bytes(&buf);

    if header.header_size as usize != Header::FILE_SIZE {
        return Err(format!(
            "File specified unexpected header size, expected {}, got {}",
            Header::FILE_SIZE, header.header_size).into()
        );
    }

    let header_crc16 = crc16(&buf[8..Header::FILE_SIZE], 0);
    if header_crc16 != header.header_crc16 {
        return Err("Header CRC16 failed".into());
    }

    let data_crc16 = crc16(&buf[Header::FILE_SIZE..], 0);
    if data_crc16 != header.data_crc16 {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = {
        let start = header.slice_desc_file_ofs as usize;
        let count = header.total_slices as usize;
        let mut res = Vec::with_capacity(count);
        for i in 0..count {
            let slice_start = start + i * SliceDesc::FILE_SIZE;
            if !SliceDesc::check_size(&buf[slice_start..]) {
                let message = format!(
                    "Expected {} byte slice desc at pos {}, only {} bytes remain",
                    SliceDesc::FILE_SIZE, slice_start, buf.len()-slice_start
                );
                return Err(message.into());
            }
            let slice_desc = SliceDesc::from_bytes(&buf[slice_start..]);
            res.push(slice_desc);
        }
        res
    };

    if header.tex_format == TexFormat::ETC1S as u8 {

        let decoder = etc1s::Etc1sDecoder::from_file_bytes(&header, &buf)?;

        let has_alpha = (header.flags & HeaderFlags::HasAlphaSlices as u16) != 0;
        let slices_per_image = if has_alpha { 2 } else { 1 };

        assert_eq!(header.total_slices as usize % slices_per_image, 0);

        let slice_count = header.total_slices as usize / slices_per_image;

        let mut images = Vec::with_capacity(slice_count);

        if has_alpha {
            for slice_desc in slice_descs.chunks_exact(2) {
                assert_ne!(slice_desc[1].flags | SliceDescFlags::HasAlpha as u8, 0);
                let mut rgb = decoder.decode_slice(&slice_desc[0], &buf)?;
                let alpha = decoder.decode_slice(&slice_desc[1], &buf)?;
                for (rgb, alpha) in rgb.data.iter_mut().zip(alpha.data.iter()) {
                    rgb.0[3] = alpha.0[1];
                }
                images.push(rgb.into_rgba_bytes());
            }
        } else {
            for rgb_desc in &slice_descs {
                let rgb = decoder.decode_slice(rgb_desc, &buf)?;
                images.push(rgb.into_rgba_bytes());
            }
        }

        return Ok(images);
    } else {
        unimplemented!();
    }
}

fn crc16(r: &[u8], mut crc: u16) -> u16 {
  crc = !crc;
  for &b in r {
    let q: u16 = (b as u16) ^ (crc >> 8);
    let k: u16 = (q >> 4) ^ q;
    crc = (((crc << 8) ^ k) ^ (k << 5)) ^ (k << 12);
  }
  !crc
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
    pub data: Vec<T>,
}

impl Image<Color32> {
    pub fn into_rgba_bytes(self) -> Image<u8> {
        Image {
            w: self.w,
            h: self.h,
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
