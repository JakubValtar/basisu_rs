use alloc::{format, vec::Vec};
use core::convert::TryFrom;

use byteorder::{ByteOrder, LE};

use crate::{basis_lz, bytereader::ByteReaderLE, uastc, Error, Image, Result};

pub fn read_to_rgba(buf: &[u8]) -> Result<(Header, Vec<Image<u8>>)> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    if header.texture_format()? == TexFormat::ETC1S {
        if header.has_alpha() && (header.total_slices % 2) != 0 {
            return Err("File has alpha, but slice count is odd".into());
        }

        let decoder = make_basis_lz_decoder(&header, buf)?;

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
                    rgb_desc.data(buf),
                    Some(alpha_desc.data(buf)),
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
                    slice_desc.data(buf),
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
                decoder.decode_to_rgba(slice_desc.data(buf), slice_desc.num_blocks_x as usize)?;
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

pub fn read_to_etc1(buf: &[u8]) -> Result<Vec<Image<u8>>> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    let format = header.texture_format()?;
    if format == TexFormat::ETC1S {
        if header.has_alpha() && (header.total_slices % 2) != 0 {
            return Err("File has alpha, but slice count is odd".into());
        }

        let decoder = make_basis_lz_decoder(&header, buf)?;

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode_to_etc1(
                slice_desc.num_blocks_x,
                slice_desc.num_blocks_y,
                slice_desc.data(buf),
            )?;
            let image = Image {
                w: slice_desc.orig_width as u32,
                h: slice_desc.orig_height as u32,
                stride: basis_lz::ETC1S_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
                data,
            };
            images.push(image);
        }
        Ok(images)
    } else if format == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode(uastc::TargetTextureFormat::Etc1, slice_desc.data(buf))?;
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

pub fn read_to_etc2(buf: &[u8]) -> Result<Vec<Image<u8>>> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    let format = header.texture_format()?;
    if format == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode(uastc::TargetTextureFormat::Etc2, slice_desc.data(buf))?;
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

pub fn read_to_uastc(buf: &[u8]) -> Result<Vec<Image<u8>>> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.read_to_uastc(slice_desc.data(buf))?;
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

pub fn read_to_astc(buf: &[u8]) -> Result<Vec<Image<u8>>> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode(uastc::TargetTextureFormat::Astc, slice_desc.data(buf))?;
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

pub fn read_to_bc7(buf: &[u8]) -> Result<Vec<Image<u8>>> {
    let header = read_header(buf)?;

    if !check_file_checksum(buf, &header) {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = read_slice_descs(buf, &header)?;

    if header.texture_format()? == TexFormat::UASTC4x4 {
        let decoder = uastc::Decoder::new();

        let mut images = Vec::with_capacity(header.total_slices as usize);
        for slice_desc in &slice_descs {
            let data = decoder.transcode(uastc::TargetTextureFormat::Bc7, slice_desc.data(buf))?;
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

fn make_basis_lz_decoder(header: &Header, bytes: &[u8]) -> Result<basis_lz::Decoder> {
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

    basis_lz::Decoder::new(
        header.total_selectors,
        header.total_selectors,
        endpoints,
        selectors,
        tables,
        extended,
        is_video,
    )
}

pub const SIG: u16 = 0x4273;

pub fn check_file_sig(bytes: &[u8]) -> bool {
    let sig = LE::read_u16(bytes);
    sig == SIG
}

pub fn read_header(bytes: &[u8]) -> Result<Header> {
    if !check_file_sig(bytes) {
        return Err("Sig mismatch, not a Basis Universal file".into());
    }

    if !Header::check_size(bytes) {
        return Err(format!(
            "Expected at least {} byte header, got {} bytes",
            Header::FILE_SIZE,
            bytes.len()
        ));
    }

    let header = Header::from_file_bytes(bytes);

    if header.header_size as usize != Header::FILE_SIZE {
        return Err(format!(
            "File specified unexpected header size, expected {}, got {}",
            Header::FILE_SIZE,
            header.header_size
        ));
    }

    let header_crc16 = crc16(&bytes[8..Header::FILE_SIZE], 0);
    if header_crc16 != header.header_crc16 {
        return Err("Header CRC16 failed".into());
    }

    Ok(header)
}

pub fn check_file_checksum(bytes: &[u8], header: &Header) -> bool {
    let data_crc16 = crc16(&bytes[Header::FILE_SIZE..], 0);
    data_crc16 == header.data_crc16
}

pub fn read_slice_descs(bytes: &[u8], header: &Header) -> Result<Vec<SliceDesc>> {
    let start = header.slice_desc_file_ofs as usize;
    let count = header.total_slices as usize;
    let mut res = Vec::with_capacity(count);
    for i in 0..count {
        let slice_start = start + i * SliceDesc::FILE_SIZE;
        if !SliceDesc::check_size(&bytes[slice_start..]) {
            let message = format!(
                "Expected {} byte slice desc at pos {}, only {} bytes remain",
                SliceDesc::FILE_SIZE,
                slice_start,
                bytes.len() - slice_start
            );
            return Err(message);
        }
        let slice_desc = SliceDesc::from_file_bytes(&bytes[slice_start..]);
        res.push(slice_desc);
    }
    Ok(res)
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

// basis_file_header::m_tex_type
#[allow(dead_code)]
pub enum TextureType {
    Type2D = 0,
    Type2DArray = 1,
    CubemapArray = 2,
    VideoFrames = 3,
    Volume = 4,
}

// basis_slice_desc::flags
#[allow(dead_code)]
pub enum SliceDescFlags {
    HasAlpha = 1,
    FrameIsIFrame = 2,
}

// basis_file_header::m_tex_format
#[derive(Clone, Copy, PartialEq)]
pub enum TexFormat {
    ETC1S = 0,
    UASTC4x4 = 1,
}

impl TryFrom<u8> for TexFormat {
    type Error = Error;
    fn try_from(v: u8) -> Result<Self> {
        match v {
            0 => Ok(TexFormat::ETC1S),
            1 => Ok(TexFormat::UASTC4x4),
            _ => Err("Unknown texture format".into()),
        }
    }
}

// basis_file_header::m_flags
#[allow(dead_code)]
pub enum HeaderFlags {
    ETC1S = 1,
    YFlipped = 2,
    HasAlphaSlices = 4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[rustfmt::skip]
pub struct Header {
    pub sig: u16,                  // 2 byte file signature
    pub ver: u16,                  // File version
    pub header_size: u16,          // Header size in bytes, sizeof(basis_file_header) or 0x4D
    pub header_crc16: u16,         // CRC16/genibus of the remaining header data

    pub data_size: u32,            // The total size of all data after the header
    pub data_crc16: u16,           // The CRC16 of all data after the header

    pub total_slices: u32, /*24*/  // The number of compressed slices
    pub total_images: u32, /*24*/  // The total # of images

    pub tex_format: u8,            // enum basis_tex_format
    pub flags: u16,                // enum basis_header_flags
    pub tex_type: u8,              // enum basis_texture_type
    pub us_per_frame: u32, /*24*/  // Video: microseconds per frame

    pub reserved: u32,             // For future use
    pub userdata0: u32,            // For client use
    pub userdata1: u32,            // For client use

    pub total_endpoints: u16,               // ETC1S: The number of endpoints in the endpoint codebook
    pub endpoint_cb_file_ofs: u32,          // ETC1S: The compressed endpoint codebook's file offset relative to the header
    pub endpoint_cb_file_size: u32, /*24*/  // ETC1S: The compressed endpoint codebook's size in bytes

    pub total_selectors: u16,               // ETC1S: The number of selectors in the selector codebook
    pub selector_cb_file_ofs: u32,          // ETC1S: The compressed selector codebook's file offset relative to the header
    pub selector_cb_file_size: u32, /*24*/  // ETC1S: The compressed selector codebook's size in bytes

    pub tables_file_ofs: u32,               // ETC1S: The file offset of the compressed Huffman codelength tables.
    pub tables_file_size: u32,              // ETC1S: The file size in bytes of the compressed Huffman codelength tables.

    pub slice_desc_file_ofs: u32,           // The file offset to the slice description array, usually follows the header
    pub extended_file_ofs: u32,             // The file offset of the "extended" header and compressed data, for future use
    pub extended_file_size: u32,            // The file size in bytes of the "extended" header and compressed data, for future use
}

impl Header {
    pub const FILE_SIZE: usize = 77;

    pub fn check_size(buf: &[u8]) -> bool {
        buf.len() >= Self::FILE_SIZE
    }

    pub fn has_alpha(&self) -> bool {
        (self.flags & HeaderFlags::HasAlphaSlices as u16) != 0
    }

    pub fn has_y_flipped(&self) -> bool {
        (self.flags & HeaderFlags::YFlipped as u16) != 0
    }

    pub fn texture_format(&self) -> Result<TexFormat> {
        TexFormat::try_from(self.tex_format)
    }

    pub fn from_file_bytes(buf: &[u8]) -> Self {
        assert!(Self::check_size(buf));
        let mut r = ByteReaderLE::new(buf);
        let res = Self {
            sig: r.read_u16(),
            ver: r.read_u16(),
            header_size: r.read_u16(),
            header_crc16: r.read_u16(),

            data_size: r.read_u32(),
            data_crc16: r.read_u16(),

            total_slices: r.read_u24(),
            total_images: r.read_u24(),

            tex_format: r.read_u8(),
            flags: r.read_u16(),
            tex_type: r.read_u8(),
            us_per_frame: r.read_u24(),

            reserved: r.read_u32(),
            userdata0: r.read_u32(),
            userdata1: r.read_u32(),

            total_endpoints: r.read_u16(),
            endpoint_cb_file_ofs: r.read_u32(),
            endpoint_cb_file_size: r.read_u24(),

            total_selectors: r.read_u16(),
            selector_cb_file_ofs: r.read_u32(),
            selector_cb_file_size: r.read_u24(),

            tables_file_ofs: r.read_u32(),
            tables_file_size: r.read_u32(),

            slice_desc_file_ofs: r.read_u32(),
            extended_file_ofs: r.read_u32(),
            extended_file_size: r.read_u32(),
        };
        assert_eq!(r.pos(), Self::FILE_SIZE);
        res
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceDesc {
    pub image_index: u32, /*24*/
    pub level_index: u8,
    pub flags: u8,

    pub orig_width: u16,
    pub orig_height: u16,

    pub num_blocks_x: u16,
    pub num_blocks_y: u16,

    pub file_ofs: u32,
    pub file_size: u32,

    pub slice_data_crc16: u16,
}

impl SliceDesc {
    pub const FILE_SIZE: usize = 23;

    pub fn check_size(buf: &[u8]) -> bool {
        buf.len() >= Self::FILE_SIZE
    }

    pub fn has_alpha(&self) -> bool {
        (self.flags & SliceDescFlags::HasAlpha as u8) != 0
    }

    pub fn data<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        let start = self.file_ofs as usize;
        let len = self.file_size as usize;
        &buf[start..start + len]
    }

    pub fn from_file_bytes(buf: &[u8]) -> Self {
        assert!(Self::check_size(buf));
        let mut r = ByteReaderLE::new(buf);
        let res = Self {
            image_index: r.read_u24(),
            level_index: r.read_u8(),
            flags: r.read_u8(),
            orig_width: r.read_u16(),
            orig_height: r.read_u16(),
            num_blocks_x: r.read_u16(),
            num_blocks_y: r.read_u16(),
            file_ofs: r.read_u32(),
            file_size: r.read_u32(),
            slice_data_crc16: r.read_u16(),
        };
        assert_eq!(r.pos(), Self::FILE_SIZE);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_header() {
        let bytes: Vec<u8> = (0..Header::FILE_SIZE as u8).collect();
        let actual = Header::from_file_bytes(&bytes);
        let expected = Header {
            sig: LE::read_u16(&[0, 1]),
            ver: LE::read_u16(&[2, 3]),
            header_size: LE::read_u16(&[4, 5]),
            header_crc16: LE::read_u16(&[6, 7]),

            data_size: LE::read_u32(&[8, 9, 10, 11]),
            data_crc16: LE::read_u16(&[12, 13]),

            total_slices: LE::read_u24(&[14, 15, 16]),
            total_images: LE::read_u24(&[17, 18, 19]),

            tex_format: 20,
            flags: LE::read_u16(&[21, 22]),
            tex_type: 23,
            us_per_frame: LE::read_u24(&[24, 25, 26]),

            reserved: LE::read_u32(&[27, 28, 29, 30]),
            userdata0: LE::read_u32(&[31, 32, 33, 34]),
            userdata1: LE::read_u32(&[35, 36, 37, 38]),

            total_endpoints: LE::read_u16(&[39, 40]),
            endpoint_cb_file_ofs: LE::read_u32(&[41, 42, 43, 44]),
            endpoint_cb_file_size: LE::read_u24(&[45, 46, 47]),

            total_selectors: LE::read_u16(&[48, 49]),
            selector_cb_file_ofs: LE::read_u32(&[50, 51, 52, 53]),
            selector_cb_file_size: LE::read_u24(&[54, 55, 56]),

            tables_file_ofs: LE::read_u32(&[57, 58, 59, 60]),
            tables_file_size: LE::read_u32(&[61, 62, 63, 64]),

            slice_desc_file_ofs: LE::read_u32(&[65, 66, 67, 68]),
            extended_file_ofs: LE::read_u32(&[69, 70, 71, 72]),
            extended_file_size: LE::read_u32(&[73, 74, 75, 76]),
        };

        assert_eq!(actual, expected);
    }
}
