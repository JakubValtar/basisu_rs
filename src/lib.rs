#![warn(clippy::all)]

use byteorder::{ByteOrder, LE};
use std::ops::{Index, IndexMut};
use std::path::Path;

mod huffman;
mod bitreader;
mod etc1s;

use bitreader::BitReaderLSB;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check_sig(buf: &[u8]) -> bool{
    let sig = LE::read_u16(&buf);
    sig == BASIS_SIG
}

pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Vec<Image<u8>>> {
    let buf = std::fs::read(path)?;

    if !check_sig(&buf) {
        return Err("Sig mismatch, not a Basis Universal file".into());
    }

    if !BasisFileHeader::check_size(&buf) {
        return Err(format!(
            "Expected at least {} byte header, got {} bytes",
            BasisFileHeader::FILE_SIZE, buf.len()).into()
        );
    }

    let header = BasisFileHeader::from_bytes(&buf);

    if header.header_size as usize != BasisFileHeader::FILE_SIZE {
        return Err(format!(
            "File specified unexpected header size, expected {}, got {}",
            BasisFileHeader::FILE_SIZE, header.header_size).into()
        );
    }

    let header_crc16 = crc16(&buf[8..BasisFileHeader::FILE_SIZE], 0);
    if header_crc16 != header.header_crc16 {
        return Err("Header CRC16 failed".into());
    }

    let data_crc16 = crc16(&buf[BasisFileHeader::FILE_SIZE..], 0);
    if data_crc16 != header.data_crc16 {
        return Err("Data CRC16 failed".into());
    }

    let slice_descs = {
        let start = header.slice_desc_file_ofs as usize;
        let count = header.total_slices as usize;
        let mut res = Vec::with_capacity(count);
        for i in 0..count {
            let slice_start = start + i * BasisSliceDesc::FILE_SIZE;
            if !BasisSliceDesc::check_size(&buf[slice_start..]) {
                let message = format!(
                    "Expected {} byte slice desc at pos {}, only {} bytes remain",
                    BasisSliceDesc::FILE_SIZE, slice_start, buf.len()-slice_start
                );
                return Err(message.into());
            }
            let slice_desc = BasisSliceDesc::from_bytes(&buf[slice_start..]);
            res.push(slice_desc);
        }
        res
    };

    if header.tex_format == BasisTexFormat::ETC1S as u8 {

        let decoder = etc1s::Etc1sDecoder::from_file_bytes(&header, &buf)?;

        let has_alpha = (header.flags & BasisHeaderFlags::HasAlphaSlices as u16) != 0;
        let slices_per_image = if has_alpha { 2 } else { 1 };

        assert_eq!(header.total_slices as usize % slices_per_image, 0);

        let slice_count = header.total_slices as usize / slices_per_image;

        let mut images = Vec::with_capacity(slice_count);

        if has_alpha {
            for slice_desc in slice_descs.chunks_exact(2) {
                assert_ne!(slice_desc[1].flags | BasisSliceDescFlags::HasAlpha as u8, 0);
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

fn read_endpoints(num_endpoints: usize, bytes: &[u8]) -> Result<Vec<Endpoint>> {
    let reader = &mut BitReaderLSB::new(bytes);

    let color5_delta_model0 = huffman::read_huffman_table(reader)?;
    let color5_delta_model1 = huffman::read_huffman_table(reader)?;
    let color5_delta_model2 = huffman::read_huffman_table(reader)?;
    let inten_delta_model = huffman::read_huffman_table(reader)?;
    let grayscale = reader.read(1) == 1;

    // const int COLOR5_PAL0_PREV_HI = 9, COLOR5_PAL0_DELTA_LO = -9, COLOR5_PAL0_DELTA_HI = 31;
    // const int COLOR5_PAL1_PREV_HI = 21, COLOR5_PAL1_DELTA_LO = -21, COLOR5_PAL1_DELTA_HI = 21;
    // const int COLOR5_PAL2_PREV_HI = 31, COLOR5_PAL2_DELTA_LO = -31, COLOR5_PAL2_DELTA_HI = 9;

    const COLOR5_PAL0_PREV_HI: i32 = 9;
    const COLOR5_PAL0_DELTA_LO: i32 = -9;
    const COLOR5_PAL0_DELTA_HI: i32 = 31;
    const COLOR5_PAL1_PREV_HI: i32 = 21;
    const COLOR5_PAL1_DELTA_LO: i32 = -21;
    const COLOR5_PAL1_DELTA_HI: i32 = 21;
    const COLOR5_PAL2_PREV_HI: i32 = 31;
    const COLOR5_PAL2_DELTA_LO: i32 = -31;
    const COLOR5_PAL2_DELTA_HI: i32 = 9;

    // Assume previous endpoint color is (16, 16, 16), and the previous intensity is 0.
    let mut prev_color5 = Color32::new(16, 16, 16, 0);
    let mut prev_inten: u32 = 0;

    let mut endpoints: Vec<Endpoint> = vec![Endpoint::default(); num_endpoints as usize];

    // For each endpoint codebook entry
    for i in 0..num_endpoints {

        let endpoint = &mut endpoints[i];

        // Decode the intensity delta Huffman code
        let inten_delta = inten_delta_model.decode_symbol(reader)?;
        endpoint.inten5 = ((inten_delta as u32 + prev_inten) & 7) as u8;
        prev_inten = endpoint.inten5 as u32;

        // Now decode the endpoint entry's color or intensity value
        let channel_count = if grayscale { 1 } else { 3 };
        for c in 0..channel_count {

            // The Huffman table used to decode the delta depends on the previous color's value
            let delta = match prev_color5[c as usize] as i32 {
                i if i <= COLOR5_PAL0_PREV_HI => color5_delta_model0.decode_symbol(reader),
                i if i <= COLOR5_PAL1_PREV_HI => color5_delta_model1.decode_symbol(reader),
                _ => color5_delta_model2.decode_symbol(reader),
            }?;

            // Apply the delta
            let v = (prev_color5[c] as u32 + delta as u32) & 31;

            endpoint.color5[c] = v as u8;

            prev_color5[c] = v as u8;
        }

        // If the endpoints are grayscale, set G and B to match R.
        if grayscale {
            endpoint.color5[1] = endpoint.color5[0];
            endpoint.color5[2] = endpoint.color5[0];
        }
    }

    Ok(endpoints)
}

fn read_selectors(num_selectors: usize, bytes: &[u8]) -> Result<Vec<Selector>> {
    let reader = &mut BitReaderLSB::new(bytes);

    let global = reader.read(1) == 1;
    let hybrid = reader.read(1) == 1;
    let raw = reader.read(1) == 1;

    if global {
        return Err("Global selector codebooks are not supported".into());
    }

    if hybrid {
        return Err("Hybrid selector codebooks are not supported".into());
    }

    let mut selectors = vec![Selector::default(); num_selectors];

    if !raw {
        let delta_selector_pal_model = huffman::read_huffman_table(reader)?;

        let mut prev_bytes = [0u8; 4];

        for i in 0..num_selectors {
            if i == 0 {
                // First selector is sent raw
                for y in 0..4 {
                    let cur_byte = reader.read(8) as u8;
                    prev_bytes[y] = cur_byte;

                    for x in 0..4 {
                        selectors[i].set_selector(x, y, (cur_byte >> (x*2)) & 3);
                    }
                }
                continue;
            }

            // Subsequent selectors are sent with a simple form of byte-wise DPCM coding.
            for y in 0..4 {
                let delta_byte = delta_selector_pal_model.decode_symbol(reader)? as u8;

                let cur_byte = delta_byte ^ prev_bytes[y];
                prev_bytes[y] = cur_byte;

                for x in 0..4 {
                    selectors[i].set_selector(x, y, (cur_byte >> (x*2)) & 3);
                }
            }
        }
    } else {
        for i in 0..num_selectors {
            for y in 0..4 {
                let cur_byte = reader.read(8) as u8;
                for x in 0..4 {
                    selectors[i].set_selector(x, y, (cur_byte >> (x*2)) & 3);
                }
            }
        }
    }

    Ok(selectors)
}

fn decode_vlc(reader: &mut BitReaderLSB, chunk_bits: u32) -> u32 {
    assert!(chunk_bits > 0);
    let chunk_size = 1 << chunk_bits;
    let chunk_mask = mask!(chunk_bits);

    let mut v = 0;
    let mut ofs = 0;

    loop {
        let s = reader.read(chunk_bits as usize + 1);
        v |= (s & chunk_mask) << ofs;
        ofs += chunk_bits;

        if (s & chunk_size) == 0 {
            break;
        }

        if ofs >= 32 {
            panic!();
        }
    }

    return v;
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

#[derive(Clone, Copy, Debug, Default)]
struct Endpoint {
    inten5: u8,
    color5: Color32,
}

#[derive(Clone, Copy, Debug,  Default)]
struct Selector {
    // Plain selectors (2-bits per value)
    selectors: [u8; 16],
}

impl Selector {

    // Returned selector value ranges from 0-3 and is a direct index into g_etc1_inten_tables.
    fn get_selector(&self, x: usize, y: usize) -> usize {
        assert!(x < 4);
        assert!(y < 4);
        self.selectors[x + 4 * y] as usize
    }

    fn set_selector(&mut self, x: usize, y: usize, val: u8) {
        assert!(x < 4);
        assert!(y < 4);
        assert!(val < 4);
        self.selectors[x + 4 * y] = val as u8;
    }
}


struct ApproxMoveToFront {
    values: Vec<u32>,
    rover: usize,
}

impl ApproxMoveToFront {
    fn new(n: usize) -> Self {
        Self {
            values: vec![0; n],
            rover: n / 2,
        }
    }

    fn size(&self) -> usize {
        self.values.len()
    }

    fn add(&mut self, new_value: u32) {
        self.values[self.rover] = new_value;
        self.rover += 1;
        if self.rover == self.values.len() {
            self.rover = self.values.len() / 2;
        }
    }

    fn use_index(&mut self, index: usize) {
        if index > 0 {
            let x = self.values[index / 2];
            let y = self.values[index];
            self.values[index / 2] = y;
            self.values[index] = x;
        }
    }
}

impl Index<usize> for ApproxMoveToFront {
    type Output = u32;
    fn index<'a>(&'a self, i: usize) -> &'a Self::Output {
        &self.values[i]
    }
}

impl IndexMut<usize> for ApproxMoveToFront {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Self::Output {
        &mut self.values[i]
    }
}


// basis_file_header::m_tex_type
enum BasisTextureType {
    Type2D = 0,
    Type2DArray = 1,
    CubemapArray = 2,
    VideoFrames = 3,
    Volume = 4,
}

// basis_slice_desc::flags
enum BasisSliceDescFlags {
    HasAlpha = 1,
    FrameIsIFrame = 2
}

// basis_file_header::m_tex_format
enum BasisTexFormat {
    ETC1S = 0,
    UASTC4x4 = 1
}

// basis_file_header::m_flags
enum BasisHeaderFlags {
    ETC1S = 1,
    YFlipped = 2,
    HasAlphaSlices = 4,
}

const BASIS_SIG: u16 = 0x4273;

#[derive(Clone, Copy, Debug, PartialEq)]
struct BasisFileHeader {
    sig: u16,                  // 2 byte file signature
    ver: u16,                  // File version
    header_size: u16,          // Header size in bytes, sizeof(basis_file_header) or 0x4D
    header_crc16: u16,         // CRC16/genibus of the remaining header data

    data_size: u32,            // The total size of all data after the header
    data_crc16: u16,           // The CRC16 of all data after the header

    total_slices: u32, /*24*/  // The number of compressed slices
    total_images: u32, /*24*/  // The total # of images

    tex_format: u8,            // enum basis_tex_format
    flags: u16,                // enum basis_header_flags
    tex_type: u8,              // enum basis_texture_type
    us_per_frame: u32, /*24*/  // Video: microseconds per frame

    reserved: u32,             // For future use
    userdata0: u32,            // For client use
    userdata1: u32,            // For client use

    total_endpoints: u16,               // ETC1S: The number of endpoints in the endpoint codebook
    endpoint_cb_file_ofs: u32,          // ETC1S: The compressed endpoint codebook's file offset relative to the header
    endpoint_cb_file_size: u32, /*24*/  // ETC1S: The compressed endpoint codebook's size in bytes

    total_selectors: u16,               // ETC1S: The number of selectors in the selector codebook
    selector_cb_file_ofs: u32,          // ETC1S: The compressed selector codebook's file offset relative to the header
    selector_cb_file_size: u32, /*24*/  // ETC1S: The compressed selector codebook's size in bytes

    tables_file_ofs: u32,               // ETC1S: The file offset of the compressed Huffman codelength tables.
    tables_file_size: u32,              // ETC1S: The file size in bytes of the compressed Huffman codelength tables.

    slice_desc_file_ofs: u32,           // The file offset to the slice description array, usually follows the header
    extended_file_ofs: u32,             // The file offset of the "extended" header and compressed data, for future use
    extended_file_size: u32,            // The file size in bytes of the "extended" header and compressed data, for future use
}

impl BasisFileHeader {
    const FILE_SIZE: usize = 77;

    fn check_size(buf: &[u8]) -> bool {
        buf.len() >= Self::FILE_SIZE
    }

    fn from_bytes(buf: &[u8]) -> Self {
        assert!(Self::check_size(&buf));
        Self {
            sig: LE::read_u16(&buf[0..]),
            ver: LE::read_u16(&buf[2..]),
            header_size: LE::read_u16(&buf[4..]),
            header_crc16: LE::read_u16(&buf[6..]),

            data_size: LE::read_u32(&buf[8..]),
            data_crc16: LE::read_u16(&buf[12..]),

            total_slices: LE::read_u24(&buf[14..]),
            total_images: LE::read_u24(&buf[17..]),

            tex_format: buf[20],
            flags: LE::read_u16(&buf[21..]),
            tex_type: buf[23],
            us_per_frame: LE::read_u24(&buf[24..]),

            reserved: LE::read_u32(&buf[27..]),
            userdata0: LE::read_u32(&buf[31..]),
            userdata1: LE::read_u32(&buf[35..]),

            total_endpoints: LE::read_u16(&buf[39..]),
            endpoint_cb_file_ofs: LE::read_u32(&buf[41..]),
            endpoint_cb_file_size: LE::read_u24(&buf[45..]),

            total_selectors: LE::read_u16(&buf[48..]),
            selector_cb_file_ofs: LE::read_u32(&buf[50..]),
            selector_cb_file_size: LE::read_u24(&buf[54..]),

            tables_file_ofs: LE::read_u32(&buf[57..]),
            tables_file_size: LE::read_u32(&buf[61..]),

            slice_desc_file_ofs: LE::read_u32(&buf[65..]),
            extended_file_ofs: LE::read_u32(&buf[69..]),
            extended_file_size: LE::read_u32(&buf[73..]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BasisSliceDesc {
    image_index: u32, /*24*/
    level_index: u8,
    flags: u8,

    orig_width: u16,
    orig_height: u16,

    num_blocks_x: u16,
    num_blocks_y: u16,

    file_ofs: u32,
    file_size: u32,

    slice_data_crc16: u16,
}

impl BasisSliceDesc {
    const FILE_SIZE: usize = 23;

    fn check_size(buf: &[u8]) -> bool {
        buf.len() >= Self::FILE_SIZE
    }

    fn from_bytes(buf: &[u8]) -> Self {
        assert!(Self::check_size(&buf));
        Self {
            image_index: LE::read_u24(&buf[0..]),
            level_index: buf[3],
            flags: buf[4],
            orig_width: LE::read_u16(&buf[5..]),
            orig_height: LE::read_u16(&buf[7..]),
            num_blocks_x: LE::read_u16(&buf[9..]),
            num_blocks_y: LE::read_u16(&buf[11..]),
            file_ofs: LE::read_u32(&buf[13..]),
            file_size: LE::read_u32(&buf[17..]),
            slice_data_crc16: LE::read_u16(&buf[21..]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_header() {
        let bytes: Vec<u8> = (0..BasisFileHeader::FILE_SIZE as u8).collect();
        let actual = BasisFileHeader::from_bytes(&bytes);
        let expected = BasisFileHeader {
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
