use byteorder::{
    ByteOrder,
    LE,
};

// basis_file_header::m_tex_type
pub enum TextureType {
    Type2D = 0,
    Type2DArray = 1,
    CubemapArray = 2,
    VideoFrames = 3,
    Volume = 4,
}

// basis_slice_desc::flags
pub enum SliceDescFlags {
    HasAlpha = 1,
    FrameIsIFrame = 2
}

// basis_file_header::m_tex_format
pub enum TexFormat {
    ETC1S = 0,
    UASTC4x4 = 1
}

// basis_file_header::m_flags
pub enum HeaderFlags {
    ETC1S = 1,
    YFlipped = 2,
    HasAlphaSlices = 4,
}

pub const SIG: u16 = 0x4273;

pub fn check_file_sig(bytes: &[u8]) -> bool {
    let sig = LE::read_u16(&bytes);
    sig == SIG
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub fn from_bytes(buf: &[u8]) -> Self {
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

    pub fn from_bytes(buf: &[u8]) -> Self {
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
        let bytes: Vec<u8> = (0..Header::FILE_SIZE as u8).collect();
        let actual = Header::from_bytes(&bytes);
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
