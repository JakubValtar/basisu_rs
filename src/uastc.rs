
use crate::{
    Color32,
    Image,
    Result,
    basis::{
        Header,
        SliceDesc,
    },
    bitreader::BitReaderLSB,
    bitwriter::BitWriterLsb,
};

use std::fmt;

#[cfg(test)]
mod tests_to_rgba;

#[cfg(test)]
mod tests_to_astc;

const MAX_ENDPOINT_COUNT: usize = 18;

#[derive(Copy, Clone)]
struct BlockData {
    mode: Mode,
    pat: u8,
    compsel: u8,
    endpoint_count: u8,
    weight_count: u8,
    data: [u8; 40],
}

impl BlockData {
    fn new(mode: Mode, pat: u8, compsel: u8, endpoint_count: u8, weight_count: u8) -> Self {
        Self {
            mode,
            pat,
            compsel,
            endpoint_count,
            weight_count,
            data: [0; 40],
        }
    }

    fn get_endpoints_weights(&self) -> (&[u8], &[u8]) {
        let combined_count = (self.endpoint_count + self.weight_count) as usize;
        self.data[..combined_count].split_at(self.endpoint_count as usize)
    }

    fn get_endpoints_weights_mut(&mut self) -> (&mut [u8], &mut [u8]) {
        let combined_count = (self.endpoint_count + self.weight_count) as usize;
        self.data[..combined_count].split_at_mut(self.endpoint_count as usize)
    }
}

impl fmt::Debug for BlockData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (e, w) = self.get_endpoints_weights();
        f.debug_struct("ModeEW")
            .field("endpoints", &e)
            .field("weights", &w)
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
struct Mode8Etc1Flags {
    etc1d: bool,
    etc1i: u8,
    etc1s: u8,
    etc1r: u8,
    etc1g: u8,
    etc1b: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TranscodingFlags {
    bc1h0: bool,
    bc1h1: bool,
    etc1f: bool,
    etc1d: bool,
    etc1i0: u8,
    etc1i1: u8,
    etc1bias: u8,
    etc2tm: u8,
}

pub struct Decoder {
    y_flipped: bool,
}

impl Decoder {
    pub(crate) fn from_file_bytes(header: &Header, bytes: &[u8]) -> Result<Self> {
        // TODO: LUTs
        Ok(Self {
            y_flipped: header.has_y_flipped(),
        })
    }

    pub(crate) fn decode_to_rgba(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<Color32>> {

        let mut image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: 4*slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: vec![Color32::default(); slice_desc.num_blocks_x as usize * slice_desc.num_blocks_y as usize * 16],
        };

        let block_to_rgba = |block_x: u32, block_y: u32, _block_offset: usize, block_bytes: &[u8]| {
            let rgba = decode_block_to_rgba(&block_bytes);
            for y in 0..4 {
                let x_start = 4 * block_x as usize;
                let image_start = (4 * block_y as usize + y) * image.stride as usize + x_start;
                image.data[image_start..image_start + 4].copy_from_slice(&rgba[4 * y..4 * y + 4]);
            }
        };

        self.iterate_blocks(slice_desc, bytes, block_to_rgba)?;

        Ok(image)
    }

    pub(crate) fn transcode_to_astc(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {

        const ASTC_BLOCK_SIZE: usize = 16;

        let mut image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: ASTC_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: vec![0u8; slice_desc.num_blocks_x as usize * slice_desc.num_blocks_y as usize * ASTC_BLOCK_SIZE],
        };

        let block_to_astc = |_block_x: u32, _block_y: u32, block_offset: usize, block_bytes: &[u8]| {
            let output = &mut image.data[block_offset..block_offset + ASTC_BLOCK_SIZE];
            decode_block_to_astc(&block_bytes, output);
        };

        self.iterate_blocks(slice_desc, bytes, block_to_astc)?;

        Ok(image)
    }

    pub(crate) fn iterate_blocks<F>(&self, slice_desc: &SliceDesc, bytes: &[u8], mut f: F) -> Result<()>
        where F: FnMut(u32, u32, usize, &[u8])
    {
        let num_blocks_x = slice_desc.num_blocks_x as u32;
        let num_blocks_y = slice_desc.num_blocks_y as u32;

        let bytes = {
            let start = slice_desc.file_ofs as usize;
            let len = slice_desc.file_size as usize;
            &bytes[start..start+len]
        };

        let mut block_offset = 0;

        const BLOCK_SIZE: usize = 16;

        if bytes.len() < BLOCK_SIZE * num_blocks_x as usize * num_blocks_y as usize {
            return Err("Not enough bytes for all blocks".into());
        }

        for block_y in 0..num_blocks_y {
            for block_x in 0..num_blocks_x {
                f(block_x, block_y, block_offset, &bytes[block_offset..block_offset + BLOCK_SIZE]);
                block_offset += BLOCK_SIZE;
            }
        }

        Ok(())
    }
}

fn block_to_rgba(block: &BlockData) -> [Color32; 16] {

    let (endpoints, weights) = block.get_endpoints_weights();

    let srgb = false;
    let mut output = [Color32::default(); 16];

    let mode = block.mode;

    if mode.subset_count == 1 {
        let (e0, e1) = match mode.cem {
            // CEM 8 - RGB Direct
            CEM_RGB => (
                Color32::new(endpoints[0], endpoints[2], endpoints[4], 0xFF),
                Color32::new(endpoints[1], endpoints[3], endpoints[5], 0xFF),
            ),
            // CEM 12 - RGBA Direct
            CEM_RGBA => (
                Color32::new(endpoints[0], endpoints[2], endpoints[4], endpoints[6]),
                Color32::new(endpoints[1], endpoints[3], endpoints[5], endpoints[7]),
            ),
            // CEM 4 - LA Direct
            CEM_LA => (
                Color32::new(endpoints[0], endpoints[0], endpoints[0], endpoints[2]),
                Color32::new(endpoints[1], endpoints[1], endpoints[1], endpoints[3]),
            ),
            _ => unreachable!()
        };

        let mut w_plane_id = [0; 4];
        let ws_per_texel = mode.plane_count as usize;
        if ws_per_texel > 1 {
            w_plane_id[block.compsel as usize] = 1;
        }

        for id in 0..16 {
            let wr = weights[ws_per_texel*id + w_plane_id[0]] as u32;
            let wg = weights[ws_per_texel*id + w_plane_id[1]] as u32;
            let wb = weights[ws_per_texel*id + w_plane_id[2]] as u32;
            let wa = weights[ws_per_texel*id + w_plane_id[3]] as u32;

            output[id] = Color32::new(
                astc_interpolate(e0[0] as u32, e1[0] as u32, wr, srgb),
                astc_interpolate(e0[1] as u32, e1[1] as u32, wg, srgb),
                astc_interpolate(e0[2] as u32, e1[2] as u32, wb, srgb),
                astc_interpolate(e0[3] as u32, e1[3] as u32, wa, false),
            );
        }
    } else {
        let mut e = [(Color32::default(), Color32::default()); 3];

        match mode.cem {
            // CEM 8 - RGB Direct
            CEM_RGB => for subset in 0..mode.subset_count as usize {
                let i = 6 * subset;
                e[subset] = (
                    Color32::new(endpoints[i+0], endpoints[i+2], endpoints[i+4], 0xFF),
                    Color32::new(endpoints[i+1], endpoints[i+3], endpoints[i+5], 0xFF),
                );
            }
            // CEM 12 - RGBA Direct
            CEM_RGBA => for subset in 0..mode.subset_count as usize {
                let i = 8 * subset;
                e[subset] = (
                    Color32::new(endpoints[i+0], endpoints[i+2], endpoints[i+4], endpoints[i+6]),
                    Color32::new(endpoints[i+1], endpoints[i+3], endpoints[i+5], endpoints[i+7]),
                );
            }
            // CEM 4 - LA Direct
            CEM_LA => for subset in 0..mode.subset_count as usize {
                let i = 4 * subset;
                e[subset] = (
                    Color32::new(endpoints[i+0], endpoints[i+0], endpoints[i+0], endpoints[i+2]),
                    Color32::new(endpoints[i+1], endpoints[i+1], endpoints[i+1], endpoints[i+3]),
                );
            }
            _ => unreachable!()
        }

        let pattern = match (mode.id, mode.subset_count) {
            // Mode 7 has 2 subsets, but needs 2/3 patern table
            (7, _) => PATTERNS_2_3[block.pat as usize],
            (_, 2) => PATTERNS_2[block.pat as usize],
            (_, 3) => PATTERNS_3[block.pat as usize],
            _ => unreachable!(),
        };

        for id in 0..16 {
            let subset = pattern[id] as usize;
            let (e0, e1) = e[subset];
            let w = weights[id] as u32;

            output[id] = Color32::new(
                astc_interpolate(e0[0] as u32, e1[0] as u32, w, srgb),
                astc_interpolate(e0[1] as u32, e1[1] as u32, w, srgb),
                astc_interpolate(e0[2] as u32, e1[2] as u32, w, srgb),
                astc_interpolate(e0[3] as u32, e1[3] as u32, w, false),
            );
        }
    }

    output
}

fn astc_interpolate(mut l: u32, mut h: u32, w: u32, srgb: bool) -> u8 {
    if srgb {
        l = (l << 8) | 0x80;
        h = (h << 8) | 0x80;
    } else {
        l = (l << 8) | l;
        h = (h << 8) | h;
    }

    let k = (l * (64 - w) + h * w + 32) >> 6;

    return (k >> 8) as u8;
}

fn decode_block_to_rgba(bytes: &[u8]) -> [Color32; 16] {
    match decode_block_to_rgba_result(bytes) {
        Ok(rgba) => rgba,
        _ => [INVALID_BLOCK_COLOR; 16],
    }
}

fn decode_block_to_rgba_result(bytes: &[u8]) -> Result<[Color32; 16]> {

    let reader = &mut BitReaderLSB::new(bytes);

    let mode = decode_mode(reader)?;

    if mode.id == 8 {
        return Ok([decode_mode8_rgba(reader); 16]);
    }

    skip_trans_flags(reader, mode);

    // Component selector for dual-plane modes
    let compsel = decode_compsel(reader, mode);

    // Pattern id for modes with multiple subsets
    let pat = decode_pattern_index(reader, mode)?;

    let endpoint_count = mode.endpoint_count;
    let weight_count = mode.plane_count * 16;

    let mut data = BlockData::new(mode, pat, compsel, endpoint_count, weight_count);
    let (endpoints, weights) = data.get_endpoints_weights_mut();

    let quant_endpoints = decode_endpoints(reader, mode.endpoint_range_index, endpoints.len());
    for (quant, unquant) in quant_endpoints.iter().zip(endpoints.iter_mut()) {
        *unquant = unquant_endpoint(*quant, mode.endpoint_range_index);
    }
    let plane_count = mode.plane_count as usize;
    let anchors = get_anchor_weight_indices(mode, pat);

    decode_weights(reader, mode.weight_bits, plane_count, anchors, weights);
    unquant_weights(weights, mode.weight_bits);

    Ok(block_to_rgba(&data))
}

fn get_anchor_weight_indices(mode: Mode, pat: u8) -> &'static [u8] {
    match (mode.id, mode.subset_count) {
        (7, _) => &PATTERNS_2_3_ANCHORS[pat as usize],
        (_, 2) => &PATTERNS_2_ANCHORS[pat as usize],
        (_, 3) => &PATTERNS_3_ANCHORS[pat as usize],
        _ => &[0],
    }
}

fn decode_block_to_astc(bytes: &[u8], output: &mut [u8]) {
    match decode_block_to_astc_result(bytes, output) {
        Ok(_) => (),
        _ => output.copy_from_slice(&[0; 16]),
    }
}

fn decode_block_to_astc_result(bytes: &[u8], output: &mut [u8]) -> Result<()> {
    let reader = &mut BitReaderLSB::new(bytes);

    let mode = decode_mode(reader)?;

    let writer = &mut BitWriterLsb::new(output);

    if mode.id == 8 {
        let rgba = decode_mode8_rgba(reader);

        // 0..=8: void-extent signature
        // 9: 0 means endpoints are UNORM16, 1 means FP16
        // 10..=11: reserved, must be 1
        writer.write_u16(12, 0b1101_1111_1100);

        // 4x 13 bits of void extent coordinates, we don't calculate
        // them yet so we set them to all 1s to get them ignored
        writer.write_u32(20, 0x000F_FFFF);
        writer.write_u32(32, 0xFFFF_FFFF);

        let (r, g, b, a) = (rgba[0] as u16, rgba[1] as u16, rgba[2] as u16, rgba[3] as u16);

        writer.write_u16(16, r << 8 | r);
        writer.write_u16(16, g << 8 | g);
        writer.write_u16(16, b << 8 | b);
        writer.write_u16(16, a << 8 | a);

        Ok(())
    } else {
        unimplemented!();
    }
}

fn decode_mode(reader: &mut BitReaderLSB) -> Result<Mode> {
    let mode_code = reader.peek(7) as usize;
    let mode_index = MODE_LUT[mode_code] as usize;

    if mode_index >= 19 {
        return Err("invalid mode index".into());
    }

    let mode = MODES[mode_index];

    reader.remove(mode.code_size as usize);

    Ok(mode)
}

fn decode_compsel(reader: &mut BitReaderLSB, mode: Mode) -> u8 {
    match (mode.plane_count, mode.cem) {
        // LA modes always have component selector 3 for alpha
        (2, CEM_LA) => 3,
        (2, _) => reader.read_u8(2),
        _ => 0,
    }
}

fn decode_pattern_index(reader: &mut BitReaderLSB, mode: Mode) -> Result<u8> {
    if mode.subset_count == 1 {
        return Ok(0);
    }

    let (pattern_index, pattern_count) = match (mode.id, mode.subset_count) {
        (7, _) => (reader.read_u8(5), TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS),
        (_, 2) => (reader.read_u8(5), TOTAL_ASTC_BC7_COMMON_PARTITIONS2),
        (_, 3) => (reader.read_u8(4), TOTAL_ASTC_BC7_COMMON_PARTITIONS3),
        _ => unreachable!(),
    };

    // Check pattern bounds
    if (pattern_index as usize) < pattern_count {
        Ok(pattern_index)
    } else {
        Err("block pattern is not valid".into())
    }
}

fn decode_mode8_rgba(reader: &mut BitReaderLSB) -> Color32 {
    Color32::new(
        reader.read_u8(8), // R
        reader.read_u8(8), // G
        reader.read_u8(8), // B
        reader.read_u8(8), // A
    )
}

fn decode_mode8_etc1_flags(reader: &mut BitReaderLSB) -> Mode8Etc1Flags {
    Mode8Etc1Flags {
        etc1d: reader.read_bool(),
        etc1i: reader.read_u8(3),
        etc1s: reader.read_u8(2),
        etc1r: reader.read_u8(5),
        etc1g: reader.read_u8(5),
        etc1b: reader.read_u8(5),
    }
}

fn decode_trans_flags(reader: &mut BitReaderLSB, mode: Mode) -> TranscodingFlags {
    assert_ne!(mode.id, 8);

    let mut flags = TranscodingFlags::default();

    flags.bc1h0 = reader.read_bool();
    if mode.id < 10 || mode.id > 12 {
        flags.bc1h1 = reader.read_bool();
    }
    flags.etc1f = reader.read_bool();
    flags.etc1d = reader.read_bool();
    flags.etc1i0 = reader.read_u8(3);
    flags.etc1i1 = reader.read_u8(3);
    if mode.id < 10 || mode.id > 12 {
        flags.etc1bias = reader.read_u8(5);
    }
    if mode.cem == CEM_RGBA || mode.cem == CEM_LA {
        // Only for modes with alpha
        flags.etc2tm = reader.read_u8(8);
    }
    flags
}

fn skip_trans_flags(reader: &mut BitReaderLSB, mode: Mode) {
    assert_ne!(mode.id, 8);
    reader.remove(mode.trans_flags_bits as usize);
}

const INVALID_BLOCK_COLOR: Color32 = Color32::new(0xFF, 0, 0xFF, 0xFF);

const CEM_RGB: u8 = 8;
const CEM_RGBA: u8 = 12;
const CEM_LA: u8 = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct Mode {
    id: u8,
    code_size: u8,
    endpoint_range_index: u8,
    endpoint_count: u8,
    weight_bits: u8,
    plane_count: u8,
    subset_count: u8,
    trans_flags_bits: u8,
    cem: u8,
}

static MODES: [Mode; 20] = [
    // CEM 8 - RGB Direct
    Mode { id:  0, code_size: 4, endpoint_range_index: 19, endpoint_count:  6, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  1, code_size: 6, endpoint_range_index: 20, endpoint_count:  6, weight_bits: 2, plane_count: 1, subset_count: 1, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  2, code_size: 5, endpoint_range_index:  8, endpoint_count: 12, weight_bits: 3, plane_count: 1, subset_count: 2, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  3, code_size: 5, endpoint_range_index:  7, endpoint_count: 18, weight_bits: 2, plane_count: 1, subset_count: 3, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  4, code_size: 5, endpoint_range_index: 12, endpoint_count: 12, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  5, code_size: 5, endpoint_range_index: 20, endpoint_count:  6, weight_bits: 3, plane_count: 1, subset_count: 1, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  6, code_size: 5, endpoint_range_index: 18, endpoint_count:  6, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 15, cem: CEM_RGB },
    Mode { id:  7, code_size: 5, endpoint_range_index: 12, endpoint_count: 12, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 15, cem: CEM_RGB },

    // Void-Extent
    Mode { id:  8, code_size: 5, endpoint_range_index:  0, endpoint_count:  0, weight_bits: 0, plane_count: 0, subset_count: 0, trans_flags_bits:  0, cem:  0 },

    // CEM 12 - RGBA Direct
    Mode { id:  9, code_size: 5, endpoint_range_index:  8, endpoint_count: 16, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 23, cem: CEM_RGBA },
    Mode { id: 10, code_size: 3, endpoint_range_index: 13, endpoint_count:  8, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 17, cem: CEM_RGBA },
    Mode { id: 11, code_size: 2, endpoint_range_index: 13, endpoint_count:  8, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 17, cem: CEM_RGBA },
    Mode { id: 12, code_size: 3, endpoint_range_index: 19, endpoint_count:  8, weight_bits: 3, plane_count: 1, subset_count: 1, trans_flags_bits: 17, cem: CEM_RGBA },
    Mode { id: 13, code_size: 5, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 1, plane_count: 2, subset_count: 1, trans_flags_bits: 23, cem: CEM_RGBA },
    Mode { id: 14, code_size: 5, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 2, plane_count: 1, subset_count: 1, trans_flags_bits: 23, cem: CEM_RGBA },

    // CEM 4 - LA Direct
    Mode { id: 15, code_size: 7, endpoint_range_index: 20, endpoint_count:  4, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 23, cem: CEM_LA },
    Mode { id: 16, code_size: 6, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 23, cem: CEM_LA },
    Mode { id: 17, code_size: 6, endpoint_range_index: 20, endpoint_count:  4, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 23, cem: CEM_LA },

    // CEM 8 - RGB Direct
    Mode { id: 18, code_size: 4, endpoint_range_index: 11, endpoint_count:  6, weight_bits: 5, plane_count: 1, subset_count: 1, trans_flags_bits: 15, cem: CEM_RGB },

    Mode { id: 19, code_size: 7, endpoint_range_index:  0, endpoint_count:  0, weight_bits: 0, plane_count: 0, subset_count: 0, trans_flags_bits:  0, cem: 0 }, // reserved
];

static MODE_LUT: [u8; 128] = [
    11,  0, 10, 3, 11, 15, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11, 16, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 17, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11,  1, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 19, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11, 16, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 17, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11,  1, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
];

#[derive(Clone, Copy, Default)]
struct QuantEndpoint {
    trit_quint: u8,
    bits: u8,
}

fn unquant_endpoint(quant: QuantEndpoint, range_index: u8) -> u8 {
    let range = BISE_RANGES[range_index as usize];
    let quant_bits = quant.bits as u16;
    if range.trits == 0 && range.quints == 0 && range.bits > 0 {
        // Left align bits
        let mut bits_la = quant_bits << (8 - range.bits);
        let mut val: u16 = 0;
        // Repeat bits into val
        while bits_la > 0 {
            val |= bits_la;
            bits_la >>= range.bits;
        }
        val as u8
    } else {
        let a = if quant_bits & 1 != 0 { 511 } else { 0 };
        let mut b: u16 = 0;
        for j in 0..9 {
            b <<= 1;
            let shift = range.deq_b[j];
            if shift != b'0' {
                b |= (quant_bits >> (shift - b'a')) & 0x1;
            }
        }
        let c = range.deq_c as u16;
        let d = quant.trit_quint as u16;
        let mut val = d * c + b;
        val ^= a;
        (a & 0x80 | val >> 2) as u8
    }
}

fn decode_endpoints(reader: &mut BitReaderLSB, range_index: u8, value_count: usize) -> [QuantEndpoint; MAX_ENDPOINT_COUNT] {
    assert!(value_count <= MAX_ENDPOINT_COUNT);

    let mut output = [QuantEndpoint::default(); MAX_ENDPOINT_COUNT];

    let range = BISE_RANGES[range_index as usize];

    let bit_count = range.bits;

    if range.quints > 0 {
        const QUINTS_PER_GROUP: usize = 3;
        const BITS_PER_GROUP: usize = 7;
        let mut out_pos = 0;
        for _ in 0..(value_count / QUINTS_PER_GROUP) as usize {
            let mut quints = reader.read_u8(BITS_PER_GROUP);
            for _ in 0..QUINTS_PER_GROUP {
                output[out_pos as usize].trit_quint = quints % 5;
                quints /= 5;
                out_pos += 1;
            }
        }
        let remaining = value_count - out_pos;
        if remaining > 0 {
            let bits_used = match remaining {
                1 => 3,
                2 => 5,
                _ => unreachable!(),
            };
            let mut quints = reader.read_u8(bits_used);
            for _ in 0..remaining {
                output[out_pos as usize].trit_quint = quints % 5;
                quints /= 5;
                out_pos += 1;
            }
        }
    }

    if range.trits > 0 {
        const TRITS_PER_GROUP: usize = 5;
        const BITS_PER_GROUP: usize = 8;
        let mut out_pos = 0;
        for _ in 0..(value_count / TRITS_PER_GROUP) as usize {
            let mut trits = reader.read_u8(BITS_PER_GROUP);
            for _ in 0..TRITS_PER_GROUP {
                output[out_pos as usize].trit_quint = trits % 3;
                trits /= 3;
                out_pos += 1;
            }
        }
        let remaining = value_count - out_pos;
        if remaining > 0 {
            let bits_used = match remaining {
                1 => 2,
                2 => 4,
                3 => 5,
                4 => 7,
                _ => unreachable!(),
            };
            let mut trits = reader.read_u8(bits_used);
            for _ in 0..remaining {
                output[out_pos as usize].trit_quint = trits % 3;
                trits /= 3;
                out_pos += 1;
            }
        }
    }

    if bit_count > 0 {
        for i in 0..value_count {
            let bits = reader.read_u8(bit_count as usize);
            output[i].bits = bits;
        }
    }

    output
}

fn unquant_weights(weights: &mut [u8], weight_bits: u8) {
    const LUT1: [u8; 2] = [ 0, 64 ];
    const LUT2: [u8; 4] = [ 0, 21, 43, 64 ];
    const LUT3: [u8; 8] = [ 0, 9, 18, 27, 37, 46, 55, 64 ];
    const LUT4: [u8; 16] = [ 0, 4, 8, 12, 17, 21, 25, 29, 35, 39, 43, 47, 52, 56, 60, 64 ];
    const LUT5: [u8; 32] = [ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64 ];

    let lut = match weight_bits {
        1 => &LUT1[..],
        2 => &LUT2[..],
        3 => &LUT3[..],
        4 => &LUT4[..],
        5 => &LUT5[..],
        _ => unreachable!()
    };

    for weight in weights {
        *weight = lut[*weight as usize];
    }
}

fn decode_weights(reader: &mut BitReaderLSB, weight_bits: u8, plane_count: usize, anchors: &[u8], output: &mut [u8]) {
    let mut bits = [weight_bits; 16];
    for &anchor in anchors {
        bits[anchor as usize] = weight_bits - 1;
    }

    // First weight of each subset is encoded with one less bit (MSB = 0)
    for i in 0..16 {
        let bits = bits[i] as usize;
        for plane in 0..plane_count {
            output[plane_count * i as usize + plane] = reader.read_u8(bits);
        }
    }
}

#[derive(Clone, Copy)]
struct BiseCounts {
    bits: u8,
    trits: u8,
    quints: u8,
    max: u8,
    deq_b: &'static[u8; 9],
    deq_c: u8,
}

static BISE_RANGES: [BiseCounts; 21] = [
    BiseCounts { bits: 1, trits: 0, quints: 0, max:   1, deq_b: b"         ", deq_c:   0 }, //  0
    BiseCounts { bits: 0, trits: 1, quints: 0, max:   2, deq_b: b"         ", deq_c:   0 }, //  1
    BiseCounts { bits: 2, trits: 0, quints: 0, max:   3, deq_b: b"         ", deq_c:   0 }, //  2
    BiseCounts { bits: 0, trits: 0, quints: 1, max:   4, deq_b: b"         ", deq_c:   0 }, //  3
    BiseCounts { bits: 1, trits: 1, quints: 0, max:   5, deq_b: b"000000000", deq_c: 204 }, //  4
    BiseCounts { bits: 3, trits: 0, quints: 0, max:   7, deq_b: b"         ", deq_c:   0 }, //  5
    BiseCounts { bits: 1, trits: 0, quints: 1, max:   9, deq_b: b"000000000", deq_c: 113 }, //  6
    BiseCounts { bits: 2, trits: 1, quints: 0, max:  11, deq_b: b"b000b0bb0", deq_c:  93 }, //  7
    BiseCounts { bits: 4, trits: 0, quints: 0, max:  15, deq_b: b"         ", deq_c:   0 }, //  8
    BiseCounts { bits: 2, trits: 0, quints: 1, max:  19, deq_b: b"b0000bb00", deq_c:  54 }, //  9
    BiseCounts { bits: 3, trits: 1, quints: 0, max:  23, deq_b: b"cb000cbcb", deq_c:  44 }, // 10
    BiseCounts { bits: 5, trits: 0, quints: 0, max:  31, deq_b: b"         ", deq_c:   0 }, // 11
    BiseCounts { bits: 3, trits: 0, quints: 1, max:  39, deq_b: b"cb0000cbc", deq_c:  26 }, // 12
    BiseCounts { bits: 4, trits: 1, quints: 0, max:  47, deq_b: b"dcb000dcb", deq_c:  22 }, // 13
    BiseCounts { bits: 6, trits: 0, quints: 0, max:  63, deq_b: b"         ", deq_c:   0 }, // 14
    BiseCounts { bits: 4, trits: 0, quints: 1, max:  79, deq_b: b"dcb0000dc", deq_c:  13 }, // 15
    BiseCounts { bits: 5, trits: 1, quints: 0, max:  95, deq_b: b"edcb000ed", deq_c:  11 }, // 16
    BiseCounts { bits: 7, trits: 0, quints: 0, max: 127, deq_b: b"         ", deq_c:   0 }, // 17
    BiseCounts { bits: 5, trits: 0, quints: 1, max: 159, deq_b: b"edcb0000e", deq_c:   6 }, // 18
    BiseCounts { bits: 6, trits: 1, quints: 0, max: 191, deq_b: b"fedcb000f", deq_c:   5 }, // 19
    BiseCounts { bits: 8, trits: 0, quints: 0, max: 255, deq_b: b"         ", deq_c:   0 }, // 20
];

const TOTAL_ASTC_BC7_COMMON_PARTITIONS2: usize = 30;
const TOTAL_ASTC_BC7_COMMON_PARTITIONS3: usize = 11;
const TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS: usize = 19;

// UASTC pattern table for the 2-subset modes
static PATTERNS_2: [[u8; 16]; TOTAL_ASTC_BC7_COMMON_PARTITIONS2] = [
   [ 0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1 ], [ 0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1 ],
   [ 1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0 ], [ 0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1 ],
   [ 1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0 ], [ 0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1 ],
   [ 1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0 ], [ 1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0 ],
   [ 0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1 ], [ 1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0 ],
   [ 0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1 ], [ 1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0 ],
   [ 1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0 ], [ 1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0 ],
   [ 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 ], [ 1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0 ],
   [ 1,0,0,0,1,1,1,0,1,1,1,1,1,1,1,1 ], [ 1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,1 ],
   [ 0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0 ], [ 0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0 ],
   [ 0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0 ], [ 1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1 ],
   [ 1,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0 ], [ 0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0 ],
   [ 1,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1 ], [ 0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0 ],
   [ 1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1 ], [ 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0 ],
   [ 1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0 ], [ 1,0,0,1,0,0,1,1,0,1,1,0,1,1,0,0 ],
];

// UASTC pattern table for the 3-subset modes
static PATTERNS_3: [[u8; 16]; TOTAL_ASTC_BC7_COMMON_PARTITIONS3] = [
   [ 0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2 ], [ 1,1,1,1,1,1,1,1,0,0,0,0,2,2,2,2 ],
   [ 1,1,1,1,0,0,0,0,0,0,0,0,2,2,2,2 ], [ 1,1,1,1,2,2,2,2,0,0,0,0,0,0,0,0 ],
   [ 1,1,2,0,1,1,2,0,1,1,2,0,1,1,2,0 ], [ 0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2 ],
   [ 0,2,1,1,0,2,1,1,0,2,1,1,0,2,1,1 ], [ 2,0,0,0,2,0,0,0,2,1,1,1,2,1,1,1 ],
   [ 2,0,1,2,2,0,1,2,2,0,1,2,2,0,1,2 ], [ 1,1,1,1,0,0,0,0,2,2,2,2,1,1,1,1 ],
   [ 0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2 ]
];

static PATTERNS_2_3: [[u8; 16]; TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS] = [
   [ 0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0 ], [ 0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0 ],
   [ 1,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0 ], [ 0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1 ],
   [ 1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1 ], [ 0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0 ],
   [ 0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1 ], [ 0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1 ],
   [ 1,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0 ], [ 0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0 ],
   [ 0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,0 ], [ 1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0 ],
   [ 0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0 ], [ 0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1 ],
   [ 1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0 ], [ 1,1,0,0,1,1,0,0,1,1,0,0,1,0,0,0 ],
   [ 1,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0 ], [ 0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,0 ],
   [ 1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0 ]
];

static PATTERNS_2_ANCHORS: [[u8; 2]; TOTAL_ASTC_BC7_COMMON_PARTITIONS2] = [
   [ 0, 2 ], [ 0, 3 ], [ 1, 0 ], [ 0, 3 ], [ 7, 0 ], [ 0, 2 ], [ 3, 0 ],
   [ 7, 0 ], [ 0, 11 ], [ 2, 0 ], [ 0, 7 ], [ 11, 0 ], [ 3, 0 ], [ 8, 0 ],
   [ 0, 4 ], [ 12, 0 ], [ 1, 0 ], [ 8, 0 ], [ 0, 1 ], [ 0, 2 ], [ 0, 4 ],
   [ 8, 0 ], [ 1, 0 ], [ 0, 2 ], [ 4, 0 ], [ 0, 1 ], [ 4, 0 ], [ 1, 0 ],
   [ 4, 0 ], [ 1, 0 ]
];

static PATTERNS_3_ANCHORS: [[u8; 3]; TOTAL_ASTC_BC7_COMMON_PARTITIONS3] = [
   [ 0, 8, 10 ],  [ 8, 0, 12 ], [ 4, 0, 12 ], [ 8, 0, 4 ], [ 3, 0, 2 ],
   [ 0, 1, 3 ], [ 0, 2, 1 ], [ 1, 9, 0 ], [ 1, 2, 0 ], [ 4, 0, 8 ], [ 0, 6, 2 ]
];

static PATTERNS_2_3_ANCHORS: [[u8; 2]; TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS] = [
   [ 0, 4 ], [ 0, 2 ], [ 2, 0 ], [ 0, 7 ], [ 8, 0 ], [ 0, 1 ], [ 0, 3 ],
   [ 0, 1 ], [ 2, 0 ], [ 0, 1 ], [ 0, 8 ], [ 2, 0 ], [ 0, 1 ], [ 0, 7 ],
   [ 12, 0 ], [ 2, 0 ], [ 9, 0 ], [ 0, 2 ], [ 4, 0 ]
];
