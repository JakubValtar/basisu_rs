use crate::{
    Color32,
    Image,
    Result,
    basis::{
        Header,
        SliceDesc,
    },
    bitreader::BitReaderLsb,
    bitwriter::BitWriterLsb,
    mask, astc, bc7, etc1s,
};

#[cfg(test)]
mod tests_to_rgba;

#[cfg(test)]
mod tests_to_astc;

#[cfg(test)]
mod tests_to_bc7;

#[cfg(test)]
mod tests_to_etc1;

#[cfg(test)]
mod tests_to_etc2;

const MAX_ENDPOINT_COUNT: usize = 18;
const MAX_WEIGHT_COUNT: usize = 32;

#[derive(Clone, Copy, Debug)]
struct Mode8Etc1Flags {
    etc1d: bool,
    etc1i: u8,
    etc1s: u8,
    etc1r: u8,
    etc1g: u8,
    etc1b: u8,
}

const ETC1BIAS_NONE: u8 = 0xFF;
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

    pub(crate) fn read_to_uastc(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {

        const UASTC_BLOCK_SIZE: usize = 16;

        let block_bytes = {
            let start = slice_desc.file_ofs as usize;
            let len = slice_desc.file_size as usize;
            &bytes[start..start+len]
        };

        let image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: UASTC_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: block_bytes.to_vec(),
        };

        Ok(image)
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
            astc::convert_block_from_uastc(&block_bytes, output);
        };

        self.iterate_blocks(slice_desc, bytes, block_to_astc)?;

        Ok(image)
    }

    pub(crate) fn transcode_to_bc7(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {

        const BC7_BLOCK_SIZE: usize = 16;

        let mut image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: BC7_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: vec![0u8; slice_desc.num_blocks_x as usize * slice_desc.num_blocks_y as usize * BC7_BLOCK_SIZE],
        };

        let block_to_bc7 = |_block_x: u32, _block_y: u32, block_offset: usize, block_bytes: &[u8]| {
            let output = &mut image.data[block_offset..block_offset + BC7_BLOCK_SIZE];
            bc7::convert_block_from_uastc(&block_bytes, output);
        };

        self.iterate_blocks(slice_desc, bytes, block_to_bc7)?;

        Ok(image)
    }

    pub(crate) fn transcode_to_etc1(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {

        const ETC1_BLOCK_SIZE: usize = 8;

        let mut image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: ETC1_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: vec![0u8; slice_desc.num_blocks_x as usize * slice_desc.num_blocks_y as usize * ETC1_BLOCK_SIZE],
        };

        let block_to_etc1 = |_block_x: u32, _block_y: u32, block_offset: usize, block_bytes: &[u8]| {
            let output = &mut image.data[block_offset/2..block_offset/2 + ETC1_BLOCK_SIZE];
            decode_block_to_etc(&block_bytes, output, false);
        };

        self.iterate_blocks(slice_desc, bytes, block_to_etc1)?;

        Ok(image)
    }

    pub(crate) fn transcode_to_etc2(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {

        const ETC2_BLOCK_SIZE: usize = 16;

        let mut image = Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: ETC2_BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: vec![0u8; slice_desc.num_blocks_x as usize * slice_desc.num_blocks_y as usize * ETC2_BLOCK_SIZE],
        };

        let block_to_etc1 = |_block_x: u32, _block_y: u32, block_offset: usize, block_bytes: &[u8]| {
            let output = &mut image.data[block_offset..block_offset + ETC2_BLOCK_SIZE];
            decode_block_to_etc(&block_bytes, output, true);
        };

        self.iterate_blocks(slice_desc, bytes, block_to_etc1)?;

        Ok(image)
    }

    fn iterate_blocks<F>(&self, slice_desc: &SliceDesc, bytes: &[u8], mut f: F) -> Result<()>
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

pub(crate) fn assemble_endpoint_pairs(mode: Mode, endpoint_bytes: &[u8]) -> [[Color32; 2]; 3] {
    let mut endpoint_pairs = [[Color32::default(); 2]; 3];

    match mode.cem {
        // CEM 8 - RGB Direct
        astc::CEM_RGB => {
            for (pair, bytes) in endpoint_pairs.iter_mut().zip(endpoint_bytes.chunks_exact(6)) {
                *pair = [
                    Color32::new(bytes[0], bytes[2], bytes[4], 0xFF),
                    Color32::new(bytes[1], bytes[3], bytes[5], 0xFF),
                ];
            }
        }
        // CEM 12 - RGBA Direct
        astc::CEM_RGBA => {
            for (pair, bytes) in endpoint_pairs.iter_mut().zip(endpoint_bytes.chunks_exact(8)) {
                *pair = [
                    Color32::new(bytes[0], bytes[2], bytes[4], bytes[6]),
                    Color32::new(bytes[1], bytes[3], bytes[5], bytes[7]),
                ];
            }
        }
        // CEM 4 - LA Direct
        astc::CEM_LA => {
            for (pair, bytes) in endpoint_pairs.iter_mut().zip(endpoint_bytes.chunks_exact(4)) {
                *pair = [
                    Color32::new(bytes[0], bytes[0], bytes[0], bytes[2]),
                    Color32::new(bytes[1], bytes[1], bytes[1], bytes[3]),
                ];
            }
        }
        _ => unreachable!()
    }

    endpoint_pairs
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

    (k >> 8) as u8
}

fn decode_block_to_rgba(bytes: &[u8]) -> [Color32; 16] {
    match decode_block_to_rgba_result(bytes) {
        Ok(rgba) => rgba,
        _ => [INVALID_BLOCK_COLOR; 16],
    }
}

fn decode_block_to_rgba_result(bytes: &[u8]) -> Result<[Color32; 16]> {

    let reader = &mut BitReaderLsb::new(bytes);

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

    let endpoints = &mut [0u8; MAX_ENDPOINT_COUNT][..endpoint_count as usize];
    let weights = &mut [0u8; MAX_WEIGHT_COUNT][..weight_count as usize];

    let quant_endpoints = decode_endpoints(reader, mode.endpoint_range_index, endpoints.len());
    for (quant, unquant) in quant_endpoints.iter().zip(endpoints.iter_mut()) {
        *unquant = unquant_endpoint(*quant, mode.endpoint_range_index);
    }

    let weight_consumer = |i, weight| {
        weights[i] = weight;
    };
    decode_weights(reader, mode, pat, weight_consumer);
    unquant_weights(weights, mode.weight_bits);

    let srgb = false;
    let mut output = [Color32::default(); 16];

    if mode.subset_count == 1 {
        let [e0, e1] = assemble_endpoint_pairs(mode, endpoints)[0];

        let mut w_plane_id = [0; 4];
        let ws_per_texel = mode.plane_count as usize;
        if ws_per_texel > 1 {
            w_plane_id[compsel as usize] = 1;
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
        let e = assemble_endpoint_pairs(mode, endpoints);

        let pattern = get_pattern(mode, pat);

        for id in 0..16 {
            let subset = pattern[id] as usize;
            let [e0, e1] = e[subset];
            let w = weights[id] as u32;

            output[id] = Color32::new(
                astc_interpolate(e0[0] as u32, e1[0] as u32, w, srgb),
                astc_interpolate(e0[1] as u32, e1[1] as u32, w, srgb),
                astc_interpolate(e0[2] as u32, e1[2] as u32, w, srgb),
                astc_interpolate(e0[3] as u32, e1[3] as u32, w, false),
            );
        }
    }

    Ok(output)
}

fn decode_block_to_etc(bytes: &[u8], output: &mut [u8], alpha: bool) {
    match decode_block_to_etc_result(bytes, output, alpha) {
        Ok(_) => (),
        _ => output.copy_from_slice(&[0; 8]), // TODO: purple or black?
    }
}

fn decode_block_to_etc_result(bytes: &[u8], output: &mut [u8], alpha: bool) -> Result<()> {
    let reader = &mut BitReaderLsb::new(bytes);

    let mode = decode_mode(reader)?;

    let writer = &mut BitWriterLsb::new(output);

    if mode.id == 8 {
        if alpha {
            let rgba = decode_mode8_rgba(reader);
            write_solid_etc2_alpha_block(writer, rgba[3]);
        } else {
            skip_mode8_rgba(reader);
        }

        let trans_flags = decode_mode8_etc1_flags(reader);

        if !trans_flags.etc1d {
            writer.write_u8(8, trans_flags.etc1r << 4 | trans_flags.etc1r);
            writer.write_u8(8, trans_flags.etc1g << 4 | trans_flags.etc1g);
            writer.write_u8(8, trans_flags.etc1b << 4 | trans_flags.etc1b);
        } else {
            writer.write_u8(8, trans_flags.etc1r << 3);
            writer.write_u8(8, trans_flags.etc1g << 3);
            writer.write_u8(8, trans_flags.etc1b << 3);
        }
        // codeword1 (3), codeword2 (3), diff bit, flip bit
        writer.write_u8(8, trans_flags.etc1i << 5 | trans_flags.etc1i << 2 | (trans_flags.etc1d as u8) << 1);

        let selector = [0b11, 0b10, 0b00, 0b01][trans_flags.etc1s as usize];
        let s_lo = selector & 1;
        let s_hi = selector >> 1;

        writer.write_u16(16, 0u16.wrapping_sub(s_hi as u16));
        writer.write_u16(16, 0u16.wrapping_sub(s_lo as u16));

        return Ok(());
    }

    let trans_flags = decode_trans_flags(reader, mode);

    let mut rgba = decode_block_to_rgba(bytes);

    if alpha {
        write_etc2_alpha_block(writer, trans_flags.etc2tm, &rgba);
    }

    if !trans_flags.etc1f {
        // Transpose to have the two subblocks in 0..8 and 8..16
        for y in 0..3 {
            for x in (y+1)..4 {
                let a = y * 4 + x;
                let b = x * 4 + y;
                rgba.swap(a, b);
            }
        }
    }

    let color_bits = if !trans_flags.etc1d { 4 } else { 5 };
    let limit = mask!(color_bits as u32);

    let mut avg_colors = [Color32::default(); 2];
    for (subblock, avg) in rgba.chunks_exact(8).zip(avg_colors.iter_mut()) {
        let sum = subblock.iter().fold([0u16; 4], |mut acc, c| {
            for (acc, &c) in acc.iter_mut().zip(c.0.iter()) {
                *acc += c as u16;
            }
            acc
        });
        for (&sum, avg) in sum.iter().zip(avg.0.iter_mut()).take(3) {
            *avg = ((sum as u32 * limit + 1020) / (8*255)) as u8;
        }
    }

    let block_colors: [[Color32; 4]; 2];

    let c0 = apply_etc1_bias(avg_colors[0], trans_flags.etc1bias, limit, 0);
    let c1 = apply_etc1_bias(avg_colors[1], trans_flags.etc1bias, limit, 1);

    if !trans_flags.etc1d {
        writer.write_u8(8, c0[0] << 4 | c1[0]);
        writer.write_u8(8, c0[1] << 4 | c1[1]);
        writer.write_u8(8, c0[2] << 4 | c1[2]);
        block_colors = [
            etc1s::apply_mod_to_base_color(etc1s::color_4_to_8(c0), trans_flags.etc1i0),
            etc1s::apply_mod_to_base_color(etc1s::color_4_to_8(c1), trans_flags.etc1i1),
        ];
    } else {
        let d = [
            (c1[0] as i16 - c0[0] as i16).max(-4).min(3),
            (c1[1] as i16 - c0[1] as i16).max(-4).min(3),
            (c1[2] as i16 - c0[2] as i16).max(-4).min(3),
        ];
        writer.write_u8(8, c0[0] << 3 | (d[0] & 0b111) as u8);
        writer.write_u8(8, c0[1] << 3 | (d[1] & 0b111) as u8);
        writer.write_u8(8, c0[2] << 3 | (d[2] & 0b111) as u8);
        let c1 = Color32::new(
            (c0[0] as i16 + d[0]) as u8,
            (c0[1] as i16 + d[1]) as u8,
            (c0[2] as i16 + d[2]) as u8,
            255,
        );
        block_colors = [
            etc1s::apply_mod_to_base_color(etc1s::color_5_to_8(c0), trans_flags.etc1i0),
            etc1s::apply_mod_to_base_color(etc1s::color_5_to_8(c1), trans_flags.etc1i1),
        ];
    }

    {   // Write codebooks, diff and flip bits
        let val =
            trans_flags.etc1i0 << 5 |
            trans_flags.etc1i1 << 2 |
            (trans_flags.etc1d as u8) << 1 |
            trans_flags.etc1f as u8;
        writer.write_u8(8, val);
    }

    let mut selector = etc1s::Selector::default();

    for (subblock, (rgba, block_colors)) in rgba.chunks_exact(8).zip(block_colors.iter()).enumerate() {

        const LUM_FACTORS: [i32; 3] = [108, 366, 38];
        let mut block_lums = [0; 4];
        for (block_lum, block_color) in block_lums.iter_mut().zip(block_colors.iter()) {
            *block_lum = block_color.0.iter().zip(LUM_FACTORS.iter())
                .map(|(&col, &f)| col as i32 * f)
                .sum();
        }
        let block_lum_01 = (block_lums[0] + block_lums[1]) / 2;
        let block_lum_12 = (block_lums[1] + block_lums[2]) / 2;
        let block_lum_23 = (block_lums[2] + block_lums[3]) / 2;

        for (i, c) in rgba.iter().enumerate() {
            let lum: i32 = c.0.iter().zip(LUM_FACTORS.iter())
                .map(|(&col, &f)| col as i32 * f)
                .sum();
            let sel = (lum >= block_lum_01) as u8 + (lum >= block_lum_12) as u8 + (lum >= block_lum_23) as u8;
            let x = i & 0b11;
            let y = 2 * subblock + (i >> 2);
            if trans_flags.etc1f {
                selector.set_selector(x, y, sel);
            } else {
                selector.set_selector(y, x, sel);
            }
        }
    }

    writer.write_u32(32, u32::from_le_bytes(selector.etc1_bytes));

    Ok(())
}

fn apply_etc1_bias(mut block_color: Color32, bias: u8, limit: u32, subblock: u32) -> Color32 {
    if bias == ETC1BIAS_NONE {
        return block_color;
    }

    const S_DIVS: [u8; 3] = [ 1, 3, 9 ];

    for c in 0..3 {
        let delta: i32 = match bias {
            2 => if subblock == 1 { 0 } else if c == 0 { -1 } else { 0 },
            5 => if subblock == 1 { 0 } else if c == 1 { -1 } else { 0 },
            6 => if subblock == 1 { 0 } else if c == 2 { -1 } else { 0 },

             7 => if subblock == 1 { 0 } else if c == 0 { 1 } else { 0 },
            11 => if subblock == 1 { 0 } else if c == 1 { 1 } else { 0 },
            15 => if subblock == 1 { 0 } else if c == 2 { 1 } else { 0 },

            18 => if subblock == 1 { if c == 0 { -1 } else { 0 } } else { 0 },
            19 => if subblock == 1 { if c == 1 { -1 } else { 0 } } else { 0 },
            20 => if subblock == 1 { if c == 2 { -1 } else { 0 } } else { 0 },

            21 => if subblock == 1 { if c == 0 { 1 } else { 0 } } else { 0 },
            24 => if subblock == 1 { if c == 1 { 1 } else { 0 } } else { 0 },
             8 => if subblock == 1 { if c == 2 { 1 } else { 0 } } else { 0 },

            10 => -2,

            27 => if subblock == 1 {  0 } else { -1 },
            28 => if subblock == 1 { -1 } else {  1 },
            29 => if subblock == 1 {  1 } else {  0 },
            30 => if subblock == 1 { -1 } else {  0 },
            31 => if subblock == 1 {  0 } else {  1 },

            _ => ((bias / S_DIVS[c]) % 3) as i32 - 1,
        };

        let mut v = block_color[c] as i32;
        if v == 0 {
            if delta == -2 {
                v += 3;
            } else {
                v += delta + 1;
            }
        } else if v == limit as i32 {
            v += delta - 1;
        } else {
            v += delta;
            if (v < 0) || (v > limit as i32) {
                v = (v - delta) - delta;
            }
        }

        assert!(v >= 0);
        assert!(v <= limit as i32);

        block_color[c] = v as u8;
    }

    block_color
}

fn write_solid_etc2_alpha_block(writer: &mut BitWriterLsb, value: u8) {
    writer.write_u8(8, value);

    // Multiplier, doesn't matter, but has to be non-zero, so choosing 1
    // Modifier table index: choosing 13 (only one with 0 in it)
    writer.write_u8(8, (1 << 4) | 13);

    // Weight indices, 16x 3 bits, value 4 (0b100)
    writer.write_u8(8, 0b10010010);
    writer.write_u8(8, 0b01001001);
    writer.write_u8(8, 0b00100100);
    writer.write_u8(8, 0b10010010);
    writer.write_u8(8, 0b01001001);
    writer.write_u8(8, 0b00100100);
}

fn write_etc2_alpha_block(writer: &mut BitWriterLsb, etc2tm: u8, rgba: &[Color32; 16]) {
    if etc2tm == 0 {
        write_solid_etc2_alpha_block(writer, 255);
    } else {
        let mut min_alpha = 255;
        let mut max_alpha = 0;

        for c in rgba.iter() {
            min_alpha = min_alpha.min(c[3]);
            max_alpha = max_alpha.max(c[3]);
        }

        if min_alpha == max_alpha {
            write_solid_etc2_alpha_block(writer, min_alpha);
        } else {
            let table_index = (etc2tm & mask!(4u8)) as usize;
            let multiplier = (etc2tm >> 4) as u32;

            let mod_table = ETC2_ALPHA_MODIFIERS[table_index];

            let mod_min = mod_table[ETC2_ALPHA_MODIFIERS_MIN_INDEX];
            let mod_max = mod_table[ETC2_ALPHA_MODIFIERS_MAX_INDEX];
            let range = mod_max as i32 - mod_min as i32;

            fn lerp(a: f32, b: f32, amt: f32) -> f32 {
                a * (1.0 - amt) + b * amt
            }

            let mod_0_at_range_fraction = -(mod_min as f32) / range as f32;
            let center = (lerp(min_alpha as f32, max_alpha as f32, mod_0_at_range_fraction)).round() as i32;

            let mut values = [0u8; 8];
            for (val, &modifier) in values.iter_mut().zip(mod_table.iter()) {
                *val = (center + (modifier as i32 * multiplier as i32)).max(0).min(255) as u8;
            }

            let mut selectors = 0u64;
            for (i, c) in rgba.iter().enumerate() {
                let a = c[3];
                let best_selector = values.iter().enumerate()
                    .min_by_key(|(_, &val)| {
                        (val as i32 - a as i32).abs()
                    })
                    .map(|(i, _)| i)
                    .unwrap() as u64;

                // Transpose to match ETC2 pixel order
                let x = i / 4;
                let y = i % 4;
                let id = y * 4 + x;

                selectors |= best_selector << (45 - id * 3);
            }

            writer.write_u8(8, center as u8);
            // Multiplier, doesn't matter, but has to be non-zero, so choosing 1
            // Modifier table index: choosing 13 (only one with 0 in it)
            writer.write_u8(8, etc2tm);

            // Weight indices, 16x 3 bits, value 4 (0b100)
            for &byte in selectors.to_be_bytes().iter().skip(2) {
                writer.write_u8(8, byte);
            }
        }
    }
}

const ETC2_ALPHA_MODIFIERS_MIN_INDEX: usize = 3;
const ETC2_ALPHA_MODIFIERS_MAX_INDEX: usize = 7;

static ETC2_ALPHA_MODIFIERS: [[i8; 8]; 16] = [
    [ -3, -6,  -9, -15, 2, 5, 8, 14 ],
    [ -3, -7, -10, -13, 2, 6, 9, 12 ],
    [ -2, -5,  -8, -13, 1, 4, 7, 12 ],
    [ -2, -4,  -6, -13, 1, 3, 5, 12 ],
    [ -3, -6,  -8, -12, 2, 5, 7, 11 ],
    [ -3, -7,  -9, -11, 2, 6, 8, 10 ],
    [ -4, -7,  -8, -11, 3, 6, 7, 10 ],
    [ -3, -5,  -8, -11, 2, 4, 7, 10 ],
    [ -2, -6,  -8, -10, 1, 5, 7,  9 ],
    [ -2, -5,  -8, -10, 1, 4, 7,  9 ],
    [ -2, -4,  -8, -10, 1, 3, 7,  9 ],
    [ -2, -5,  -7, -10, 1, 4, 6,  9 ],
    [ -3, -4,  -7, -10, 2, 3, 6,  9 ],
    [ -1, -2,  -3, -10, 0, 1, 2,  9 ], // entry 13
    [ -4, -6,  -8,  -9, 3, 5, 7,  8 ],
    [ -3, -5,  -7,  -9, 2, 4, 6,  8 ],
];

pub fn decode_mode(reader: &mut BitReaderLsb) -> Result<Mode> {
    let mode_code = reader.peek(7) as usize;
    let mode_index = MODE_LUT[mode_code] as usize;

    if mode_index >= 19 {
        return Err("invalid mode index".into());
    }

    let mode = MODES[mode_index];

    reader.remove(mode.code_size as usize);

    Ok(mode)
}

pub fn decode_compsel(reader: &mut BitReaderLsb, mode: Mode) -> u8 {
    match (mode.plane_count, mode.cem) {
        // LA modes always have component selector 3 for alpha
        (2, astc::CEM_LA) => 3,
        (2, _) => reader.read_u8(2),
        _ => 0,
    }
}

pub fn decode_pattern_index(reader: &mut BitReaderLsb, mode: Mode) -> Result<u8> {
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

pub fn get_pattern(mode: Mode, pat: u8) -> &'static [u8] {
    match (mode.id, mode.subset_count) {
        // Mode 7 has 2 subsets, but needs 2/3 patern table
        (7, _) => &PATTERNS_2_3[pat as usize],
        (_, 2) => &PATTERNS_2[pat as usize],
        (_, 3) => &PATTERNS_3[pat as usize],
        _ => unreachable!(),
    }
}

fn get_anchor_weight_indices(mode: Mode, pat: u8) -> &'static [u8] {
    match (mode.id, mode.subset_count) {
        (7, _) => &PATTERNS_2_3_ANCHORS[pat as usize],
        (_, 2) => &PATTERNS_2_ANCHORS[pat as usize],
        (_, 3) => &PATTERNS_3_ANCHORS[pat as usize],
        _ => &[0],
    }
}

pub(crate) fn decode_mode8_rgba(reader: &mut BitReaderLsb) -> Color32 {
    Color32::new(
        reader.read_u8(8), // R
        reader.read_u8(8), // G
        reader.read_u8(8), // B
        reader.read_u8(8), // A
    )
}

fn skip_mode8_rgba(reader: &mut BitReaderLsb) {
    reader.remove(32);
}

fn decode_mode8_etc1_flags(reader: &mut BitReaderLsb) -> Mode8Etc1Flags {
    Mode8Etc1Flags {
        etc1d: reader.read_bool(),
        etc1i: reader.read_u8(3),
        etc1s: reader.read_u8(2),
        etc1r: reader.read_u8(5),
        etc1g: reader.read_u8(5),
        etc1b: reader.read_u8(5),
    }
}

fn decode_trans_flags(reader: &mut BitReaderLsb, mode: Mode) -> TranscodingFlags {
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
    flags.etc1bias = if mode.id < 10 || mode.id > 12 {
        reader.read_u8(5)
    } else {
        ETC1BIAS_NONE
    };
    if mode.cem == astc::CEM_RGBA || mode.cem == astc::CEM_LA {
        // Only for modes with alpha
        flags.etc2tm = reader.read_u8(8);
    }
    flags
}

pub fn skip_trans_flags(reader: &mut BitReaderLsb, mode: Mode) {
    assert_ne!(mode.id, 8);
    reader.remove(mode.trans_flags_bits as usize);
}

const INVALID_BLOCK_COLOR: Color32 = Color32::new(0xFF, 0, 0xFF, 0xFF);

#[derive(Clone, Copy, Debug, Default)]
pub struct Mode {
    pub id: u8,
    code_size: u8,
    pub endpoint_range_index: u8,
    pub endpoint_count: u8,
    pub weight_bits: u8,
    pub plane_count: u8,
    pub subset_count: u8,
    trans_flags_bits: u8,
    pub astc_block_mode_13: u16,
    pub cem: u8,
    pub bc7_mode: u8,
}

static MODES: [Mode; 20] = [
    // CEM 8 - RGB Direct
    Mode { id:  0, code_size: 4, endpoint_range_index: 19, endpoint_count:  6, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 15, astc_block_mode_13: 0x0242, cem:  astc::CEM_RGB, bc7_mode: 6 },
    Mode { id:  1, code_size: 6, endpoint_range_index: 20, endpoint_count:  6, weight_bits: 2, plane_count: 1, subset_count: 1, trans_flags_bits: 15, astc_block_mode_13: 0x0042, cem:  astc::CEM_RGB, bc7_mode: 3 },
    Mode { id:  2, code_size: 5, endpoint_range_index:  8, endpoint_count: 12, weight_bits: 3, plane_count: 1, subset_count: 2, trans_flags_bits: 15, astc_block_mode_13: 0x0853, cem:  astc::CEM_RGB, bc7_mode: 1 },
    Mode { id:  3, code_size: 5, endpoint_range_index:  7, endpoint_count: 18, weight_bits: 2, plane_count: 1, subset_count: 3, trans_flags_bits: 15, astc_block_mode_13: 0x1042, cem:  astc::CEM_RGB, bc7_mode: 2 },
    Mode { id:  4, code_size: 5, endpoint_range_index: 12, endpoint_count: 12, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 15, astc_block_mode_13: 0x0842, cem:  astc::CEM_RGB, bc7_mode: 3 },
    Mode { id:  5, code_size: 5, endpoint_range_index: 20, endpoint_count:  6, weight_bits: 3, plane_count: 1, subset_count: 1, trans_flags_bits: 15, astc_block_mode_13: 0x0053, cem:  astc::CEM_RGB, bc7_mode: 6 },
    Mode { id:  6, code_size: 5, endpoint_range_index: 18, endpoint_count:  6, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 15, astc_block_mode_13: 0x0442, cem:  astc::CEM_RGB, bc7_mode: 5 },
    Mode { id:  7, code_size: 5, endpoint_range_index: 12, endpoint_count: 12, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 15, astc_block_mode_13: 0x0842, cem:  astc::CEM_RGB, bc7_mode: 2 },

    // Void-Extent
    Mode { id:  8, code_size: 5, endpoint_range_index:  0, endpoint_count:  0, weight_bits: 0, plane_count: 0, subset_count: 0, trans_flags_bits:  0, astc_block_mode_13:      0, cem:              0, bc7_mode: 0 },

    // CEM 12 - RGBA Direct
    Mode { id:  9, code_size: 5, endpoint_range_index:  8, endpoint_count: 16, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 23, astc_block_mode_13: 0x0842, cem: astc::CEM_RGBA, bc7_mode: 7 },
    Mode { id: 10, code_size: 3, endpoint_range_index: 13, endpoint_count:  8, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 17, astc_block_mode_13: 0x0242, cem: astc::CEM_RGBA, bc7_mode: 6 },
    Mode { id: 11, code_size: 2, endpoint_range_index: 13, endpoint_count:  8, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 17, astc_block_mode_13: 0x0442, cem: astc::CEM_RGBA, bc7_mode: 5 },
    Mode { id: 12, code_size: 3, endpoint_range_index: 19, endpoint_count:  8, weight_bits: 3, plane_count: 1, subset_count: 1, trans_flags_bits: 17, astc_block_mode_13: 0x0053, cem: astc::CEM_RGBA, bc7_mode: 6 },
    Mode { id: 13, code_size: 5, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 1, plane_count: 2, subset_count: 1, trans_flags_bits: 23, astc_block_mode_13: 0x0441, cem: astc::CEM_RGBA, bc7_mode: 5 },
    Mode { id: 14, code_size: 5, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 2, plane_count: 1, subset_count: 1, trans_flags_bits: 23, astc_block_mode_13: 0x0042, cem: astc::CEM_RGBA, bc7_mode: 6 },

    // CEM 4 - LA Direct
    Mode { id: 15, code_size: 7, endpoint_range_index: 20, endpoint_count:  4, weight_bits: 4, plane_count: 1, subset_count: 1, trans_flags_bits: 23, astc_block_mode_13: 0x0242, cem:   astc::CEM_LA, bc7_mode: 6 },
    Mode { id: 16, code_size: 6, endpoint_range_index: 20, endpoint_count:  8, weight_bits: 2, plane_count: 1, subset_count: 2, trans_flags_bits: 23, astc_block_mode_13: 0x0842, cem:   astc::CEM_LA, bc7_mode: 7 },
    Mode { id: 17, code_size: 6, endpoint_range_index: 20, endpoint_count:  4, weight_bits: 2, plane_count: 2, subset_count: 1, trans_flags_bits: 23, astc_block_mode_13: 0x0442, cem:   astc::CEM_LA, bc7_mode: 5 },

    // CEM 8 - RGB Direct
    Mode { id: 18, code_size: 4, endpoint_range_index: 11, endpoint_count:  6, weight_bits: 5, plane_count: 1, subset_count: 1, trans_flags_bits: 15, astc_block_mode_13: 0x0253, cem:  astc::CEM_RGB, bc7_mode: 6 },

    Mode { id: 19, code_size: 7, endpoint_range_index:  0, endpoint_count:  0, weight_bits: 0, plane_count: 0, subset_count: 0, trans_flags_bits:  0, astc_block_mode_13:      0, cem:              0, bc7_mode: 0 }, // reserved
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

#[derive(Clone, Copy, Debug, Default)]
pub struct QuantEndpoint {
    pub trit_quint: u8,
    pub bits: u8,
}

pub fn unquant_endpoint(quant: QuantEndpoint, range_index: u8) -> u8 {
    let range = astc::BISE_RANGES[range_index as usize];
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

pub fn decode_endpoints(reader: &mut BitReaderLsb, range_index: u8, value_count: usize) -> [QuantEndpoint; MAX_ENDPOINT_COUNT] {
    assert!(value_count <= MAX_ENDPOINT_COUNT);

    let mut output = [QuantEndpoint::default(); MAX_ENDPOINT_COUNT];

    let range = astc::BISE_RANGES[range_index as usize];

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

pub fn decode_weights<F>(reader: &mut BitReaderLsb, mode: Mode, pat: u8, mut f: F)
    where F: FnMut(usize, u8)
{
    let plane_count = mode.plane_count as usize;
    let anchors = get_anchor_weight_indices(mode, pat);

    // One anchor weight in each subset is encoded with one less bit (MSB = 0)
    let mut bits = [mode.weight_bits; 16];
    for &anchor in anchors {
        bits[anchor as usize] = mode.weight_bits - 1;
    }

    for i in 0..16 {
        let bits = bits[i] as usize;
        for plane in 0..plane_count {
            f(plane_count * i as usize + plane, reader.read_u8(bits));
        }
    }
}

pub const TOTAL_ASTC_BC7_COMMON_PARTITIONS2: usize = 30;
pub const TOTAL_ASTC_BC7_COMMON_PARTITIONS3: usize = 11;
pub const TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS: usize = 19;

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
