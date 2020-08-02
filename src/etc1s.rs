
use std::ops::{
    Index,
    IndexMut,
};
use crate::{
    Color32,
    Image,
    mask,
    Result,
    basis::{
        Header,
        SliceDesc,
        TextureType,
    },
    bitreader::BitReaderLSB,
    huffman::{
        self,
        HuffmanDecodingTable,
    }
};

const ENDPOINT_PRED_TOTAL_SYMBOLS: u16 = (4 * 4 * 4 * 4) + 1;
const ENDPOINT_PRED_REPEAT_LAST_SYMBOL: u16 = ENDPOINT_PRED_TOTAL_SYMBOLS - 1;
const ENDPOINT_PRED_MIN_REPEAT_COUNT: u32 = 3;
const ENDPOINT_PRED_COUNT_VLC_BITS: u32 = 4;

const NUM_ENDPOINT_PREDS: u8 = 3;
const CR_ENDPOINT_PRED_INDEX: u8 = NUM_ENDPOINT_PREDS - 1;
const NO_ENDPOINT_PRED_INDEX: u8 = 3;

const MAX_SELECTOR_HISTORY_BUF_SIZE: u32 = 64;
const SELECTOR_HISTORY_BUF_RLE_COUNT_THRESH: u32 = 3;
const SELECTOR_HISTORY_BUF_RLE_COUNT_BITS: u32 = 6;
const SELECTOR_HISTORY_BUF_RLE_COUNT_TOTAL: u32 = 1 << SELECTOR_HISTORY_BUF_RLE_COUNT_BITS;

// const int COLOR5_PAL0_PREV_HI = 9, COLOR5_PAL0_DELTA_LO = -9, COLOR5_PAL0_DELTA_HI = 31;
// const int COLOR5_PAL1_PREV_HI = 21, COLOR5_PAL1_DELTA_LO = -21, COLOR5_PAL1_DELTA_HI = 21;
// const int COLOR5_PAL2_PREV_HI = 31, COLOR5_PAL2_DELTA_LO = -31, COLOR5_PAL2_DELTA_HI = 9;

const COLOR5_PAL0_PREV_LO: u8 = 0;
const COLOR5_PAL0_PREV_HI: u8 = 9;
const COLOR5_PAL0_DELTA_LO: i32 = -9;
const COLOR5_PAL0_DELTA_HI: i32 = 31;
const COLOR5_PAL1_PREV_LO: u8 = 10;
const COLOR5_PAL1_PREV_HI: u8 = 21;
const COLOR5_PAL1_DELTA_LO: i32 = -21;
const COLOR5_PAL1_DELTA_HI: i32 = 21;
const COLOR5_PAL2_PREV_LO: u8 = 22;
const COLOR5_PAL2_PREV_HI: u8 = 31;
const COLOR5_PAL2_DELTA_LO: i32 = -31;
const COLOR5_PAL2_DELTA_HI: i32 = 9;

const INTENS: [[i16; 4]; 8] = [
    [-8, -2, 2, 8],
    [-17, -5, 5, 17],
    [-29, -9, 9, 29],
    [-42, -13, 13, 42],
    [-60, -18, 18, 60],
    [-80, -24, 24, 80],
    [-106, -33, 33, 106],
    [-183, -47, 47, 183],
];

pub struct DecodedBlock {
    block_x: u32,
    block_y: u32,
    endpoint_index: u16,
    selector_index: u16,
}

pub struct Decoder {
    endpoint_pred_model: HuffmanDecodingTable,
    delta_endpoint_model: HuffmanDecodingTable,
    selector_model: HuffmanDecodingTable,
    selector_history_buf_rle_model: HuffmanDecodingTable,

    selector_history_buffer_size: u32,
    is_video: bool,
    y_flipped: bool,

    endpoints: Vec<Endpoint>,
    selectors: Vec<Selector>,
}

impl Decoder {
    pub(crate) fn from_file_bytes(header: &Header, bytes: &[u8]) -> Result<Self> {
        let endpoints = {
            let num_endpoints = header.total_endpoints as usize;
            let start = header.endpoint_cb_file_ofs as usize;
            let len = header.endpoint_cb_file_size as usize;
            decode_endpoints(num_endpoints, &bytes[start..start + len])?
        };

        let selectors = {
            let num_selectors = header.total_selectors as usize;
            let start = header.selector_cb_file_ofs as usize;
            let len = header.selector_cb_file_size as usize;
            decode_selectors(num_selectors, &bytes[start..start + len])?
        };

        let start = header.tables_file_ofs as usize;
        let len = header.tables_file_size as usize;

        let reader = &mut BitReaderLSB::new(&bytes[start..start + len]);

        let endpoint_pred_model = huffman::read_huffman_table(reader)?;
        let delta_endpoint_model = huffman::read_huffman_table(reader)?;
        let selector_model = huffman::read_huffman_table(reader)?;
        let selector_history_buf_rle_model = huffman::read_huffman_table(reader)?;
        let selector_history_buffer_size = reader.read_u32(13);

        Ok(Self {
            endpoint_pred_model,
            delta_endpoint_model,
            selector_model,
            selector_history_buf_rle_model,
            selector_history_buffer_size,
            endpoints,
            selectors,
            is_video: header.tex_type == TextureType::VideoFrames as u8,
            y_flipped: header.has_y_flipped(),
        })
    }

    pub(crate) fn decode_to_rgba(&self, rgb_desc: &SliceDesc, alpha_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<Color32>> {
        if !alpha_desc.has_alpha() {
            return Err("Expected slice with alpha".into());
        }
        let mut rgb = self.decode_to_rgb(rgb_desc, bytes)?;
        let alpha = self.decode_to_rgb(alpha_desc, bytes)?;
        for (rgb, alpha) in rgb.data.iter_mut().zip(alpha.data.iter()) {
            rgb.0[3] = alpha.0[1];
        }
        Ok(rgb)
    }

    pub(crate) fn decode_to_rgb(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<Color32>> {
        let num_blocks_x = slice_desc.num_blocks_x as u32;
        let num_blocks_y = slice_desc.num_blocks_y as u32;

        let mut pixels = vec![Color32::default(); (num_blocks_x * num_blocks_y) as usize * 16];

        let block_to_rgb = |block: DecodedBlock| {
            let endpoint: Endpoint = self.endpoints[block.endpoint_index as usize];
            let selector: Selector = self.selectors[block.selector_index as usize];

            let modifiers = INTENS[endpoint.inten5 as usize];

            let mut colors: [Color32; 4] = [endpoint.color5; 4];

            fn extend_5_to_8(x: u8) -> u8 {
                (x << 3) | (x >> 2)
            }

            for i in 0..4 {
                let modifier = modifiers[i];
                for c in 0..3 {
                    let val = extend_5_to_8(colors[i].0[c]) as i16 + modifier;
                    colors[i].0[c] = i16::max(0, i16::min(val, 255)) as u8;
                }
            }

            let block_pos_x = (block.block_x * 4) as usize;
            let block_pos_y = (block.block_y * 4) as usize;
            let stride = (num_blocks_x * 4) as usize;

            for y in 0..4 {
                for x in 0..4 {
                    let sel = selector.get_selector(x, y);
                    let mut col = colors[sel];
                    col.0[3] = 0xFF;
                    let gx = block_pos_x + x;
                    let gy = block_pos_y + y;
                    let gid = gx + gy * stride;
                    pixels[gid] = col;
                }
            }
        };

        self.decode_blocks(slice_desc, bytes, block_to_rgb)?;

        Ok(Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: slice_desc.num_blocks_x as u32 * 4,
            y_flipped: self.y_flipped,
            data: pixels,
        })
    }

    pub(crate) fn transcode_to_etc1(&self, slice_desc: &SliceDesc, bytes: &[u8]) -> Result<Image<u8>> {
        let num_blocks_x = slice_desc.num_blocks_x as u32;
        let num_blocks_y = slice_desc.num_blocks_y as u32;

        const BLOCK_SIZE: usize = 8;

        let block_count = (num_blocks_x * num_blocks_y) as usize;

        let mut blocks = vec![0u8; BLOCK_SIZE * block_count];

        let block_to_etc1 = |block: DecodedBlock| {
            let endpoint: Endpoint = self.endpoints[block.endpoint_index as usize];
            let selector: Selector = self.selectors[block.selector_index as usize];

            let block_id = (block.block_y * num_blocks_x + block.block_x) as usize;
            let block_start = BLOCK_SIZE * block_id;
            let block = &mut blocks[block_start..block_start + BLOCK_SIZE];

            // color_r: 5 | delta: 3
            block[0] = endpoint.color5[0] << 3;
            // color_g: 5 | delta: 3
            block[1] = endpoint.color5[1] << 3;
            // color_b: 5 | delta: 3
            block[2] = endpoint.color5[2] << 3;
            // codeword: 3 | codeword: 3 | diff: 1 | flip: 1
            block[3] = endpoint.inten5 << 5 | endpoint.inten5 << 2 | 0b11;
            // selector bits: 16 x 2 bits
            block[4..].copy_from_slice(&selector.etc1_bytes);
        };

        self.decode_blocks(slice_desc, bytes, block_to_etc1)?;

        Ok(Image {
            w: slice_desc.orig_width as u32,
            h: slice_desc.orig_height as u32,
            stride: BLOCK_SIZE as u32 * slice_desc.num_blocks_x as u32,
            y_flipped: self.y_flipped,
            data: blocks,
        })
    }

    pub(crate) fn decode_blocks<F>(&self, slice_desc: &SliceDesc, bytes: &[u8], mut f: F) -> Result<()>
        where F: FnMut(DecodedBlock)
    {

        let reader = {
            let start = slice_desc.file_ofs as usize;
            let len = slice_desc.file_size as usize;
            &mut BitReaderLSB::new(&bytes[start..start+len])
        };

        let num_endpoints = self.endpoints.len() as u16;
        let num_selectors = self.selectors.len() as u16;

        // Endpoint/selector codebooks - decoded previously. See sections 7.0 and 8.0.

        let num_blocks_x = slice_desc.num_blocks_x as u32;
        let num_blocks_y = slice_desc.num_blocks_y as u32;

        #[derive(Clone, Copy, Default)]
        struct BlockPreds {
            endpoint_index: u16,
            pred_bits: u8,
        }
        // Array of per-block values used for endpoint index prediction (enough for 2 rows).
        let mut block_endpoint_preds: [Vec<BlockPreds>; 2] = [
            vec![BlockPreds::default(); num_blocks_x as usize],
            vec![BlockPreds::default(); num_blocks_x as usize],
        ];

        // Some constants and state used during block decoding
        let selector_history_buf_first_symbol_index: u16 = num_selectors;
        let selector_history_buf_rle_symbol_index: u16 = self.selector_history_buffer_size as u16 + selector_history_buf_first_symbol_index;
        let mut cur_selector_rle_count: u32 = 0;

        let mut cur_pred_bits: u8 = 0;
        let mut prev_endpoint_pred_sym: u8 = 0;
        let mut endpoint_pred_repeat_count: u32 = 0;
        let mut prev_endpoint_index: u16 = 0;

        #[derive(Clone, Copy, Default)]
        struct PrevFrameIndices {
            endpoint_index: u16,
            selector_index: u16,
        }
        // This array is only used for texture video. It holds the previous frame's endpoint and selector indices (each 16-bits, for 32-bits total).
        let mut prev_frame_indices = vec![PrevFrameIndices::default(); (num_blocks_x * num_blocks_y) as usize];

        // Selector history buffer - See section 10.1.
        // For the selector history buffer's size, see section 9.0.
        let mut selector_history_buf = ApproxMoveToFront::new(self.selector_history_buffer_size as usize);

        // Loop over all slice blocks in raster order
        for block_y in 0..num_blocks_y {
            // The index into the block_endpoint_preds array
            let cur_block_endpoint_pred_array: u32 = block_y & 1;

            for block_x in 0..num_blocks_x {
                // Check if we're at the start of a 2x2 block group.
                if block_x & 1 == 0 {
                    // Are we on an even or odd row of blocks?
                    if block_y & 1 == 0 {
                        // We're on an even row and column of blocks. Decode the combined endpoint index predictor symbols for 2x2 blocks.
                        // This symbol tells the decoder how the endpoints are decoded for each block in a 2x2 group of blocks.

                        // Are we in an RLE run?
                        if endpoint_pred_repeat_count != 0 {
                            // Inside a run of endpoint predictor symbols.
                            endpoint_pred_repeat_count -= 1;
                            cur_pred_bits = prev_endpoint_pred_sym;
                        } else {
                            // Decode the endpoint prediction symbol, using the "endpoint pred" Huffman table (see section 9.0).
                            let pred_bits_sym = self.endpoint_pred_model.decode_symbol(reader)?;
                            if pred_bits_sym == ENDPOINT_PRED_REPEAT_LAST_SYMBOL {
                                // It's a run of symbols, so decode the count using VLC decoding (see section 10.2)
                                endpoint_pred_repeat_count = decode_vlc(reader, ENDPOINT_PRED_COUNT_VLC_BITS) + ENDPOINT_PRED_MIN_REPEAT_COUNT - 1;
                                cur_pred_bits = prev_endpoint_pred_sym;
                            } else {
                                // It's not a run of symbols
                                cur_pred_bits = pred_bits_sym as u8;
                                prev_endpoint_pred_sym = cur_pred_bits;
                            }
                        }

                        // The symbol has enough endpoint prediction information for 4 blocks (2 bits per block), so 8 bits total.
                        // Remember the prediction information we should use for the next row of 2 blocks beneath the current block.
                        block_endpoint_preds[cur_block_endpoint_pred_array as usize ^ 1][block_x as usize].pred_bits = (cur_pred_bits >> 4) as u8;
                    } else {
                        // We're on an odd row of blocks, so use the endpoint prediction information we previously stored on the previous even row.
                        cur_pred_bits = block_endpoint_preds[cur_block_endpoint_pred_array as usize][block_x as usize].pred_bits;
                    }
                }

                // Decode the current block's endpoint and selector indices.

                // Get the 2-bit endpoint prediction index for this block.
                let pred: u8 = cur_pred_bits & 3;

                // Get the next block's endpoint prediction bits ready.
                cur_pred_bits >>= 2;

                // Now check to see if we should reuse a previously encoded block's endpoints.
                let endpoint_index = match pred {
                    0 => {
                        // Reuse the left block's endpoint index
                        assert!(block_x > 0, "block_x: {}, block_y: {}, cur_pred_bits: {}", block_x, block_y, cur_pred_bits);
                        prev_endpoint_index
                    }
                    1 => {
                        // Reuse the upper block's endpoint index
                        assert!(block_y > 0, "block_x: {}, block_y: {}, cur_pred_bits: {}", block_x, block_y, cur_pred_bits);
                        block_endpoint_preds[cur_block_endpoint_pred_array as usize ^ 1][block_x as usize].endpoint_index as u16
                    }
                    2 => {
                        if self.is_video {
                            // If it's texture video, reuse the previous frame's endpoint index, at this block.
                            assert_eq!(pred, CR_ENDPOINT_PRED_INDEX);
                            prev_frame_indices[(block_x + block_y * num_blocks_x) as usize].endpoint_index
                        } else {
                            // Reuse the upper left block's endpoint index.
                            assert!(block_x > 0);
                            assert!(block_y > 0);
                            block_endpoint_preds[cur_block_endpoint_pred_array as usize ^ 1][block_x as usize - 1].endpoint_index
                        }
                    }
                    _ => {
                        // We need to decode and apply a DPCM encoded delta to the previously used endpoint index.
                        // This uses the delta endpoint Huffman table (see section 9.0).
                        let delta_sym = self.delta_endpoint_model.decode_symbol(reader)?;

                        let mut endpoint_index = delta_sym + prev_endpoint_index;

                        // Wrap around if the index goes beyond the end of the endpoint codebook
                        if endpoint_index >= num_endpoints {
                            endpoint_index -= num_endpoints;
                        }

                        endpoint_index
                    }
                };

                // Remember the endpoint index we used on this block, so the next row can potentially reuse the index.
                block_endpoint_preds[cur_block_endpoint_pred_array as usize][block_x as usize].endpoint_index = endpoint_index;

                // Remember the endpoint index used
                prev_endpoint_index = endpoint_index;

                // Now we have fully decoded the ETC1S endpoint codebook index, in endpoint_index.

                // Now decode the selector index (see the next block of code, below).

                // Decode selector index, unless it's texture video and the endpoint predictor indicated that the
                // block's endpoints were reused from the previous frame.
                let selector_index = if !self.is_video || pred != CR_ENDPOINT_PRED_INDEX {
                    // Are we in a selector RLE run?
                    let selector_sym: u16 = if cur_selector_rle_count > 0 {
                        cur_selector_rle_count -= 1;
                        num_selectors
                    } else {
                        // Decode the selector symbol, using the selector Huffman table (see section 9.0).
                        let sym = self.selector_model.decode_symbol(reader)?;

                        // Is it a run?
                        if sym == selector_history_buf_rle_symbol_index as u16 {
                            // Decode the selector run's size, using the selector history buf RLE Huffman table (see section 9.0).
                            let run_sym = self.selector_history_buf_rle_model.decode_symbol(reader)? as u32;

                            // Is it a very long run?
                            if run_sym == (SELECTOR_HISTORY_BUF_RLE_COUNT_TOTAL - 1) {
                                cur_selector_rle_count = SELECTOR_HISTORY_BUF_RLE_COUNT_THRESH + decode_vlc(reader, 7);
                            } else {
                                cur_selector_rle_count = SELECTOR_HISTORY_BUF_RLE_COUNT_THRESH + run_sym;
                            }

                            cur_selector_rle_count -= 1;

                            num_selectors
                        } else {
                            sym
                        }
                    };

                    // Is it a reference into the selector history buffer?
                    if selector_sym >= num_selectors {
                        assert!(self.selector_history_buffer_size > 0);

                        // Compute the history buffer index
                        let history_buf_index = (selector_sym - num_selectors) as usize;

                        assert!(history_buf_index < selector_history_buf.size());

                        // Access the history buffer
                        let index = selector_history_buf[history_buf_index];

                        // Update the history buffer
                        if history_buf_index != 0 {
                            selector_history_buf.use_index(history_buf_index);
                        }

                        index
                    } else {
                        // It's an index into the selector codebook
                        // Add it to the selector history buffer
                        if self.selector_history_buffer_size > 0 {
                            selector_history_buf.add(selector_sym);
                        }
                        selector_sym
                    }
                } else {
                    // If it's texture video, reuse the previous frame's selector index, at this block.
                    prev_frame_indices[(block_x + block_y * num_blocks_x) as usize].selector_index
                };

                // For texture video, remember the endpoint and selector indices used by the block on this frame, for later reuse on the next frame.
                if self.is_video {
                    let curr_frame_indices = &mut prev_frame_indices[(block_x + num_blocks_x * block_y) as usize];
                    curr_frame_indices.endpoint_index = endpoint_index;
                    curr_frame_indices.selector_index = selector_index;
                }

                // The block is fully decoded here. The codebook indices are endpoint_index and selector_index.
                // Make sure they are valid
                assert!(endpoint_index < num_endpoints);
                assert!(selector_index < num_selectors);

                let block = DecodedBlock {
                    block_x, block_y, endpoint_index, selector_index
                };

                f(block);
            }
        }

        Ok(())
    }
}

fn decode_endpoints(num_endpoints: usize, bytes: &[u8]) -> Result<Vec<Endpoint>> {
    let reader = &mut BitReaderLSB::new(bytes);

    let color5_delta_model0 = huffman::read_huffman_table(reader)?;
    let color5_delta_model1 = huffman::read_huffman_table(reader)?;
    let color5_delta_model2 = huffman::read_huffman_table(reader)?;
    let inten_delta_model = huffman::read_huffman_table(reader)?;
    let grayscale = reader.read_bool();

    // Assume previous endpoint color is (16, 16, 16), and the previous intensity is 0.
    let mut prev_color5 = Color32::new(16, 16, 16, 0);
    let mut prev_inten: u32 = 0;

    let mut endpoints: Vec<Endpoint> = vec![Endpoint::default(); num_endpoints as usize];

    // For each endpoint codebook entry
    for endpoint in &mut endpoints {

        // Decode the intensity delta Huffman code
        let inten_delta = inten_delta_model.decode_symbol(reader)?;
        endpoint.inten5 = ((inten_delta as u32 + prev_inten) & 7) as u8;
        prev_inten = endpoint.inten5 as u32;

        // Now decode the endpoint entry's color or intensity value
        let channel_count = if grayscale { 1 } else { 3 };
        for c in 0..channel_count {

            // The Huffman table used to decode the delta depends on the previous color's value
            let delta = match prev_color5[c as usize] {
                COLOR5_PAL0_PREV_LO..=COLOR5_PAL0_PREV_HI => color5_delta_model0.decode_symbol(reader)?,
                COLOR5_PAL1_PREV_LO..=COLOR5_PAL1_PREV_HI => color5_delta_model1.decode_symbol(reader)?,
                COLOR5_PAL2_PREV_LO..=COLOR5_PAL2_PREV_HI => color5_delta_model2.decode_symbol(reader)?,
                _ => unreachable!()
            };

            // Apply the delta
            let v = prev_color5[c].wrapping_add(delta as u8) & 31;

            endpoint.color5[c] = v;

            prev_color5[c] = v;
        }

        // If the endpoints are grayscale, set G and B to match R.
        if grayscale {
            endpoint.color5[1] = endpoint.color5[0];
            endpoint.color5[2] = endpoint.color5[0];
        }
    }

    Ok(endpoints)
}

#[derive(Clone, Copy, Debug, Default)]
struct Endpoint {
    inten5: u8,
    color5: Color32,
}

#[derive(Clone, Copy, Debug,  Default)]
struct Selector {
    // Plain selectors (2-bits per value), one byte for each row
    selectors: [u8; 4],

    // Selectors in ETC1 format, ready to be written to an ETC1 texture
    etc1_bytes: [u8; 4],
}

impl Selector {

    // Returned selector value ranges from 0-3 and is a direct index into g_etc1_inten_tables.
    fn get_selector(&self, x: usize, y: usize) -> usize {
        assert!(x < 4);
        assert!(y < 4);

        let shift = 2 * x;
        let val = (self.selectors[y] >> shift) & 0b11;
        val as usize
    }

    fn set_selector(&mut self, x: usize, y: usize, val: u8) {
        assert!(x < 4);
        assert!(y < 4);
        assert!(val < 4);

        // Pack the two-bit value into the byte for the appropriate row
        let shift = 2 * x;
        self.selectors[y] &= !(0b11 << shift);
        self.selectors[y] |= (val as u8) << shift;

        // Convert to ETC1 format according to the spec
        let mod_id: u8 = [0b11, 0b10, 0b00, 0b01][val as usize];

        // ETC1 indexes pixels from top to bottom within each column
        let pixel_id = x * 4 + y;

        // MS bit of pixel 0..8 goes to byte 1
        // MS bit of pixel 8..16 goes to byte 0
        let ms_byte_id = 1 - (pixel_id / 8);

        // LS bit of pixel 0..8 goes to byte 3
        // LS bit of pixel 8..16 goes to byte 2
        let ls_byte_id = ms_byte_id + 2;

        let bit_id = pixel_id % 8;

        self.etc1_bytes[ls_byte_id] &= !(1 << bit_id);
        self.etc1_bytes[ls_byte_id] |= (mod_id % 2) << bit_id;
        self.etc1_bytes[ms_byte_id] &= !(1 << bit_id);
        self.etc1_bytes[ms_byte_id] |= (mod_id / 2) << bit_id;
    }
}

fn decode_selectors(num_selectors: usize, bytes: &[u8]) -> Result<Vec<Selector>> {
    let reader = &mut BitReaderLSB::new(bytes);

    let global = reader.read_bool();
    let hybrid = reader.read_bool();
    let raw = reader.read_bool();

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

        for (i, selector) in selectors.iter_mut().enumerate() {
            if i == 0 {
                // First selector is sent raw
                for (y, prev_byte) in prev_bytes.iter_mut().enumerate() {
                    let cur_byte = reader.read_u8(8);
                    *prev_byte = cur_byte;

                    for x in 0..4 {
                        selector.set_selector(x, y, (cur_byte >> (x*2)) & 3);
                    }
                }
            } else {
                // Subsequent selectors are sent with a simple form of byte-wise DPCM coding.
                for (y, prev_byte) in prev_bytes.iter_mut().enumerate() {
                    let delta_byte = delta_selector_pal_model.decode_symbol(reader)? as u8;

                    let cur_byte = delta_byte ^ *prev_byte;
                    *prev_byte = cur_byte;

                    for x in 0..4 {
                        selector.set_selector(x, y, (cur_byte >> (x*2)) & 3);
                    }
                }
            }
        }
    } else {
        for selector in &mut selectors {
            for y in 0..4 {
                let cur_byte = reader.read_u8(8);
                for x in 0..4 {
                    selector.set_selector(x, y, (cur_byte >> (x*2)) & 3);
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
        let s = reader.read_u32(chunk_bits as usize + 1);
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


struct ApproxMoveToFront {
    values: Vec<u16>,
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

    fn add(&mut self, new_value: u16) {
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
    type Output = u16;
    fn index<'a>(&'a self, i: usize) -> &'a Self::Output {
        &self.values[i]
    }
}

impl IndexMut<usize> for ApproxMoveToFront {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut Self::Output {
        &mut self.values[i]
    }
}
