use crate::{
    Color32,
    Result,
    bitreader::BitReaderLsb,
    bitwriter::BitWriterLsb,
    mask, uastc
};

use std::sync::Once;

pub fn convert_block_from_uastc(bytes: &[u8], output: &mut [u8]) {
    match convert_block_from_uastc_result(bytes, output) {
        Ok(_) => (),
        _ => output.copy_from_slice(&[0; 16]),
    }
}

fn convert_block_from_uastc_result(bytes: &[u8], output: &mut [u8]) -> Result<()> {
    let reader = &mut BitReaderLsb::new(bytes);

    let mode = uastc::decode_mode(reader)?;

    let writer = &mut BitWriterLsb::new(output);

    if mode.id == 8 {
        let rgba = uastc::decode_mode8_rgba(reader);

        let (mode, endpoint, p_bits, weights) = convert_mode_8_to_bc7_mode_endpoint_p_bits_weights(rgba);

        let bc7_mode = BC7_MODES[mode as usize];

        let weights = &weights[0..bc7_mode.plane_count as usize];

        writer.write_u8(mode as usize + 1, 1 << mode);

        if mode == 5 {
            writer.write_u8(2, 0);
        }

        for channel in 0..4 {
            let bit_count = if channel != ALPHA_CHANNEL { bc7_mode.color_bits } else { bc7_mode.alpha_bits } as usize;
            writer.write_u8(bit_count, endpoint[0][channel]);
            writer.write_u8(bit_count, endpoint[1][channel]);
        }

        if mode == 6 {
            writer.write_u8(2, (p_bits[1] << 1) | p_bits[0]);
        }

        {   // Write weights
            let bit_count = bc7_mode.weight_bits as usize;
            for &weight in weights.iter() {
                writer.write_u8(bit_count - 1, weight);
                for _ in 0..15 {
                    writer.write_u8(bit_count as usize, weight);
                }
            }
        }
        return Ok(())
    }

    let bc7_mode_index = UASTC_TO_BC7_MODES[mode.id as usize];
    let bc7_mode = BC7_MODES[bc7_mode_index as usize];

    uastc::skip_trans_flags(reader, mode);

    let compsel = uastc::decode_compsel(reader, mode);
    let uastc_pat = uastc::decode_pattern_index(reader, mode)?;

    let bc7_plane_count = bc7_mode.plane_count as usize;
    let bc7_subset_count = bc7_mode.subset_count as usize;
    let bc7_endpoints_per_channel = 2 * bc7_subset_count;
    let bc7_channel_count = bc7_mode.endpoint_count as usize / bc7_endpoints_per_channel;

    let mut endpoints = {
        let endpoint_count = mode.endpoint_count();
        let quant_endpoints = uastc::decode_endpoints(reader, mode.endpoint_range_index, endpoint_count);
        let mut unquant_endpoints = [0; 18];
        for (quant, unquant) in quant_endpoints.iter().zip(unquant_endpoints.iter_mut()).take(endpoint_count) {
            *unquant = uastc::unquant_endpoint(*quant, mode.endpoint_range_index);
        }
        uastc::assemble_endpoint_pairs(mode, &unquant_endpoints)
    };

    let mut weights = [[0; 16]; 2];
    {
        if mode.plane_count == 1 {
            uastc::decode_weights(reader, mode, uastc_pat, |i, w| {
                weights[0][i] = w;
            });
            convert_weights_to_bc7(&mut weights[0], mode.weight_bits, bc7_mode.weight_bits);
        } else {
            uastc::decode_weights(reader, mode, uastc_pat, |i, w| {
                let plane = i & 1;
                let wi = i >> 1;
                weights[plane][wi] = w;
            });
            convert_weights_to_bc7(&mut weights[0], mode.weight_bits, bc7_mode.weight_bits);
            convert_weights_to_bc7(&mut weights[1], mode.weight_bits, bc7_mode.weight_bits);
        }
    }

    let endpoints = &mut endpoints[0..bc7_subset_count];
    let weights = &mut weights[0..bc7_plane_count];

    // Write block mode
    writer.write_u8(bc7_mode_index as usize + 1, 1 << bc7_mode_index);

    let mut bc7_anchors: &[u8] = &[0];

    const ALPHA_CHANNEL: usize = 3;

    // Write partition bits
    if bc7_subset_count > 1 {
        let (bc7_pat, pattern, anchors, perm): (_, _, &[u8], &[u8]) = match (mode.id, mode.subset_count) {
            (1, _) => {
                let (index, _) = PATTERNS_2_BC7_INDEX_INV[0];
                (index, &PATTERNS_2_BC7[uastc_pat as usize], &PATTERNS_2_BC7_ANCHORS[index as usize], &[0, 0])
            }
            (7, _) => {
                let (index, p) = PATTERNS_2_3_BC7_INDEX_PERM[uastc_pat as usize];
                let perm = &PATTERNS_2_3_BC7_TO_ASTC_PERMUTATIONS[p as usize];
                (index, &PATTERNS_2_3_BC7[uastc_pat as usize], &PATTERNS_3_BC7_ANCHORS[index as usize], perm)
            }
            (_, 2) => {
                let (index, inv) = PATTERNS_2_BC7_INDEX_INV[uastc_pat as usize];
                (index, &PATTERNS_2_BC7[uastc_pat as usize], &PATTERNS_2_BC7_ANCHORS[index as usize], if inv { &[1, 0] } else { &[0, 1] })
            }
            (_, 3) => {
                let (index, p) = PATTERNS_3_BC7_INDEX_PERM[uastc_pat as usize];
                let perm = &PATTERNS_3_BC7_TO_ASTC_PERMUTATIONS[p as usize];
                (index, &PATTERNS_3_BC7[uastc_pat as usize], &PATTERNS_3_BC7_ANCHORS[index as usize], perm)
            }
            _ => unreachable!()
        };
        bc7_anchors = anchors;

        writer.write_u8(bc7_mode.pat_bits as usize, bc7_pat);

        {   // Permute endpoints
            let mut permuted_endpoints = [[Color32::default(); 2]; 3];
            permute(&endpoints, &mut permuted_endpoints, perm);
            endpoints[0..bc7_subset_count].copy_from_slice(&permuted_endpoints[0..bc7_subset_count]);
        }

        {   // Swap endpoints and invert weights if anchor weight MSB is not 0
            let weight_mask = mask!(bc7_mode.weight_bits);
            let weight_msb_mask = 1 << (bc7_mode.weight_bits - 1);

            // Check which subset to invert
            let mut invert_subset = [false; 3];
            for (&anchor, inv) in anchors.iter().zip(invert_subset.iter_mut()) {
                *inv = weights[0][anchor as usize] & weight_msb_mask != 0;
            }

            // Swap endpoints
            for (endpoint_pair, &inv) in endpoints.iter_mut().zip(invert_subset.iter()) {
                if inv {
                    endpoint_pair.swap(0, 1);
                }
            }

            // Invert weights
            for (weight, &subset) in weights[0].iter_mut().zip(pattern.iter()) {
                if invert_subset[subset as usize] {
                    *weight = !*weight & weight_mask;
                }
            }
        }
    } else {
        let weight_mask = mask!(bc7_mode.weight_bits);
        let weight_msb_mask = 1 << (bc7_mode.weight_bits - 1);

        if mode.plane_count == 1 {
            if weights[0][0] & weight_msb_mask != 0 {
                endpoints[0].swap(0, 1);
                for weight in weights[0].iter_mut() {
                    *weight = !*weight & weight_mask;
                }
            }
        } else {
            assert_eq!(mode.plane_count, 2);

            let invert_plane = [
                weights[0][0] & weight_msb_mask != 0,
                weights[1][0] & weight_msb_mask != 0,
            ];

            let endpoint_pair = &mut endpoints[0];

            // Apply channel rotation
            endpoint_pair[0].0.swap(compsel as usize, ALPHA_CHANNEL);
            endpoint_pair[1].0.swap(compsel as usize, ALPHA_CHANNEL);

            // Invert planes
            if invert_plane[0] {
                endpoint_pair.swap(0, 1);
            }
            if invert_plane[0] != invert_plane[1] {
                let [e0, e1] = endpoint_pair;
                std::mem::swap(&mut e0[ALPHA_CHANNEL], &mut e1[ALPHA_CHANNEL]);
            }

            for (&inv, weight_plane) in invert_plane.iter().zip(weights.iter_mut()) {
                if inv {
                    for w in weight_plane.iter_mut() {
                        *w = !*w & weight_mask;
                    }
                }
            }

            // Write rotation bits
            writer.write_u8(2, (compsel + 1) & 0b11);

            if bc7_mode.id == 4 {
                // Index selection bit, not used
                writer.write_u8(1, 0);
            }
        }
    }

    let color_bits = bc7_mode.color_bits as usize;
    let alpha_bits = bc7_mode.alpha_bits as usize;

    let mut p_bits = [[0u8; 2]; 3];
    if bc7_mode.p_bits != 0 {
        for (endpoint_pair, p) in endpoints.iter_mut().zip(p_bits.iter_mut()) {
            *p = determine_unique_pbits(bc7_channel_count, bc7_mode.color_bits, endpoint_pair);
        }
    } else if bc7_mode.sp_bits != 0 {
        for (endpoint_pair, p) in endpoints.iter_mut().zip(p_bits.iter_mut()) {
            *p = determine_shared_pbits(bc7_channel_count, bc7_mode.color_bits, endpoint_pair);
        }
    } else {
        fn scale_endpoint_to_bc7(e: u8, bits: usize) -> u8 {
            ((e as u32 * mask!(bits as u32) + 127) / 255) as u8
        }
        for endpoint_pair in endpoints.iter_mut() {
            for e in endpoint_pair {
                for channel in 0..3 {
                    e[channel] = scale_endpoint_to_bc7(e[channel], color_bits);
                }
                e[ALPHA_CHANNEL] = scale_endpoint_to_bc7(e[ALPHA_CHANNEL], alpha_bits);
            }
        }
    }
    let p_bits = &mut p_bits[0..bc7_subset_count];

    for channel in 0..bc7_channel_count {
        let bit_count = if channel != ALPHA_CHANNEL { color_bits } else { alpha_bits };
        for e in endpoints.iter() {
            writer.write_u8(bit_count, e[0][channel]);
            writer.write_u8(bit_count, e[1][channel]);
        }
    }

    if bc7_mode.p_bits != 0 {
        for p in p_bits.iter() {
            writer.write_u8(2, (p[1] << 1) | p[0]);
        }
    } else if bc7_mode.sp_bits != 0 {
        writer.write_u8(2, (p_bits[1][0] << 1) | p_bits[0][0]);
    }

    {   // Write weights
        let mut bit_counts = [bc7_mode.weight_bits; 16];
        for &anchor in bc7_anchors {
            bit_counts[anchor as usize] -= 1;
        }
        for plane_weights in weights.iter() {
            for (&weight, &bit_count) in plane_weights.iter().zip(bit_counts.iter()) {
                writer.write_u8(bit_count as usize, weight);
            }
        }
    }

    Ok(())
}

fn convert_mode_8_to_bc7_mode_endpoint_p_bits_weights(solid_color: Color32) -> (u8, [Color32; 2], [u8; 2], [u8; 2]) {

    let (mode_5_optimal_endpoints, mode_6_optimal_endpoints) = get_mode_8_bc7_tables();

    // Compute the error from BC7 mode 6 p-bit 0
    let best_err0: u32 = solid_color.0.iter().map(|&c| mode_6_optimal_endpoints[c as usize][0].err as u32).sum();

    // Compute the error from BC7 mode 6 p-bit 1
    let best_err1: u32 = solid_color.0.iter().map(|&c| mode_6_optimal_endpoints[c as usize][1].err as u32).sum();

    let mode;
    let mut endpoint = [Color32::default(); 2];
    let mut p_bits = [0u8; 2];
    let mut weights = [0u8; 2];

    // Is BC7 mode 6 not lossless? If so, use mode 5 instead.
    if best_err0 > 0 && best_err1 > 0 {
        // Output BC7 mode 5
        mode = 5;

        // Convert the endpoints
        for c in 0..3 {
            endpoint[0][c] = mode_5_optimal_endpoints[solid_color[c] as usize].lo;
            endpoint[1][c] = mode_5_optimal_endpoints[solid_color[c] as usize].hi;
        }
        // Set the output alpha
        endpoint[0][3] = solid_color[3];
        endpoint[1][3] = solid_color[3];

        // Set the output BC7 selectors/weights to all 1's
        weights[0] = BC7ENC_MODE_5_OPTIMAL_INDEX;

        // Set the output BC7 alpha selectors/weights to all 0's
        weights[1] = 0;
    } else {
        // Output BC7 mode 6
        mode = 6;

        // Choose the p-bit with minimal error
        let best_p = if best_err1 < best_err0 { 1 } else { 0 };

        // Convert the components
        for c in 0..4 {
            endpoint[0][c] = mode_6_optimal_endpoints[solid_color[c] as usize][best_p as usize].lo;
            endpoint[1][c] = mode_6_optimal_endpoints[solid_color[c] as usize][best_p as usize].hi;
        }

        // Set the output p-bits
        p_bits[0] = best_p;
        p_bits[1] = best_p;

        // Set the output BC7 selectors/weights to all 5's
        weights = [BC7ENC_MODE_6_OPTIMAL_INDEX, BC7ENC_MODE_6_OPTIMAL_INDEX];
    }

    (mode, endpoint, p_bits, weights)
}

fn convert_weights_to_bc7(weights: &mut [u8; 16], uastc_weight_bits: u8, bc7_weight_bits: u8) {
    const UASTC_1_TO_BC7_2: &[u8; 2] = &[ 0, 3 ];
    const UASTC_2_TO_BC7_4: &[u8; 4] = &[ 0, 5, 10, 15 ];
    const UASTC_3_TO_BC7_4: &[u8; 8] = &[ 0, 2, 4, 6, 9, 11, 13, 15 ];
    const UASTC_5_TO_BC7_4: &[u8; 32] = &[ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15 ];

    let lut = match (uastc_weight_bits, bc7_weight_bits) {
        (1, 2) => &UASTC_1_TO_BC7_2[..],
        (2, 4) => &UASTC_2_TO_BC7_4[..],
        (3, 4) => &UASTC_3_TO_BC7_4[..],
        (5, 4) => &UASTC_5_TO_BC7_4[..],
        (a, b) if a == b => return,
        _ => unreachable!()
    };

    for weight in weights {
        *weight = lut[*weight as usize];
    }
}

fn permute<T: Copy>(src: &[T], dst: &mut[T], src_for_dst: &[u8]) {
    // dst[X] = src[src_for_dst[X]]
    for (&src_id, dst_pair) in src_for_dst.iter().zip(dst.iter_mut()) {
        *dst_pair = src[src_id as usize];
    }
}

// Determines the best shared pbits to use to encode xl/xh
fn determine_shared_pbits(
    total_comps: usize, comp_bits: u8, endpoint_pair: &mut [Color32; 2]
) -> [u8; 2] {
    let total_bits = comp_bits + 1;
    assert!(total_bits >= 4 && total_bits <= 8);

    let iscalep = (1 << total_bits) - 1;
    let scalep = iscalep as f32;

    let [xl_col, xh_col] = endpoint_pair;

    let mut xl = [0f32; 4];
    for (f, &i) in xl.iter_mut().zip(xl_col.0.iter()) {
        *f = i as f32 / 255.;
    }
    let mut xh = [0f32; 4];
    for (f, &i) in xh.iter_mut().zip(xh_col.0.iter()) {
        *f = i as f32 / 255.;
    }

    *xl_col = Default::default();
    *xh_col = Default::default();

    let mut best_err = 1e+9f32;

    let mut s_bit = 0;

    for p in 0..2 {
        let mut x_min_col = Color32::default();
        let mut x_max_col = Color32::default();
        for c in 0..4 {
            x_min_col[c] = (((xl[c] * scalep - p as f32) / 2. + 0.5) as i32 * 2 + p).max(p).min(iscalep - 1 + p) as u8;
            x_max_col[c] = (((xh[c] * scalep - p as f32) / 2. + 0.5) as i32 * 2 + p).max(p).min(iscalep - 1 + p) as u8;
        }

        let mut scaled_low = Color32::default();
        let mut scaled_high = Color32::default();

        for i in 0..4 {
            scaled_low[i] = x_min_col[i] << (8 - total_bits);
            scaled_low[i] |= scaled_low[i] >> total_bits;

            scaled_high[i] = x_max_col[i] << (8 - total_bits);
            scaled_high[i] |= scaled_high[i] >> total_bits;
        }

        let mut err = 0.;
        for i in 0..total_comps {
            err += (scaled_low[i] as f32 / 255. - xl[i]).powi(2) + (scaled_high[i] as f32 / 255. - xh[i]).powi(2);
        }

        if err < best_err {
            best_err = err;
            s_bit = p as u8;
            for j in 0..4 {
                xl_col[j] = x_min_col[j] >> 1;
                xh_col[j] = x_max_col[j] >> 1;
            }
        }
    }

    [s_bit, s_bit]
}

// Determines the best unique pbits to use to encode xl/xh
fn determine_unique_pbits(
    total_comps: usize, comp_bits: u8, endpoint_pair: &mut [Color32; 2]
) -> [u8; 2] {
    let total_bits = comp_bits + 1;
    let iscalep = (1 << total_bits) - 1;
    let scalep = iscalep as f32;

    let [xl_col, xh_col] = endpoint_pair;

    let mut xl = [0f32; 4];
    for (f, &i) in xl.iter_mut().zip(xl_col.0.iter()) {
        *f = i as f32 / 255.;
    }
    let mut xh = [0f32; 4];
    for (f, &i) in xh.iter_mut().zip(xh_col.0.iter()) {
        *f = i as f32 / 255.;
    }

    *xl_col = Default::default();
    *xh_col = Default::default();

    let mut best_err0 = 1e+9f32;
    let mut best_err1 = 1e+9f32;

    let mut p_bits: [u8; 2] = Default::default();

    for p in 0..2 {
        let mut x_min_color = Color32::default();
        let mut x_max_color = Color32::default();

        for c in 0..4 {
            x_min_color[c] = (((xl[c] * scalep - p as f32) / 2. + 0.5) as i32 * 2 + p).max(p).min(iscalep - 1 + p) as u8;
            x_max_color[c] = (((xh[c] * scalep - p as f32) / 2. + 0.5) as i32 * 2 + p).max(p).min(iscalep - 1 + p) as u8;
        }

        let mut scaled_low = Color32::default();
        let mut scaled_high = Color32::default();

        for i in 0..4 {
            scaled_low[i] = x_min_color[i] << (8 - total_bits);
            scaled_low[i] |= scaled_low[i].wrapping_shr(total_bits as u32);

            scaled_high[i] = x_max_color[i] << (8 - total_bits);
            scaled_high[i] |= scaled_high[i].wrapping_shr(total_bits as u32);
        }

        let mut err0 = 0.;
        let mut err1 = 0.;
        for i in 0..total_comps {
            err0 += (scaled_low[i] as f32 - xl[i] * 255.).powi(2);
            err1 += (scaled_high[i] as f32 - xh[i] * 255.).powi(2);
        }

        if err0 < best_err0 {
            best_err0 = err0;
            p_bits[0] = p as u8;
            for j in 0..4 {
                xl_col[j] = x_min_color[j] >> 1;
            }
        }

        if err1 < best_err1 {
            best_err1 = err1;
            p_bits[1] = p as u8;
            for j in 0..4 {
                xh_col[j] = x_max_color[j] >> 1;
            }
        }
    }

    p_bits
}

#[derive(Clone, Copy, Debug, Default)]
struct Bc7Mode {
    id: u8,
    pat_bits: u8,
    endpoint_count: u8,
    color_bits: u8,
    alpha_bits: u8,
    weight_bits: u8,
    plane_count: u8,
    subset_count: u8,
    p_bits: u8,
    sp_bits: u8,
}

static BC7_MODES: [Bc7Mode; 8] = [
    Bc7Mode { id: 0, endpoint_count: 18, color_bits: 4, alpha_bits: 0, weight_bits: 3, plane_count: 1, subset_count: 3, pat_bits: 4, p_bits: 1, sp_bits: 0 },
    Bc7Mode { id: 1, endpoint_count: 12, color_bits: 6, alpha_bits: 0, weight_bits: 3, plane_count: 1, subset_count: 2, pat_bits: 6, p_bits: 0, sp_bits: 1 },
    Bc7Mode { id: 2, endpoint_count: 18, color_bits: 5, alpha_bits: 0, weight_bits: 2, plane_count: 1, subset_count: 3, pat_bits: 6, p_bits: 0, sp_bits: 0 },
    Bc7Mode { id: 3, endpoint_count: 12, color_bits: 7, alpha_bits: 0, weight_bits: 2, plane_count: 1, subset_count: 2, pat_bits: 6, p_bits: 1, sp_bits: 0 },
    Bc7Mode { id: 4, endpoint_count:  8, color_bits: 5, alpha_bits: 6, weight_bits: 2, plane_count: 2, subset_count: 1, pat_bits: 0, p_bits: 0, sp_bits: 0 },
    Bc7Mode { id: 5, endpoint_count:  8, color_bits: 7, alpha_bits: 8, weight_bits: 2, plane_count: 2, subset_count: 1, pat_bits: 0, p_bits: 0, sp_bits: 0 },
    Bc7Mode { id: 6, endpoint_count:  8, color_bits: 7, alpha_bits: 7, weight_bits: 4, plane_count: 1, subset_count: 1, pat_bits: 0, p_bits: 1, sp_bits: 0 },
    Bc7Mode { id: 7, endpoint_count: 16, color_bits: 5, alpha_bits: 5, weight_bits: 2, plane_count: 1, subset_count: 2, pat_bits: 6, p_bits: 1, sp_bits: 0 },
];

static UASTC_TO_BC7_MODES: [u8; 20] = [
    6, 3, 1, 2, 3, 6, 5, 2, // 0..=7 RGB
    0,                      // 8 Void extent
    7, 6, 5, 6, 5, 6,       // 9..=14 RGBA
    6, 7, 5,                // 15..=17 LA
    6,                      // 18 RGB
    0,                      // 19 Reserved
];

static PATTERNS_2_BC7_INDEX_INV: [(u8, bool); uastc::TOTAL_ASTC_BC7_COMMON_PARTITIONS2] = [
    ( 0, false), ( 1, false), ( 2, true ), ( 3, false),
    ( 4, true ), ( 5, false), ( 6, true ), ( 7, true ),
    ( 8, false), ( 9, true ), (10, false), (11, true ),
    (12, true ), (13, true ), (14, false), (15, true ),
    (17, true ), (18, true ), (19, false), (20, false),
    (21, false), (22, true ), (23, true ), (24, false),
    (25, true ), (26, false), (29, true ), (32, true ),
    (33, true ), (52, true ),
];

static PATTERNS_3_BC7_INDEX_PERM: [(u8, u8); uastc::TOTAL_ASTC_BC7_COMMON_PARTITIONS3] = [
    ( 4, 0), ( 8, 5), ( 9, 5), (10, 2),
    (11, 2), (12, 0), (13, 4), (20, 1),
    (35, 1), (36, 5), (57, 0),
];

static PATTERNS_3_BC7_TO_ASTC_PERMUTATIONS: [[u8; 3]; 6] = [
    [0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0], [0, 2, 1], [1, 0, 2],
];

static PATTERNS_2_3_BC7_INDEX_PERM: [(u8, u8); uastc::TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS] = [
    (10, 4), (11, 4), ( 0, 3), ( 2, 4),
    ( 8, 5), (13, 4), ( 1, 2), (33, 2),
    (40, 3), (20, 4), (21, 0), (58, 3),
    ( 3, 0), (32, 2), (59, 1), (34, 3),
    (20, 1), (14, 4), (31, 3),
];

static PATTERNS_2_3_BC7_TO_ASTC_PERMUTATIONS: [[u8; 3]; 6] = [
    [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1],
];

// fn convert_bc7_subset_indices_3_to_2(mut p: u8, k: u8) -> u8 {
//     assert!(k < 6);
//     match k >> 1 {
//         0 => if p <= 1 {
//             p = 0;
//         } else {
//             p = 1;
//         }
//         1 => if p == 0 {
//             p = 0;
//         } else {
//             p = 1;
//         }
//         2 => if (p == 0) || (p == 2) {
//             p = 0;
//         } else {
//             p = 1;
//         }
//         _ => unreachable!()
//     }
//     if k & 1 == 1 {
//         p = 1 - p;
//     }
//     p
// }

static PATTERNS_2_BC7: [[u8; 16]; uastc::TOTAL_ASTC_BC7_COMMON_PARTITIONS2] = [
    [ 0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1 ], [ 0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1 ],
    [ 0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1 ], [ 0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1 ],
    [ 0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1 ], [ 0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1 ],
    [ 0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1 ], [ 0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1 ],
    [ 0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1 ], [ 0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1 ],
    [ 0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1 ], [ 0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1 ],
    [ 0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1 ], [ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1 ],
    [ 0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 ], [ 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1 ],
    [ 0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0 ], [ 0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0 ],
    [ 0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0 ], [ 0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0 ],
    [ 0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0 ], [ 0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0 ],
    [ 0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1 ], [ 0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0 ],
    [ 0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0 ], [ 0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0 ],
    [ 0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0 ], [ 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1 ],
    [ 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 ], [ 0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1 ],
];

static PATTERNS_3_BC7: [[u8; 16]; uastc::TOTAL_ASTC_BC7_COMMON_PARTITIONS3] = [
    [ 0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2 ], [ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2 ],
    [ 0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2 ], [ 0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2 ],
    [ 0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2 ], [ 0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2 ],
    [ 0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2 ], [ 0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2 ],
    [ 0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0 ], [ 0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0 ],
    [ 0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2 ]
];

static PATTERNS_2_3_BC7: [[u8; 16]; uastc::TOTAL_BC7_3_ASTC2_COMMON_PARTITIONS] = [
    [ 0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2 ], [ 0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2 ],
    [ 0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2 ], [ 0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1 ],
    [ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2 ], [ 0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2 ],
    [ 0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1 ], [ 0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1 ],
    [ 0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1 ], [ 0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2 ],
    [ 0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1 ], [ 0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2 ],
    [ 0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1 ], [ 0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2 ],
    [ 0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2 ], [ 0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2 ],
    [ 0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2 ], [ 0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2 ],
    [ 0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1 ]
];

const TOTAL_BC7_PATTERNS: usize = 64;

static PATTERNS_2_BC7_ANCHORS: [[u8; 2]; TOTAL_BC7_PATTERNS] = [
    [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15],
    [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0, 15],
    [0, 15], [0,  2], [0,  8], [0,  2], [0,  2], [0,  8], [0,  8], [0, 15],
    [0,  2], [0,  8], [0,  2], [0,  2], [0,  8], [0,  8], [0,  2], [0,  2],
    [0, 15], [0, 15], [0,  6], [0,  8], [0,  2], [0,  8], [0, 15], [0, 15],
    [0,  2], [0,  8], [0,  2], [0,  2], [0,  2], [0, 15], [0, 15], [0,  6],
    [0,  6], [0,  2], [0,  6], [0,  8], [0, 15], [0, 15], [0,  2], [0,  2],
    [0, 15], [0, 15], [0, 15], [0, 15], [0, 15], [0,  2], [0,  2], [0, 15],
];

static PATTERNS_3_BC7_ANCHORS: [[u8; 3]; TOTAL_BC7_PATTERNS] = [
    [0,  3, 15], [0,  3,  8], [0, 15,  8], [0, 15,  3], [0,  8, 15], [0,  3, 15], [0, 15,  3], [0, 15,  8],
    [0,  8, 15], [0,  8, 15], [0,  6, 15], [0,  6, 15], [0,  6, 15], [0,  5, 15], [0,  3, 15], [0,  3,  8],
    [0,  3, 15], [0,  3,  8], [0,  8, 15], [0, 15,  3], [0,  3, 15], [0,  3,  8], [0,  6, 15], [0, 10,  8],
    [0,  5,  3], [0,  8, 15], [0,  8,  6], [0,  6, 10], [0,  8, 15], [0,  5, 15], [0, 15, 10], [0, 15,  8],
    [0,  8, 15], [0, 15,  3], [0,  3, 15], [0,  5, 10], [0,  6, 10], [0, 10,  8], [0,  8,  9], [0, 15, 10],
    [0, 15,  6], [0,  3, 15], [0, 15,  8], [0,  5, 15], [0, 15,  3], [0, 15,  6], [0, 15,  6], [0, 15,  8],
    [0,  3, 15], [0, 15,  3], [0,  5, 15], [0,  5, 15], [0,  5, 15], [0,  8, 15], [0,  5, 15], [0, 10, 15],
    [0,  5, 15], [0, 10, 15], [0,  8, 15], [0, 13, 15], [0, 15,  3], [0, 12, 15], [0,  3, 15], [0,  3,  8],
];

#[derive(Clone, Copy, Default)]
struct OptimalEndpoint {
    lo: u8,
    hi: u8,
    err: u16,
}

impl OptimalEndpoint {
    const fn const_default() -> Self {
        OptimalEndpoint { err: 0, lo: 0, hi: 0 }
    }
}

static MODE_8_BC7_TABLES: Once = Once::new();

const BC7ENC_MODE_5_OPTIMAL_INDEX: u8 = 1;
const BC7ENC_MODE_6_OPTIMAL_INDEX: u8 = 5;

static mut BC7_MODE_5_OPTIMAL_ENDPOINTS: [OptimalEndpoint; 256] = [OptimalEndpoint::const_default(); 256];
static mut BC7_MODE_6_OPTIMAL_ENDPOINTS: [[OptimalEndpoint; 2]; 256] = [[OptimalEndpoint::const_default(); 2]; 256];

fn get_mode_8_bc7_tables() -> (&'static [OptimalEndpoint; 256], &'static [[OptimalEndpoint; 2]; 256]) {

    MODE_8_BC7_TABLES.call_once(|| {
        calculate_mode_8_bc7_tables();
    });

    fn calculate_mode_8_bc7_tables() {

        let weights2: [u8; 4] = [ 0, 21, 43, 64 ];
        let weights4: [u8; 16] = [ 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 ];

        // TODO: Precompute?
        // BC7 777.1
        for c in 0..256i32 {
            for lp in 0..2 {
                let mut best = OptimalEndpoint::default();
                best.err = u16::MAX;

                for l in 0..128i32 {
                    let low = (l << 1) | lp;

                    for h in 0..128i32 {
                        let high = (h << 1) | lp;

                        let k = (low * (64 - weights4[BC7ENC_MODE_6_OPTIMAL_INDEX as usize] as i32) + high * weights4[BC7ENC_MODE_6_OPTIMAL_INDEX as usize] as i32 + 32) >> 6;

                        let err = ((k - c) * (k - c)) as u16;
                        if err < best.err {
                            best.err = err;
                            best.lo = l as u8;
                            best.hi = h as u8;
                        }
                    } // h
                } // l

                unsafe {
                    BC7_MODE_6_OPTIMAL_ENDPOINTS[c as usize][lp as usize] = best;
                }
            } // lp

        } // c

        // BC7 777
        for c in 0..256 {
            let mut best = OptimalEndpoint::default();
            best.err = u16::MAX;

            for l in 0..128 {
                let low = (l << 1) | (l >> 6);

                for h in 0..128 {
                    let high = (h << 1) | (h >> 6);

                    let k = (
                        low * (64 - weights2[BC7ENC_MODE_5_OPTIMAL_INDEX as usize] as i32)
                        + high * weights2[BC7ENC_MODE_5_OPTIMAL_INDEX as usize] as i32
                        + 32
                    ) >> 6;

                    let err = ((k - c) * (k - c)) as u16;
                    if err < best.err {
                        best.err = err;
                        best.lo = l as u8;
                        best.hi = h as u8;
                    }
                } // h
            } // l

            unsafe {
                BC7_MODE_5_OPTIMAL_ENDPOINTS[c as usize] = best;
            }

        } // c
    }

    unsafe {
        (&BC7_MODE_5_OPTIMAL_ENDPOINTS, &BC7_MODE_6_OPTIMAL_ENDPOINTS)
    }
}
