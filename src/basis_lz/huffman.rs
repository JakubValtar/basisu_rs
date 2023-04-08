#![allow(non_upper_case_globals)]

use alloc::{format, vec, vec::Vec};

use crate::{bitreader::BitReaderLsb, Result};

// Max supported Huffman code size is 16-bits
const MaxSupportedCodeSize: usize = 16;

// The maximum number of symbols  is 2^14
const MaxSymsLog2: usize = 14;
const _MaxSyms: usize = 1 << MaxSymsLog2;

// Small zero runs may range from 3-10 entries
const SmallZeroRunSizeMin: usize = 3;
const _SmallZeroRunSizeMax: usize = 10;
const SmallZeroRunExtraBits: usize = 3;

// Big zero runs may range from 11-138 entries
const BigZeroRunSizeMin: usize = 11;
const _BigZeroRunSizeMax: usize = 138;
const BigZeroRunExtraBits: usize = 7;

// Small non-zero runs may range from 3-6 entries
const SmallRepeatSizeMin: usize = 3;
const _SmallRepeatSizeMax: usize = 6;
const SmallRepeatExtraBits: usize = 2;

// Big non-zero run may range from 7-134 entries
const BigRepeatSizeMin: usize = 7;
const _BigRepeatSizeMax: usize = 134;
const BigRepeatExtraBits: usize = 7;

// There are a maximum of 21 symbols in a compressed Huffman code length table.
const TotalCodelengthCodes: usize = 21;

// Symbols [0,16] indicate code sizes. Other symbols indicate zero runs or repeats:
const SmallZeroRunCode: usize = 17;
const BigZeroRunCode: usize = 18;
const SmallRepeatCode: usize = 19;
const BigRepeatCode: usize = 20;

pub fn read_huffman_table(reader: &mut BitReaderLsb) -> Result<HuffmanDecodingTable> {
    // TODO: sanity & overflow checks

    let total_used_syms = reader.read_u32(MaxSymsLog2) as usize; // [1, MaxSyms]

    let codelength_table = {
        let num_codelength_codes = reader.read_u32(5) as usize; // [1, TotalCodelengthCodes]

        #[rustfmt::skip]
        let indices = [
            SmallZeroRunCode, BigZeroRunCode,
            SmallRepeatCode, BigRepeatCode,
            0, 8, 7, 9, 6, 0xA, 5, 0xB, 4, 0xC, 3, 0xD, 2, 0xE, 1, 0xF, 0x10
        ];

        let mut codelength_code_sizes = [0u8; TotalCodelengthCodes];
        for i in 0..num_codelength_codes {
            codelength_code_sizes[indices[i]] = reader.read_u32(3) as u8;
        }

        HuffmanDecodingTable::from_sizes(&codelength_code_sizes)?
    };

    let mut symbol_code_sizes: Vec<u8> = Vec::with_capacity(total_used_syms);
    while symbol_code_sizes.len() < total_used_syms {
        let symbol_code_size = codelength_table.decode_symbol(reader)?;
        match symbol_code_size as usize {
            0..=16 => {
                symbol_code_sizes.push(symbol_code_size as u8);
            }
            SmallZeroRunCode => {
                let count = SmallZeroRunSizeMin + reader.read_u32(SmallZeroRunExtraBits) as usize;
                symbol_code_sizes.extend(core::iter::repeat(0).take(count));
            }
            BigZeroRunCode => {
                let count = BigZeroRunSizeMin + reader.read_u32(BigZeroRunExtraBits) as usize;
                symbol_code_sizes.extend(core::iter::repeat(0).take(count));
            }
            SmallRepeatCode => {
                let prev_sym_code_size = symbol_code_sizes
                    .last()
                    .copied()
                    .ok_or("Encountered SmallRepeatCode as the first code")?;
                if prev_sym_code_size == 0 {
                    return Err(
                        "Encountered SmallRepeatCode, but the previous symbol's code length was 0"
                            .into(),
                    );
                }
                let count = SmallRepeatSizeMin + reader.read_u32(SmallRepeatExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(prev_sym_code_size);
                }
            }
            BigRepeatCode => {
                let prev_sym_code_size = symbol_code_sizes
                    .last()
                    .copied()
                    .ok_or("Encountered BigRepeatCode as the first code")?;
                if prev_sym_code_size == 0 {
                    return Err(
                        "Encountered BigRepeatCode, but the previous symbol's code length was 0"
                            .into(),
                    );
                }
                let count = BigRepeatSizeMin + reader.read_u32(BigRepeatExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(prev_sym_code_size);
                }
            }
            _ => unreachable!(),
        }
    }

    HuffmanDecodingTable::from_sizes(&symbol_code_sizes)
}

#[derive(Clone, Copy, Default)]
struct HuffmanTableEntry {
    symbol: u16,
    code_size: u8,
}

#[derive(Clone)]
pub struct HuffmanDecodingTable {
    lookup: Vec<HuffmanTableEntry>,
    max_code_size: usize,
}

impl HuffmanDecodingTable {
    pub fn from_sizes(code_sizes: &[u8]) -> Result<Self> {
        // TODO: sanity checks

        let mut syms_using_codesize = [0u32; MaxSupportedCodeSize + 1];
        let mut max_code_size = 0;
        for &count in code_sizes {
            syms_using_codesize[count as usize] += 1;
            max_code_size = max_code_size.max(count as usize);
        }

        let mut total = 0;
        let mut next_code = [0u32; MaxSupportedCodeSize + 1];
        syms_using_codesize[0] = 0;
        for bits in 1..MaxSupportedCodeSize + 1 {
            total = (total + syms_using_codesize[bits - 1]) << 1;
            next_code[bits] = total;
        }

        let mut lookup = vec![HuffmanTableEntry::default(); 1 << max_code_size];

        let code_width = core::mem::size_of_val(&next_code[0]) * 8;

        for (symbol, code_size) in code_sizes
            .iter()
            .enumerate()
            .map(|(sym, &size)| (sym as u16, size))
        {
            if code_size != 0 {
                let entry = HuffmanTableEntry { symbol, code_size };
                let size = code_size as usize;
                let code = (next_code[size].reverse_bits() >> (code_width - size)) as u16;

                // Generate all lookup entries ending with this code
                let variant_count: u16 = 1 << (max_code_size - size);
                for fill in 0..variant_count {
                    let id = (fill.wrapping_shl(size as u32) | code) as usize;
                    lookup[id] = entry;
                }

                next_code[size] += 1;
            }
        }

        if next_code.iter().any(|&c| c > u16::MAX as u32 + 1) {
            return Err("Code lengths are invalid, codes don't fit into 16 bits".into());
        }

        Ok(Self {
            lookup,
            max_code_size,
        })
    }

    pub fn decode_symbol(&self, reader: &mut BitReaderLsb) -> Result<u16> {
        let bits = reader.peek(self.max_code_size) as usize;
        let entry = self.lookup[bits];
        if entry.code_size > 0 {
            reader.remove(entry.code_size as usize);
            Ok(entry.symbol)
        } else {
            Err(format!(
                "No matching code found in the decoding table, bits: {:016b}",
                bits
            ))
        }
    }
}
