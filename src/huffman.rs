#![allow(non_upper_case_globals)]

use std::fmt;

use crate::bitreader::BitReaderLSB;

use crate::Result;

// Max supported Huffman code size is 16-bits
const MaxSupportedCodeSize: usize = 16;

// The maximum number of symbols  is 2^14
const MaxSymsLog2: usize = 14;
const MaxSyms: usize = 1 << MaxSymsLog2;

// Small zero runs may range from 3-10 entries
const SmallZeroRunSizeMin: usize = 3;
const SmallZeroRunSizeMax: usize = 10;
const SmallZeroRunExtraBits: usize = 3;

// Big zero runs may range from 11-138 entries
const BigZeroRunSizeMin: usize = 11;
const BigZeroRunSizeMax: usize = 138;
const BigZeroRunExtraBits: usize = 7;

// Small non-zero runs may range from 3-6 entries
const SmallRepeatSizeMin: usize = 3;
const SmallRepeatSizeMax: usize = 6;
const SmallRepeatExtraBits: usize = 2;

// Big non-zero run may range from 7-134 entries
const BigRepeatSizeMin: usize = 7;
const BigRepeatSizeMax: usize = 134;
const BigRepeatExtraBits: usize = 7;

// There are a maximum of 21 symbols in a compressed Huffman code length table.
const TotalCodelengthCodes: usize = 21;

// Symbols [0,16] indicate code sizes. Other symbols indicate zero runs or repeats:
const SmallZeroRunCode: usize = 17;
const BigZeroRunCode: usize = 18;
const SmallRepeatCode: usize = 19;
const BigRepeatCode: usize = 20;

pub fn read_codelength_table(buf: &[u8]) -> Result<HuffmanDecodingTable> {

    // TODO: sanity & overflow checks

    let mut reader = BitReaderLSB::new(buf);

    let total_used_syms = reader.read(MaxSymsLog2) as usize;  // [1, MaxSyms]

    println!("Total used syms: {}", total_used_syms);

    let code_length_table = {
        let num_codelength_codes = reader.read(5) as usize; // [1, TotalCodelengthCodes]

        let indices = [
            SmallZeroRunCode, BigZeroRunCode,
            SmallRepeatCode, BigRepeatCode,
            0, 8, 7, 9, 6, 0xA, 5, 0xB, 4, 0xC, 3, 0xD, 2, 0xE, 1, 0xF, 0x10
        ];

        let mut codelength_code_sizes = [0u8; TotalCodelengthCodes];
        for i in 0..num_codelength_codes {
            codelength_code_sizes[indices[i]] = reader.read(3) as u8;
        }

        HuffmanDecodingTable::from_sizes(&codelength_code_sizes)?
    };

    let mut symbol_code_sizes: Vec<u8> = Vec::with_capacity(total_used_syms);
    while symbol_code_sizes.len() < total_used_syms {
        let bits = reader.peek(16) as u16;
        let (symbol_code_size, bits_used) = code_length_table.decode_symbol(bits)
            .ok_or_else(|| format!(
                "No matching code found in the decoding table, bits: {:016b}, table: {:?}, ",
                bits, code_length_table
            ))?;
        reader.read(bits_used);
        match symbol_code_size as usize {
            0..=16 => {
                symbol_code_sizes.push(symbol_code_size as u8);
            }
            SmallZeroRunCode => {
                let count = SmallZeroRunSizeMin + reader.read(SmallZeroRunExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(0);
                }
            }
            BigZeroRunCode => {
                let count = BigZeroRunSizeMin + reader.read(BigZeroRunExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(0);
                }
            }
            SmallRepeatCode => {
                let prev_sym_code_size = symbol_code_sizes.last().copied()
                    .ok_or_else(|| "Encountered SmallRepeatCode as the first code")?;
                if prev_sym_code_size == 0 {
                    return Err("Encountered SmallRepeatCode, but the previous symbol's code length was 0".into());
                }
                let count = SmallRepeatSizeMin + reader.read(SmallRepeatExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(prev_sym_code_size);
                }
            }
            BigRepeatCode => {
                let prev_sym_code_size = symbol_code_sizes.last().copied()
                    .ok_or_else(|| "Encountered BigRepeatCode as the first code")?;
                if prev_sym_code_size == 0 {
                    return Err("Encountered BigRepeatCode, but the previous symbol's code length was 0".into());
                }
                let count = BigRepeatSizeMin + reader.read(BigRepeatExtraBits) as usize;
                for _ in 0..count {
                    symbol_code_sizes.push(prev_sym_code_size);
                }
            }
            _ => unreachable!()
        }
    }

    let symbol_table = HuffmanDecodingTable::from_sizes(&symbol_code_sizes);

    println!("symbol_table: {:#?}", symbol_table);

    return symbol_table;
}

#[derive(Clone)]
pub struct HuffmanDecodingTable {
    code_sizes: Vec<u8>,
    tree: Vec<u16>,
}

impl HuffmanDecodingTable {
    pub fn from_sizes(code_sizes: &[u8]) -> Result<Self> {
        // TODO: sanity checks

        let total_syms = code_sizes.len();

        let mut syms_using_codesize = [0u16; MaxSupportedCodeSize+1];
        for &count in code_sizes {
            syms_using_codesize[count as usize] += 1;
        }

        let mut total = 0;
        let mut next_code = [0u16; MaxSupportedCodeSize+1];
        syms_using_codesize[0] = 0;
        for bits in 1..MaxSupportedCodeSize + 1 {
            total = (total + syms_using_codesize[bits-1]) << 1;
            next_code[bits] = total;
        }

        let mut tree = vec![0u16; total_syms];

        let code_width = std::mem::size_of_val(&next_code[0]) * 8;

        for n in 0..total_syms {
            let len = code_sizes[n] as usize;
            if len != 0 {
                tree[n] = next_code[len].reverse_bits() >> (code_width - len);
                next_code[len] += 1;
            }
        }

        Ok(Self {
            code_sizes: code_sizes.to_vec(),
            tree,
        })
    }

    pub fn decode_symbol(&self, bits: u16) -> Option<(u16, usize)> {
        let sym_count = self.code_sizes.len();
        for i in 0..sym_count {
            let code_size = self.code_sizes[i] as usize;
            if code_size > 0 {
                let code = bits & ((1 << code_size) - 1);
                if code == self.tree[i] {
                    return Some((i as u16, code_size));
                }
            }
        }
        None
    }
}

impl fmt::Debug for HuffmanDecodingTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        let max_size = self.code_sizes.iter().max().copied().unwrap_or(16) as usize + 1;
        for (sym, (&size, &code)) in self.code_sizes.iter().zip(self.tree.iter()).enumerate() {
            if size > 0 {
                debug.entry(&format_args!(
                    "{:4}:{:pad_width$}{:0code_width$b}",
                    sym, " ", code,
                    pad_width = max_size - size as usize,
                    code_width = size as usize
                ));
            }
        }
        debug.finish()?;
        Ok(())
    }
}
