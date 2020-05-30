#![allow(non_upper_case_globals)]

use crate::bitreader::BitReaderLSB;

use crate::Result;

// Max supported Huffman code size is 16-bits
const MaxSupportedCodeSize: usize = 16;

// The maximum number of symbols  is 2^14
const MaxSymsLog2: u8 = 14;
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

    let total_used_syms = reader.read(MaxSymsLog2 as usize);  // [1, MaxSyms]
    let num_codelength_codes = reader.read(5) as usize; // [1, TotalCodelengthCodes]

    println!("Total used syms: {}, num codelength codes: {}", total_used_syms, num_codelength_codes);

    let code_length_table = {
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

    println!("Symbol code bits: {:?}", code_length_table);

    Err("not implemented".into())
}

#[derive(Clone, Debug)]
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

        for n in 0..total_syms {
            let len = code_sizes[n] as usize;
            if len != 0 {
                tree[n] = next_code[len];
                next_code[len] += 1;
            }
        }

        Ok(Self {
            code_sizes: code_sizes.to_vec(),
            tree,
        })
    }
}
