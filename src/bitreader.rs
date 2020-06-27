use crate::mask;

pub struct BitReaderLSB<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReaderLSB<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            bit_pos: 0,
        }
    }

    pub fn read(&mut self, count: usize) -> u32 {
        let res = self.peek(count);
        self.remove(count);
        res
    }

    pub fn remove(&mut self, count: usize) {
        self.bit_pos += count;
    }

    pub fn peek(&self, count: usize) -> u32 {
        assert!(count <= 32);
        let mut byte = self.bit_pos / 8;
        let mut result: u32 = 0;
        let mut read = 0;

        {
            let bit = self.bit_pos % 8;
            let byte_val = self.bytes.get(byte).copied().unwrap_or(0);
            result |= (byte_val >> bit) as u32;
            read += 8 - bit;
            byte += 1;
        }

        loop {
            if read >= count {
                return result & mask!(count as u32);
            }
            let byte_val = self.bytes.get(byte).copied().unwrap_or(0);
            result |= (byte_val as u32) << read;
            read += 8;
            byte += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitreader() {
        let pattern = 0x5555_5555_5555_5555u64;
        for i in 0..16 {
            let xor_mask =
                ((mask!(16u16) * ((i >> 3) & 0x1)) as u64) << 3*16 |
                ((mask!(16u16) * ((i >> 2) & 0x1)) as u64) << 2*16 |
                ((mask!(16u16) * ((i >> 1) & 0x1)) as u64) << 1*16 |
                ((mask!(16u16) * ((i >> 0) & 0x1)) as u64) << 0*16;
            let data = pattern ^ xor_mask;
            let bytes = data.to_le_bytes();
            for len in 0..32 {
                for offset in 0..32 {
                    let mut reader = BitReaderLSB::new(&bytes);
                    let actual = reader.read(offset);
                    let expected = (data & mask!(offset as u64)) as u32;
                    assert_eq!(
                        actual, expected,
                        "offset value mismatch, left: {:b}, right: {:b}, off: {}, len: {}, data: {:b}",
                        actual, expected, offset, len, data
                    );

                    let actual = reader.read(len);
                    let expected = ((data >> offset) & mask!(len as u64)) as u32;
                    assert_eq!(
                        actual, expected,
                        "value mismatch, left: {:b}, right: {:b}, off: {}, len: {}, data: {:b}",
                        actual, expected, offset, len, data
                    );
                }
            }
        }
    }
}
