use crate::mask;

pub struct BitWriterLSB<'a> {
    bytes: &'a mut [u8],
    bit_pos: usize,
}

impl<'a> BitWriterLSB<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            bit_pos: 0,
        }
    }

    pub fn write_bool(&mut self, v: bool) {
        self.write_u32(1, v as u32)
    }

    pub fn write_u8(&mut self, count: usize, v: u8) {
        assert!(count <= 8);
        self.write_u32(count, v as u32)
    }

    pub fn write_u16(&mut self, count: usize, v: u16) {
        assert!(count <= 16);
        self.write_u32(count, v as u32)
    }

    pub fn write_u32(&mut self, count: usize, v: u32) {
        assert!(count <= 32);
        let v = v & mask!(count as u32);

        let mut byte = self.bit_pos / 8;
        let mut written = 0;

        let mut trash = 0;

        {
            let bit = self.bit_pos % 8;
            let byte_val = self.bytes.get_mut(byte).unwrap_or(&mut trash);
            *byte_val |= (v << bit) as u8;
            written += 8 - bit;
            byte += 1;
        }

        self.bit_pos += count;

        loop {
            if written >= count {
                return;
            }
            let byte_val = self.bytes.get_mut(byte).unwrap_or(&mut trash);
            *byte_val |= (v >> written) as u8;
            written += 8;
            byte += 1;
        }
    }

    // TODO: check on close/drop that the buffer did not overflow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwriter() {
        let pattern = 0x5555_5555_5555_5555u64;

        // For each of these 16 pattens
        for i in 0..16 {
            let segment = mask!(16u64);
            let xor_mask =
                (segment * ((i >> 3) & 0x1)) << 48 |
                (segment * ((i >> 2) & 0x1)) << 32 |
                (segment * ((i >> 1) & 0x1)) << 16 |
                (segment * (i & 0x1));
            let data = pattern ^ xor_mask;
            let mut bytes;

            // Check that writing two numbers with all combinations of bits
            // writes the right bits into the output byte buffer
            for len in 0..32 {
                for offset in 0..32 {
                    bytes = [0; 8];
                    let mut writer = BitWriterLSB::new(&mut bytes);

                    let offset_val = (data & mask!(offset as u64)) as u32;
                    writer.write_u32(offset, offset_val);

                    let value = ((data >> offset) & mask!(len as u64)) as u32;
                    writer.write_u32(len, value);

                    let expected = data & mask!((offset + len) as u64);
                    let actual = u64::from_le_bytes(bytes);

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
