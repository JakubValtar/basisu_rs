
pub struct BitReaderLSB<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReaderLSB<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
        }
    }

    pub fn read(&mut self, count: usize) -> u32 {
        assert!(count <= 32);
        let mut byte = self.pos / 8;
        let mut result: u32 = 0;
        let mut read = 0;
        
        {
            let bit = self.pos % 8;
            let byte_val = if byte < self.bytes.len() { self.bytes[byte] } else { 0 };
            result |= (byte_val >> bit) as u32;
            read += 8 - bit;
            byte += 1;
        }

        loop {
            if read >= count {
                self.pos += count;
                if count % 8 != 0 {
                    result &= (1 << count) - 1;
                }
                return result;
            }
            let byte_val = if byte < self.bytes.len() { self.bytes[byte] } else { 0 };
            result |= (byte_val as u32) << read;
            read += 8;
            byte += 1;
        }
    }
}
