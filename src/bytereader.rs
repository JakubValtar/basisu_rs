use byteorder::{
    ByteOrder,
    LE,
};

pub struct ByteReaderLE<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReaderLE<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn read_u8(&mut self) -> u8 {
        let res = self.bytes[self.pos];
        self.pos += 1;
        res
    }

    pub fn read_u16(&mut self) -> u16 {
        let res = LE::read_u16(&self.bytes[self.pos..]);
        self.pos += 2;
        res
    }

    pub fn read_u24(&mut self) -> u32 {
        let res = LE::read_u24(&self.bytes[self.pos..]);
        self.pos += 3;
        res
    }

    pub fn read_u32(&mut self) -> u32 {
        let res = LE::read_u32(&self.bytes[self.pos..]);
        self.pos += 4;
        res
    }
}
