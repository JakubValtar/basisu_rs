#![allow(dead_code)]

use core::fmt;

mod uastc_astc;
mod uastc_bc7;
mod uastc_etc1;
mod uastc_etc2;
mod uastc_rgba;

pub use uastc_astc::TEST_DATA_UASTC_ASTC;
pub use uastc_bc7::TEST_DATA_UASTC_BC7;
pub use uastc_etc1::TEST_DATA_UASTC_ETC1;
pub use uastc_etc2::TEST_DATA_UASTC_ETC2;
pub use uastc_rgba::TEST_DATA_UASTC_RGBA;

#[repr(transparent)]
pub struct LsbDisplay([u8]);

pub fn lsb_display(data: &[u8]) -> &LsbDisplay {
    unsafe { &*(data as *const [u8] as *const LsbDisplay) }
}

impl fmt::Display for LsbDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let mut iter = self.0.iter().rev();
        write!(f, "{:08b}", iter.next().unwrap())?;
        for b in iter {
            write!(f, " {:08b}", b)?;
        }
        write!(f, " <=]")
    }
}

pub fn rgba_display(data: &[u32]) -> &RgbaDisplay {
    unsafe { &*(data as *const [u32] as *const RgbaDisplay) }
}

#[repr(transparent)]
pub struct RgbaDisplay([u32]);

impl fmt::Display for RgbaDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        write!(f, "{:08x?}", self.0[0])?;
        for b in self.0.iter().skip(1) {
            write!(f, " {:08x?}", b)?;
        }
        write!(f, "]")
    }
}
