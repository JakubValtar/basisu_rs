mod common;
use common::*;

use std::fmt::{self, Write};
use std::fs;
use std::path::Path;

const TARGET_BLOCKS_PER_MODE: usize = 32;
const MIN_RGB_DIFF: i32 = 16;

#[derive(Copy, Clone, Default)]
struct TestBlock {
    uastc: [u8; 16],
    astc: [u8; 16],
    bc7: [u8; 16],
    etc1: [u8; 8],
    etc2: [u8; 16],
    rgba: [u32; 16],
}

#[test]
#[ignore]
fn uastc_block_export() {
    let mut collected_blocks = vec![vec![]; 19];

    iterate_textures_uastc(|case| {
        collect_blocks(&case, &mut collected_blocks).unwrap();
    });

    {
        let mut output = String::new();
        writeln!(
            output,
            "static TEST_DATA_UASTC_RGBA: [[([u8; 16], [u32; 16]); {}]; 19] = [",
            TARGET_BLOCKS_PER_MODE
        )
        .unwrap();
        for (mode, mode_blocks) in collected_blocks.iter().enumerate() {
            writeln!(output, "    [   // {}", mode).unwrap();
            for block in mode_blocks {
                writeln!(
                    output,
                    "        ({}, {}),",
                    U8ArrayHexPrint(&block.uastc),
                    U32ArrayHexPrint(&block.rgba)
                )
                .unwrap();
            }
            writeln!(output, "    ],").unwrap();
        }
        writeln!(output, "];").unwrap();

        let target = Path::new("target");
        fs::create_dir_all(target).unwrap();
        fs::write(target.join("collected_uastc_rgba.txt"), output).unwrap();
    }

    {
        let mut output = String::new();
        writeln!(
            output,
            "static TEST_DATA_UASTC_ASTC: [[([u8; 16], [u8; 16]); {}]; 19] = [",
            TARGET_BLOCKS_PER_MODE
        )
        .unwrap();
        for (mode, mode_blocks) in collected_blocks.iter().enumerate() {
            writeln!(output, "    [   // {}", mode).unwrap();
            for block in mode_blocks {
                writeln!(
                    output,
                    "        ({}, {}),",
                    U8ArrayHexPrint(&block.uastc),
                    U8ArrayHexPrint(&block.astc)
                )
                .unwrap();
            }
            writeln!(output, "    ],").unwrap();
        }
        writeln!(output, "];").unwrap();

        let target = Path::new("target");
        fs::create_dir_all(target).unwrap();
        fs::write(target.join("collected_uastc_astc.txt"), output).unwrap();
    }

    {
        let mut output = String::new();
        writeln!(
            output,
            "static TEST_DATA_UASTC_BC7: [[([u8; 16], [u8; 16]); {}]; 19] = [",
            TARGET_BLOCKS_PER_MODE
        )
        .unwrap();
        for (mode, mode_blocks) in collected_blocks.iter().enumerate() {
            writeln!(output, "    [   // {}", mode).unwrap();
            for block in mode_blocks {
                writeln!(
                    output,
                    "        ({}, {}),",
                    U8ArrayHexPrint(&block.uastc),
                    U8ArrayHexPrint(&block.bc7)
                )
                .unwrap();
            }
            writeln!(output, "    ],").unwrap();
        }
        writeln!(output, "];").unwrap();

        let target = Path::new("target");
        fs::create_dir_all(target).unwrap();
        fs::write(target.join("collected_uastc_bc7.txt"), output).unwrap();
    }

    {
        let mut output = String::new();
        writeln!(
            output,
            "static TEST_DATA_UASTC_ETC1: [[([u8; 16], [u8; 8]); {}]; 19] = [",
            TARGET_BLOCKS_PER_MODE
        )
        .unwrap();
        for (mode, mode_blocks) in collected_blocks.iter().enumerate() {
            writeln!(output, "    [   // {}", mode).unwrap();
            for block in mode_blocks {
                writeln!(
                    output,
                    "        ({}, {}),",
                    U8ArrayHexPrint(&block.uastc),
                    U8ArrayHexPrint(&block.etc1)
                )
                .unwrap();
            }
            writeln!(output, "    ],").unwrap();
        }
        writeln!(output, "];").unwrap();

        let target = Path::new("target");
        fs::create_dir_all(target).unwrap();
        fs::write(target.join("collected_uastc_etc1.txt"), output).unwrap();
    }

    {
        let mut output = String::new();
        writeln!(
            output,
            "static TEST_DATA_UASTC_ETC2: [[([u8; 16], [u8; 16]); {}]; 19] = [",
            TARGET_BLOCKS_PER_MODE
        )
        .unwrap();
        for (mode, mode_blocks) in collected_blocks.iter().enumerate() {
            writeln!(output, "    [   // {}", mode).unwrap();
            for block in mode_blocks {
                writeln!(
                    output,
                    "        ({}, {}),",
                    U8ArrayHexPrint(&block.uastc),
                    U8ArrayHexPrint(&block.etc2)
                )
                .unwrap();
            }
            writeln!(output, "    ],").unwrap();
        }
        writeln!(output, "];").unwrap();

        let target = Path::new("target");
        fs::create_dir_all(target).unwrap();
        fs::write(target.join("collected_uastc_etc2.txt"), output).unwrap();
    }
}

fn collect_blocks(case: &TestCase, collected_blocks: &mut [Vec<TestBlock>]) -> Result<()> {
    let uastc_data = basisu::read_to_uastc(&case.read_basis()?)?.remove(0);
    let astc_data = open_ktx(&case.astc_rgba)?.read_textures().next().unwrap();
    let bc7_data = open_ktx(&case.bc7_rgba)?.read_textures().next().unwrap();
    let etc1_data = open_ktx(&case.etc1_rgb)?.read_textures().next().unwrap();
    let etc2_data = open_ktx(&case.etc2_rgba)?.read_textures().next().unwrap();
    let rgba_data = {
        let mut reader = open_png(&case.uastc_rgba32)?.read_info()?;
        let info = reader.info();
        let stride = ((info.width + 3) / 4 * 4) * 4;
        let height = (info.height + 3) / 4 * 4;
        let len = stride * height;
        let mut image = basisu::Image {
            w: info.width,
            h: info.height,
            stride,
            data: vec![0u8; len as usize],
        };

        let mut rgba_rows = image.data.chunks_exact_mut(stride as usize);

        match info.color_type {
            png::ColorType::Rgba => {
                while let (Some(src), Some(dst)) = (reader.next_row()?, rgba_rows.next()) {
                    let src = src.data();
                    dst[0..src.len()].copy_from_slice(src);
                }
            }
            png::ColorType::Rgb => {
                while let (Some(src), Some(dst)) = (reader.next_row()?, rgba_rows.next()) {
                    let src = src.data();
                    for i in 0..image.w as usize {
                        dst[4 * i..4 * i + 3].copy_from_slice(&src[3 * i..3 * i + 3]);
                        dst[4 * i + 3] = 255;
                    }
                }
            }
            png::ColorType::GrayscaleAlpha => {
                while let (Some(src), Some(dst)) = (reader.next_row()?, rgba_rows.next()) {
                    let src = src.data();
                    for i in 0..image.w as usize {
                        dst[4 * i] = src[2 * i];
                        dst[4 * i + 1] = src[2 * i];
                        dst[4 * i + 2] = src[2 * i];
                        dst[4 * i + 3] = src[2 * i + 1];
                    }
                }
            }
            png::ColorType::Grayscale => {
                while let (Some(src), Some(dst)) = (reader.next_row()?, rgba_rows.next()) {
                    let src = src.data();
                    for i in 0..image.w as usize {
                        dst[4 * i] = src[i];
                        dst[4 * i + 1] = src[i];
                        dst[4 * i + 2] = src[i];
                        dst[4 * i + 3] = 255;
                    }
                }
            }
            _ => unimplemented!(),
        }

        image
    };

    let block_count = uastc_data.data.len() / 16;
    let block_stride = uastc_data.stride as usize;
    let block_count_x = block_stride / 16;
    let block_count_y = block_count / block_count_x;

    let etc1_block_stride = block_count_x * 8;

    for y in 0..block_count_y {
        for x in 0..block_count_x {
            let block_offset = y * block_stride + x * 16;
            let mut rgba = [0u8; 64];
            let color_offset = 4 * y * rgba_data.stride as usize + 4 * 4 * x;
            for (cy, chunk) in rgba.chunks_exact_mut(16).enumerate() {
                let off = color_offset + rgba_data.stride as usize * cy;
                chunk.copy_from_slice(&rgba_data.data[off..off + 16]);
            }

            let uastc = &uastc_data.data[block_offset..block_offset + 16];
            let astc = &astc_data[block_offset..block_offset + 16];
            let bc7 = &bc7_data[block_offset..block_offset + 16];
            let etc2 = &etc2_data[block_offset..block_offset + 16];

            let etc1_block_offset = y * etc1_block_stride + x * 8;
            let etc1 = &etc1_data[etc1_block_offset..etc1_block_offset + 8];

            let mode = UASTC_MODE_LUT[(uastc[0] & 0b0111_1111) as usize] as usize;

            let mode_blocks = &mut collected_blocks[mode];
            if mode_blocks.len() < TARGET_BLOCKS_PER_MODE {
                let colors_differ = rgba.chunks_exact(4).any(|c| {
                    (c[0] as i32 - c[1] as i32).abs() >= MIN_RGB_DIFF
                        && (c[1] as i32 - c[2] as i32).abs() >= MIN_RGB_DIFF
                        && (c[2] as i32 - c[0] as i32).abs() >= MIN_RGB_DIFF
                });
                let not_opaque_or_transparent =
                    rgba.chunks_exact(4).any(|c| c[3] >= 5 && c[3] <= 250);
                let not_black_or_white = rgba.chunks_exact(4).any(|c| {
                    c[0] >= 5 && c[0] <= 250 && c[1] >= 5 && c[1] <= 250 && c[2] >= 5 && c[2] <= 250
                });
                let grayscale = (15..=17).contains(&mode);
                if (colors_differ && (mode != 8 || not_opaque_or_transparent))
                    || (grayscale && not_opaque_or_transparent && not_black_or_white)
                {
                    let mut test_block = TestBlock::default();
                    test_block.uastc.copy_from_slice(uastc);
                    test_block.astc.copy_from_slice(astc);
                    test_block.bc7.copy_from_slice(bc7);
                    test_block.etc1.copy_from_slice(etc1);
                    test_block.etc2.copy_from_slice(etc2);
                    test_block
                        .rgba
                        .iter_mut()
                        .zip(
                            rgba.chunks_exact(4)
                                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                        )
                        .for_each(|(a, b)| *a = b);
                    mode_blocks.push(test_block);
                }
            }
        }
    }

    Ok(())
}

#[rustfmt::skip]
static UASTC_MODE_LUT: [u8; 128] = [
    11,  0, 10, 3, 11, 15, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11, 16, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 17, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11,  1, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 19, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11, 16, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
    11,  0, 10, 3, 11, 17, 12,  7,
    11, 18, 10, 5, 11, 14, 12,  9,
    11,  0, 10, 4, 11,  1, 12,  8,
    11, 18, 10, 6, 11,  2, 12, 13,
];

struct U8ArrayHexPrint<'a>(&'a [u8]);

impl<'a> fmt::Display for U8ArrayHexPrint<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        write!(f, "0x{:02X}", self.0[0])?;
        for b in self.0.iter().skip(1) {
            write!(f, ", 0x{:02X}", b)?;
        }
        write!(f, "]")
    }
}

struct U32ArrayHexPrint<'a>(&'a [u32]);

impl<'a> fmt::Display for U32ArrayHexPrint<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        write!(f, "0x{:08X}", self.0[0])?;
        for b in self.0.iter().skip(1) {
            write!(f, ", 0x{:08X}", b)?;
        }
        write!(f, "]")
    }
}
