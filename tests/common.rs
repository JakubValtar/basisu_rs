use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use ktx::header::KtxInfo;

pub const UASTC_CORPUS: &str = "textures\\uastc\\";
pub const ETC1S_CORPUS: &str = "textures\\etc1s\\";

pub const DIR_RGB: &str = "rgb";
pub const DIR_RGBA: &str = "rgba";

pub const EXT_BASIS: &str = ".basis";
pub const EXT_UASTC_RGBA32: &str = "_unpacked_rgba_ASTC_RGBA_0000.png";
pub const EXT_ETC1S_RGB32: &str = "_unpacked_rgb_RGBA32_0_0000.png";
pub const EXT_ETC1S_ALPHA32: &str = "_unpacked_a_RGBA32_0_0000.png";
pub const EXT_ASTC_RGBA: &str = "_transcoded_ASTC_RGBA_0000.ktx";
pub const EXT_BC7_RGBA: &str = "_transcoded_BC7_RGBA_0000.ktx";
pub const EXT_ETC1_RGB: &str = "_transcoded_ETC1_RGB_0000.ktx";
pub const EXT_ETC2_RGBA: &str = "_transcoded_ETC2_RGBA_0000.ktx";

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct TestCase {
    pub basis: PathBuf,
    pub uastc_rgba32: PathBuf,
    pub etc1s_rgb32: PathBuf,
    pub etc1s_alpha32: PathBuf,
    pub astc_rgba: PathBuf,
    pub bc7_rgba: PathBuf,
    pub etc1_rgb: PathBuf,
    pub etc2_rgba: PathBuf,
}

impl TestCase {
    pub fn new<P: AsRef<Path>>(path: P, name: &str) -> Self {
        let base = PathBuf::from(path.as_ref());
        Self {
            basis: base.join(format!("{}{}", name, EXT_BASIS)),
            uastc_rgba32: base.join(format!("{}{}", name, EXT_UASTC_RGBA32)),
            etc1s_rgb32: base.join(format!("{}{}", name, EXT_ETC1S_RGB32)),
            etc1s_alpha32: base.join(format!("{}{}", name, EXT_ETC1S_ALPHA32)),
            astc_rgba: base.join(format!("{}{}", name, EXT_ASTC_RGBA)),
            bc7_rgba: base.join(format!("{}{}", name, EXT_BC7_RGBA)),
            etc1_rgb: base.join(format!("{}{}", name, EXT_ETC1_RGB)),
            etc2_rgba: base.join(format!("{}{}", name, EXT_ETC2_RGBA)),
        }
    }
}

pub fn iterate_textures_etc1s<F>(f: F)
where
    F: FnMut(TestCase),
{
    iterate_textures(ETC1S_CORPUS, f);
}

pub fn iterate_textures_uastc<F>(f: F)
where
    F: FnMut(TestCase),
{
    iterate_textures(UASTC_CORPUS, f);
}

fn iterate_textures<F>(dir: &str, mut f: F)
where
    F: FnMut(TestCase),
{
    let base = PathBuf::from(dir);
    for dir in [DIR_RGB, DIR_RGBA].iter() {
        let base = &base.join(dir);
        let textures = list_textures(&base).unwrap();
        for texture in textures {
            let case = TestCase::new(&base, &texture);
            f(case);
        }
    }
}

pub fn list_textures<P: AsRef<Path>>(dir: P) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }

        let path = entry.path();

        if let Some(ext) = path.extension() {
            if ext == "basis" {
                if let Some(stem) = path.file_stem() {
                    result.push(stem.to_str().unwrap().to_owned());
                }
            }
        }
    }
    result.sort();
    Ok(result)
}

pub fn open_ktx<P: AsRef<Path>>(path: P) -> Result<ktx::Decoder<BufReader<File>>> {
    let file = File::open(path)?;
    let r = BufReader::new(file);
    let decoder = ktx::Decoder::new(r)?;
    Ok(decoder)
}

pub fn open_png<P: AsRef<Path>>(path: P) -> Result<png::Decoder<BufReader<File>>> {
    let file = File::open(path)?;
    let r = BufReader::new(file);
    let decoder = png::Decoder::new(r);
    Ok(decoder)
}

pub fn compare_png<P: AsRef<Path>>(path: P, image: &basisu::Image<u8>) -> Result<()> {
    let decoder = open_png(path)?;

    let (info, mut reader) = decoder.read_info()?;

    assert_eq!(info.width, image.w);
    assert_eq!(info.height, image.h);

    let mut actual_rows = rgba_rows(&image);

    match info.color_type {
        png::ColorType::RGBA => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                assert_slices_eq(expected_row, actual_row);
            }
        }
        png::ColorType::RGB => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(
                        &expected_row[3 * i..3 * i + 3],
                        &actual_row[4 * i..4 * i + 3]
                    );
                    assert_eq!(255, actual_row[4 * i + 3]);
                }
            }
        }
        png::ColorType::GrayscaleAlpha => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(expected_row[2 * i], actual_row[4 * i]);
                    assert_eq!(expected_row[2 * i], actual_row[4 * i + 1]);
                    assert_eq!(expected_row[2 * i], actual_row[4 * i + 2]);
                    assert_eq!(expected_row[2 * i + 1], actual_row[4 * i + 3]);
                }
            }
        }
        png::ColorType::Grayscale => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(expected_row[i], actual_row[4 * i]);
                    assert_eq!(expected_row[i], actual_row[4 * i + 1]);
                    assert_eq!(expected_row[i], actual_row[4 * i + 2]);
                    assert_eq!(255, actual_row[4 * i + 3]);
                }
            }
        }
        _ => unimplemented!(),
    }

    Ok(())
}

pub fn compare_png_rgb<P: AsRef<Path>>(path: P, image: &basisu::Image<u8>) -> Result<()> {
    let decoder = open_png(path)?;

    let (info, mut reader) = decoder.read_info()?;

    assert_eq!(info.width, image.w);
    assert_eq!(info.height, image.h);

    let mut actual_rows = rgba_rows(&image);

    match info.color_type {
        png::ColorType::RGB => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(
                        &expected_row[3 * i..3 * i + 3],
                        &actual_row[4 * i..4 * i + 3]
                    );
                }
            }
        }
        png::ColorType::Grayscale => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(expected_row[i], actual_row[4 * i]);
                    assert_eq!(expected_row[i], actual_row[4 * i + 1]);
                    assert_eq!(expected_row[i], actual_row[4 * i + 2]);
                }
            }
        }
        _ => unimplemented!(),
    }

    Ok(())
}

pub fn compare_png_alpha<P: AsRef<Path>>(path: P, image: &basisu::Image<u8>) -> Result<()> {
    let decoder = open_png(path)?;

    let (info, mut reader) = decoder.read_info()?;

    assert_eq!(info.width, image.w);
    assert_eq!(info.height, image.h);

    let mut actual_rows = rgba_rows(&image);

    match info.color_type {
        png::ColorType::Grayscale => {
            while let (Some(expected_row), Some(actual_row)) =
                (reader.next_row()?, actual_rows.next())
            {
                for i in 0..image.w as usize {
                    assert_eq!(expected_row[i], actual_row[4 * i + 3]);
                }
            }
        }
        _ => unimplemented!(),
    }

    Ok(())
}

pub fn compare_ktx<P: AsRef<Path>>(path: P, image: &basisu::Image<u8>) -> Result<()> {
    let decoder = open_ktx(&path)?;

    assert_eq!(decoder.pixel_width(), image.w);
    assert_eq!(decoder.pixel_height(), image.h);

    let mut textures = decoder.read_textures();

    if let Some(texture) = textures.next() {
        assert_slices_eq(&texture, &image.data);
        assert_eq!(textures.next(), None);
        Ok(())
    } else {
        Err(format!("Found no texture in the KTX file: {:?}", path.as_ref()).into())
    }
}

pub fn rgba_rows<'a>(image: &'a basisu::Image<u8>) -> Box<dyn Iterator<Item = &'a [u8]> + 'a> {
    // TODO: Is texture with y flipped aligned to the top or to the bottom? This code assumes to the top.
    let bytes_per_pixel = 4;
    let res = image
        .data
        .chunks_exact(image.stride as usize)
        .take(image.h as usize)
        .map(move |r| &r[0..(image.w * bytes_per_pixel) as usize]);

    if image.y_flipped {
        Box::new(res.rev())
    } else {
        Box::new(res)
    }
}

pub fn assert_slices_eq<T: PartialEq + std::fmt::Debug>(a: &[T], b: &[T]) {
    let max = a.len().min(b.len());
    if a.len() == b.len() {
        if a == b {
            return;
        }
    } else if a[0..max] == b[0..max] {
        let min = max.saturating_sub(10);
        let a_max = a.len().min(max + 10);
        let b_max = b.len().min(max + 10);
        assert_eq!(&a[min..a_max], &b[min..b_max]);
    }
    let mut mismatch = None;
    for (i, (ta, tb)) in a.iter().zip(b.iter()).enumerate() {
        if ta != tb {
            mismatch = Some(i);
            break;
        }
    }
    if let Some(i) = mismatch {
        let min = i.saturating_sub(5);
        let max = (i + 6).min(a.len());
        assert_eq!(&a[min..max], &b[min..max], "index: {}", i);
    }
}
