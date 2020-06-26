#![warn(clippy::all)]

use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use std::path::Path;
use std::fs::File;
use std::io::{ BufWriter, Write };

use png::{ Encoder, ColorType, BitDepth };

#[test]
fn read_file() {

    let texture_dir: &Path = Path::new("textures");
    let out_dir: &Path = Path::new("out");

    std::fs::create_dir_all(out_dir).unwrap();

    let filenames = [
        "alpha3.basis",
        //"kodim03_uastc.basis",
        "kodim03.basis",
        "kodim20_1024x1024.basis",
        "kodim20.basis",
        //"kodim26_uastc_1024.basis",
    ];

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();

    for filename in &filenames {
        println!("{}", filename);
        let images = basisu::read_file(texture_dir.join(filename)).unwrap();
        for (i, image) in images.iter().enumerate() {
            let path = out_dir.join(format!("{}-slice{:02}-{}.png", filename, i, timestamp));
            save(path, &image).unwrap();
        }
    }
}

fn save<P: AsRef<Path>>(path: P, image: &basisu::Image<u8>) -> std::io::Result<()> {
    let file = File::create(path)?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = Encoder::new(w, image.w, image.h);
    encoder.set_color(ColorType::RGBA);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut w = writer.stream_writer();

    for row in rgba_rows(image) {
        w.write_all(row)?;
    }

    w.finish()?;

    Ok(())
}

pub fn rgba_rows<'a>(image: &'a basisu::Image<u8>) -> Box<dyn Iterator<Item=&'a [u8]> + 'a> {
    // TODO: Is texture with y flipped aligned to the top or to the bottom? This code assumes to the top.
    let bytes_per_pixel = 4;
    let res = image.data
        .chunks_exact(image.stride as usize)
        .take(image.h as usize)
        .map(move |r| &r[0..(image.w * bytes_per_pixel) as usize]);

    if image.y_flipped {
        Box::new(res.rev())
    } else {
        Box::new(res)
    }
}
