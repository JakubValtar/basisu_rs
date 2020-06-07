#![warn(clippy::all)]

use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use std::path::Path;
use std::fs::File;
use std::io::BufWriter;

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

    let mut encoder = Encoder::new(w, image.w, image.h); // Width is 2 pixels and height is 1.
    encoder.set_color(ColorType::RGBA);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    writer.write_image_data(&image.data)?; // Save

    Ok(())
}
