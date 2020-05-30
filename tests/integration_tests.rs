#![warn(clippy::all)]

const TEXTURES: [&str; 6] = [
    "textures/alpha3.basis",
    "textures/kodim03_uastc.basis",
    "textures/kodim03.basis",
    "textures/kodim20_1024x1024.basis",
    "textures/kodim20.basis",
    "textures/kodim26_uastc_1024.basis",
];

#[test]
fn read_file() {
    for texture in &TEXTURES {
        println!("{}", texture);
        basisu::read_file(texture).unwrap();
    }
}
