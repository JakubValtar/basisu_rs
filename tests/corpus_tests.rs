mod common;
use common::*;

#[test]
fn test_uastc_to_rgba() {
    iterate_textures_uastc(|case| {
        let (header, decoded) = basisu::read_to_rgba(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_png(&case.uastc_rgba32, &decoded[0], header.has_y_flipped()).unwrap();
    });
}

#[test]
fn test_uastc_to_astc() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_astc(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_ktx(&case.astc_rgba, &decoded[0]).unwrap();
    });
}

#[test]
fn test_uastc_to_bc7() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_bc7(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_ktx(&case.bc7_rgba, &decoded[0]).unwrap();
    });
}

#[test]
fn test_uastc_to_etc1() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_etc1(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_ktx(&case.etc1_rgb, &decoded[0]).unwrap();
    });
}

#[test]
fn test_uastc_to_etc2() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_etc2(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_ktx(&case.etc2_rgba, &decoded[0]).unwrap();
    });
}

#[test]
fn test_etc1s_to_rgba() {
    iterate_textures_etc1s(|case| {
        let (header, decoded) = basisu::read_to_rgba(&case.read_basis().unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_png_rgb(&case.etc1s_rgb32, &decoded[0], header.has_y_flipped()).unwrap();
        compare_png_alpha(&case.etc1s_alpha32, &decoded[0], header.has_y_flipped()).unwrap();
    });
}

#[test]
fn test_etc1s_to_etc1() {
    iterate_textures_etc1s(|case| {
        let decoded = basisu::read_to_etc1(&case.read_basis().unwrap()).unwrap();
        assert!(decoded.len() <= 2);
        compare_ktx(&case.etc1_rgb, &decoded[0]).unwrap();
    });
}
