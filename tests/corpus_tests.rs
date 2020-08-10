mod common;
use common::*;

#[test]
fn test_uastc_to_rgba() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_rgba(&case.basis).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_png(&case.uastc_rgba32, &decoded[0]).unwrap();
    });
}

#[test]
fn test_uastc_to_astc() {
    iterate_textures_uastc(|case| {
        let decoded = basisu::read_to_astc(&case.basis).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_ktx(&case.astc_rgba, &decoded[0]).unwrap();
    });
}

#[test]
fn test_etc1s_to_rgba() {
    iterate_textures_etc1s(|case| {
        let decoded = basisu::read_to_rgba(&case.basis).unwrap();
        assert_eq!(decoded.len(), 1);
        compare_png_rgb(&case.etc1s_rgb32, &decoded[0]).unwrap();
        compare_png_alpha(&case.etc1s_alpha32, &decoded[0]).unwrap();
    });
}

#[test]
fn test_etc1s_to_etc1() {
    iterate_textures_etc1s(|case| {
        let decoded = basisu::read_to_etc1(&case.basis).unwrap();
        assert!(decoded.len() <= 2);
        compare_ktx(&case.etc1_rgb, &decoded[0]).unwrap();
    });
}
