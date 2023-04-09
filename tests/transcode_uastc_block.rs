use block_test_cases::{
    rgba_display, TEST_DATA_UASTC_ASTC, TEST_DATA_UASTC_BC7, TEST_DATA_UASTC_ETC1,
    TEST_DATA_UASTC_ETC2, TEST_DATA_UASTC_RGBA,
};

use crate::block_test_cases::lsb_display;

mod block_test_cases;

fn test_blocks<const N: usize, O, F, D>(
    test_data: [&[([u8; 16], [O; N])]; 19],
    f: F,
    d: fn(&[O]) -> &D,
) where
    F: Fn([u8; 16]) -> Result<[O; N], String>,
    O: core::fmt::Debug + Eq,
    D: core::fmt::Display + ?Sized,
{
    for (mode, cases) in test_data.into_iter().enumerate() {
        for (uastc, expected) in cases {
            let actual = f(*uastc).unwrap();
            assert_eq!(
                &actual,
                expected,
                "\nmode {}\n{}\n{}\n{}\n",
                mode,
                lsb_display(uastc),
                d(&actual[..]),
                d(expected)
            );
        }
    }
}

#[test]
fn transcode_uastc_block_to_astc_returns_expected_block() {
    test_blocks(
        TEST_DATA_UASTC_ASTC,
        basisu::transcode_uastc_block_to_astc,
        lsb_display,
    );
}

#[test]
fn transcode_uastc_block_to_bc7_returns_expected_block() {
    test_blocks(
        TEST_DATA_UASTC_BC7,
        basisu::transcode_uastc_block_to_bc7,
        lsb_display,
    );
}

#[test]
fn transcode_uastc_block_to_etc1_returns_expected_block() {
    test_blocks(
        TEST_DATA_UASTC_ETC1,
        basisu::transcode_uastc_block_to_etc1,
        lsb_display,
    );
}

#[test]
fn transcode_uastc_block_to_etc2_returns_expected_block() {
    test_blocks(
        TEST_DATA_UASTC_ETC2,
        basisu::transcode_uastc_block_to_etc2,
        lsb_display,
    );
}

#[test]
fn unpack_uastc_block_to_rgba_returns_expected_block() {
    test_blocks(
        TEST_DATA_UASTC_RGBA,
        basisu::unpack_uastc_block_to_rgba,
        rgba_display,
    );
}
