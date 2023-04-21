use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

criterion_main!(benches);
criterion_group!(
    benches,
    uastc_to_astc,
    uastc_to_bc7,
    uastc_to_etc1,
    uastc_to_etc2,
    uastc_to_rgba
);

#[path = "../tests/block_test_cases/mod.rs"]
mod block_test_cases;

use block_test_cases::{
    TEST_DATA_UASTC_ASTC, TEST_DATA_UASTC_BC7, TEST_DATA_UASTC_ETC1, TEST_DATA_UASTC_ETC2,
    TEST_DATA_UASTC_RGBA,
};

fn uastc_to_astc(c: &mut Criterion) {
    uastc_to(
        c,
        "uastc_to_astc",
        TEST_DATA_UASTC_ASTC,
        basisu::transcode_uastc_block_to_astc,
    );
}

fn uastc_to_bc7(c: &mut Criterion) {
    uastc_to(
        c,
        "uastc_to_bc7",
        TEST_DATA_UASTC_BC7,
        basisu::transcode_uastc_block_to_bc7,
    );
}

fn uastc_to_etc1(c: &mut Criterion) {
    uastc_to(
        c,
        "uastc_to_etc1",
        TEST_DATA_UASTC_ETC1,
        basisu::transcode_uastc_block_to_etc1,
    );
}

fn uastc_to_etc2(c: &mut Criterion) {
    uastc_to(
        c,
        "uastc_to_etc2",
        TEST_DATA_UASTC_ETC2,
        basisu::transcode_uastc_block_to_etc2,
    );
}

fn uastc_to_rgba(c: &mut Criterion) {
    uastc_to(
        c,
        "uastc_to_rgba",
        TEST_DATA_UASTC_RGBA,
        basisu::unpack_uastc_block_to_rgba,
    );
}

fn uastc_to<U>(
    c: &mut Criterion,
    group_name: &str,
    data: [&[([u8; 16], U)]; 19],
    f: impl Fn([u8; 16]) -> Result<U, String>,
) where
    U: PartialEq,
{
    let mut group = c.benchmark_group(group_name);

    for (mode_id, blocks) in data.into_iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("mode{:02}", mode_id)),
            blocks,
            |b, blocks| {
                b.iter_batched(
                    || blocks,
                    |blocks| {
                        let mut ok = true;
                        for _ in 0..1000 {
                            for (uastc, res) in blocks {
                                ok &= f(*uastc).as_ref() == Ok(res);
                            }
                        }
                        ok
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}
