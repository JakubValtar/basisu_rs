# basisu_rs

This is a work in progress Rust implementation of a [Basis Universal](https://github.com/BinomialLLC/basis_universal) transcoder, based on the [.basis File Format and ETC1S Texture Video Specification](https://github.com/BinomialLLC/basis_universal/wiki/.basis-File-Format-and-ETC1S-Texture-Video-Specification) and [UASTC Texture Specification](https://github.com/BinomialLLC/basis_universal/wiki/UASTC-Texture-Specification).

The goal of this project is to support transcoding textures supercompressed with BasisLZ/ETC1S or UASTC into natively supported compressed formats, such as ETC, ASTC, BC7, or unpacked RGBA.

The initial implementation was written in 2020. Currently I'm cleaning up and refactoring the code to release it as a crate.
