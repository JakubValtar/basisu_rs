# basisu_rs

I'm writing a [Basis Universal](https://github.com/BinomialLLC/basis_universal) decoder in Rust to learn more about compression and optimization through [learning in public](https://www.mentalnodes.com/the-only-way-to-learn-in-public-is-to-build-in-public).

The code is WIP and not fit for any practical usage.

Specification: [.basis File Format and ETC1S Texture Video Specification](https://github.com/BinomialLLC/basis_universal/wiki/.basis-File-Format-and-ETC1S-Texture-Video-Specification)

Sample textures were copied from the official [basis_universal repo](https://github.com/BinomialLLC/basis_universal/tree/d0ee14e1fb34ce92adf877a20e3a8226ced6dcdd/webgl/texture/assets) under Apache License 2.0.


## Progress

- [x] Reading FileHeader
- [x] CRC-16
- [x] Reading SliceDesc
- [x] LSB-first bit reader
- [x] Decoding Huffman tables
- [x] Decoding endpoint codebooks
- [x] Decoding selector codebooks
- [x] Decoding ETC1S slices
- [x] Textures with flipped Y
- [x] Textures with dimensions not divisible by 4
- [x] Writing out ETC1 textures
- [x] Lookup tables for faster Huffman decoding
- [x] Decoding UASTC
- [x] Test on more textures
- [x] Transcoding UASTC to ASTC
- [x] Transcoding UASTC to BC7
- [x] Transcoding UASTC to ETC1
- [ ] Transcoding UASTC to ETC2
- [ ] Crate API
- [ ] Check for invalid input data (see Illegal Encodings chapter)
- [ ] Cubemap support
- [ ] Video support

## Log

Here I'm writing a log of what I did, problems I encountered, and what I learned. Have anything to say or discuss? I'd be happy to hear from you, please send me a DM or @ me on Twitter [@JakubValtar](https://twitter.com/jakubvaltar).

### 23-08-2020

I implemented UASTC to ETC1 transcoding. My results were slightly different from the reference, so I had a look at the refrence transcoder source code to figure out how it calculates subblock averages and weight indices.

### 16-08-2020

I collected BC7 test blocks and with the help of the test output implemented BC7 transcoding. The most challenging part was interleaving. Weight index planes are interlaved in UASTC and sequential in BC7. Endpoints are stored as e[subset][channel][lo/hi] in UASTC and e[channel][subset][lo/hi] in BC7. Figuring out in what stage of transcoding to deal with this took some.

I started with a plain array of unquantized endpoint bytes, but processing the endpoints in this form led to some ugly end error-prone code. I ended up decoding and processing the endpoints in UASTC representation (`[[Color32; 2]; subset]`), then handled the interleaving at the very end. Doing it like this made working with endpoints much easier.

I forgot that I have to deinterleave the weight indices. This tripped me up, because anchor indices had different values and sometimes caused inversion of their subset (swapped endpoints and inverted weights). This was not easy to diagnose, because the weight indices in output BC7 blocks would not only have a wrong order, some of them would be inverted as well.

I tested all the currently implemented formats on a test set of ~200 textures encoded and decoded with the reference `basisu` transcoder and the output is bitwise identical.

To summarize all currently implemented functionality:
- ETC1S to RGBA and ETC1
- UASTC to RGBA, ASTC and BC7

### 12-08-2020

Refactoring time! Today I chopped up the UASTC code into smaller functions, which can be used to decode parts of UASTC blocks. Previously I had a big function which fully decoded the block into a kind of intermediate representation, but I realized that transcoding to different formats and to RGBA has very different needs, so I removed the intermediate structs. Now I use the smaller functions to decode only the parts of the blocks I need and directly process the data, without needing a single type to represent all the possible fields each mode can carry.

With this done, implementing the remaining ASTC modes turned out to be quite easy. I copied the block mode bits from the reference blocks I collected earlier. They are constant for each mode, so no harm in a little shortcut. Next I wrote a new bit writer which fills the buffer from the last byte and can optionally reverse the bits it is writing. This was needed for writing the weights, because they are written in the reverse bit order from the end of the block.

The last thing, which probably took me the most time, was repacking endpoints. ASTC has an extra step for packing trits and quints and also interleaves them with bits. The spec describes only decoding, so I implemented that and then generated inverse lookup tables. I ran the test on my texture testing set and some of the textures had one bit flipped. It turns out that there are multiple equivalent representations for some combinations of trits and quints, which means my output was correct too, it just used an alternative values. Anyway, I changed my lookup tables to get the same output as the reference transcoder, to make comparing outputs easier. With this out of the way, transcoding to ASTC is working and matching the reference transcoder!

Next time I'll be looking into transcoding UASTC to BC7.

### 10-08-2020

I refactored tests and generated a set of 32 test blocks for each mode, from UASTC to ASTC and from UASTC to RGBA. This is going to be a big help when implementing transcoding to ASTC, since the block tests are fast, I can choose which modes to test, and I can easily compare expected and actual output. Compare this to testing on the texture set I put together earlier, which can take around a minute and only tells me if the whole texture got transcoded correctly (though I could improve this if I write a comparison function for each compressed format). The texture set is much larger though and is good for running a lot of variety through the transcoder, so it'll be useful in a later stage for finding more subtle bugs which my limited block set didn't find.

Next I wrote a bit writer to make outputting ASTC data easier, since ASTC uses a lot of differently sized fields.

To finally kick off ASTC transcoding, I implemented mode 8 (void-extent). Block tests for mode 8 are passing, which is a good sign.

### 02-08-2020

I put together a set of test images from my visual inspiration folder, around 200 images total. I used ImageMagick to add an alpha channel to all images (it's just the luminanace of the RGB image rotated 180°), from these RGBA images I created RGB, Luminanace-Alpha and Luminance images by making them grayscale and removing the alpha. I encoded all images to `.basis` files using the reference encoder `basisu.exe`, once using ETC1S mode and once using UASTC mode. Then I unpacked the `.basis` files using `basisu.exe`. It helpfully outputs a set KTX files which contain the texture data transcoded into various formats, like ASTC, BC7, ETC1 and ETC2. The KTX files are paired with matching PNG files, showing how they look after decoding.

I fixed a minor bug in ETC1S to ETC1 transcoding, turns out I allocated sixteen times more space for the ETC1 texture than needed and the tail was just zeros. Otherwise I didn't find any problems in ETC1S decoding and transcoding to ETC1!

I found a minor problem in decoding UASTC too, the dimensions of the decoded texture were based on the number of the blocks and not the original texture size stored in the header. This makes a difference when the original size is not a multiple of 4, because the last column/row of the 4x4 blocks needs to be cropped. I already implemanted the cropping when I was writing ETC1S part, I only had to set the right dimensions in the header of the output image.

I had a problem getting the refrence encoder to use all block modes. I had to use `-uastc_level 3` to set a higher quality (default is 2), then it started producing blocks with multiple subsets (like mode 2, 3, 7) and higher weight precision (mode 18).

Last but not least, I refactored UASTC decoding in preparation for ASTC trancoding implementation. I also made some minor changes to remove a bunch of clippy warnings.

### 18-07-2020

Today I implemented all the multiple subset modes. I used the partition pattern tables from the spec. Changing the decoding function to use the tables for appropriate modes was pretty straightforward. What gave me some trouble were anchor weight indices. At first I though I need to worry about them only during encoding. I read about anchors in the [BC7 chapter of the Khronos spec](https://www.khronos.org/registry/DataFormat/specs/1.1/dataformat.1.1.html#_bc7) and it made things a bit clearer. After that I was able to use the anchor tables from the UASTC spec for decoding weights correctly.

I also spotted an error in ETC1 selector storage. Should be fixed in [6824f2](https://github.com/JakubValtar/basisu_rs/commit/6824f262293c0435e53db7c8c32cd1ca86dcbe4a), though there will probably be more bugs, because I still don't have tests for ETC1S -> ETC1 transcoding.

### 17-07-2020

It's time for UASTC! First I implemented mode 8, which is the easiest, a single color block. Next I implemented mode 0, which looked like the simplest mode using endpoints and weights; it's an RGB mode with a single plane and a single subset. For this to work I had to first implement BISE decoding. It is not hard to understand, but dealing with incomplete groups of trits/quints make it a bit more involved.

Then comes endpoint unquantization, which is a hot contender for the most complex thing I had to figure out so far. I somehow missed the precomputed table in the UASTC spec, so I implemented it from the [Khronos spec](https://www.khronos.org/registry/DataFormat/specs/1.1/dataformat.1.1.html#astc-endpoint-unquantization). For weight dequantization I used the lookup table from the UASTC spec.

I was following the Khronos spec for calculating texel colors from endpoints and weights and I accidentally implemented blue contraction as well. Turns out it is not used in UASTC. Other thing which caused some problems was using sRGB in ASTC interpolation. I was not sure if I should use sRGB mode or not, there was no field for it and I couln't find it in the spec. Leaving it on made some colors slightly different than they should be, so I turned it off. Later I found in the project readme that UASTC indeed does not use sRGB (yet).

With mode 0 working, it was easy to add other RGB modes which have a single plane and a single subset, just different ranges of endpoints/weights (these are 1, 5, 18). After this, adding RGBA modes (10, 12, 14) and Luminance+Alpha modes (15) was pretty trivial, most of the code is the same, except for calculating base colors.

The next step was adding dual-plane modes (6, 11, 13, 17). Dual-plane means that each texel now has two weights, the second weight being used for one component in the block. E.g. red, green and alpha can use one weight, while blue uses the other one. I had to make a small change to weight decoding to decode the second set of weights correctly. I got a little bit confused by anchor weight indices, but it turned out they are only used in modes with multiple subsets, not multiple planes. Other than that, adding a second set of weights to the existing block decoding was easy.

I struggled a bit with mode 17, I forgot that it does not have a `compsel` field to select which component uses the second set of weights. My code defaulted to 0 for Red, and it caused some red/cyan artifacts in mode 17 blocks. After a while I found in the spec that mode 17 has always `compsel = 3` for Alpha. It makes sense, it is a grayscale mode, so using any other value would cause coloration, but I didn't realize it at the time. I was using the 64 example blocks from the UASTC spec as a basic test that my code is doing the right thing. After the trouble with mode 17, I found out that there are no blocks with mode 7, 16, or 17 in the example set. I'm planning to add more testing data soon.

Tomorrow I will be implementing modes with multiple subsets.

### 27-06-2020

I decided to optimize Huffman decoding today, to get the low-hanging fruit. I still run a linear search to find the right entry in the decoding table.

First I switched to a sparse table ([6824f2](https://github.com/JakubValtar/basisu_rs/commit/6824f262293c0435e53db7c8c32cd1ca86dcbe4a)). I added the symbol to the table entry, which allowed me to get rid of empty entries and to sort the table by code size, since the entries are not tied to the position in the table anymore. This led to a 2x speedup, because shorter codes appear more frequently in the bit stream.

Next I switched to a lookup table ([46ffc8](https://github.com/JakubValtar/basisu_rs/commit/46ffc8752fe78639c542719a2dbd7e8d4f5e1f47)). I decided to keep it simple and start with a full table. The maximum code size is 16 bits, which means 2^16 entries. Each entry is copied into all the slots which end with the code of the entry. This led to a 15x speedup.

The last thing I did today was to make the size of the lookup table adapt to the size of the longest code present ([29e323](https://github.com/JakubValtar/basisu_rs/commit/29e3233c3e55741211dc9cdf41f8f7ba36843fc4)). The decoding table is constructed from a slice of code sizes of the symbols, so it's easy to find the longest code. Instead of 2^16, the lookup table size is now 2^max_code_size. This helps, because half of the tables have at most 21 entries and we don't have to waste time on generating lookup tables with 2^16 entries. This led to a 1.5x speedup.

The combined speedup for today ended up being around 45x. I'm pretty happy with this :)

### 26-06-2020

Implemented writing out ETC1 textures, I used the [Khronos spec](https://www.khronos.org/registry/DataFormat/specs/1.1/dataformat.1.1.html#ETC1). I refactored selectors to use less space and added a field which stores the values in ETC1 format. Since selectors are reused, preparing this data beforehand makes much more sense than recalculating it for every block.

I found a bug in the RGBA decoding in the process. The ETC1 spec says that 5-bit color values should be converted to 8-bit by shifting three bits left and replicating top three bits into the bottom three bits. I wasn't replicating the bits, so there was some loss of quality, although barely noticeable.

There doesn't seem to be any easy way to export the ETC1 data in some file format which could be opened by some engine or texture viewer. I will have to look into validating the ETC1 output later.

### 22-06-2020

Now handling textures with dimensions not divisible by 4 and textures with Y flipped.

### 20-06-2020

Just a little quality of life improvement today: added a byte reader to simplify reading structs.

### 07-06-2020

I implemented ETC1S slice decoding, it was mostly a rewrite of the code in the spec (again). I checked the [unity blog - Crunch compression of ETC textures](https://blogs.unity3d.com/2017/12/15/crunch-compression-of-etc-textures/) to learn more about ETC1 and how to decode endpoints and selectors into RGBA data. I added PNG export to check that the textures are being decoded correctly.

I was getting garbage images at first, because I was reading from a wrong part of the file. During debugging, I went through the selector decoding code again and simplified it to a bare minimum to reduce complexity. It's probably somewhat slower now, but it's clear what's going on and there will be time for optimization later.

Last but not least, I organized the crate a bit to make space for upcoming tasks: writing out ETC1S textures, which can be decoded by the graphics card, and decoding UASTC textures.

### 01-06-2020

I added functions to read endpoints and selectors. This was basically a rewrite of the code from the spec, I need to go through it tomorrow again to get a better grasp of what is happening and how the selectors are stored.

I'm not sure how to test this code, I think I will have to wait till I have the slice decoding working, then I can verify CRC-16 of the decoded texture data for each ETC1S slice.

### 31-05-2020

I added a test for the bit reader and fixed a bug which I found.

I implemented a simple Huffman decoding function which decodes a bitstream into symbols. It does a linear search through the codes in the decoding table until if finds a match. Decoding could be optimized by storing the symbol in the table entry and sorting the entries by frequency (code size). Ideally there should be a lookup table, but I'm not adding it right now, because I want to have a working decoder first and make optimizations later.

The decoding didn't work at first, because the codes in the Huffman tables (as generated with the code from the Deflate RFC 1951) expect a bit encounter order `MSB -> LSB`, but the bit reader returns the bits in order `LSB -> MSB`. Reversing the bit order of the codes in the decoding table solved this problem.

Now that the Huffman decoding works, I'm able to decode all the Huffman tables in the file.

Another issue was that if the highest 16-bit code was used, the `next_code` would overflow and crash the program. Making `next_code` 32-bit fixed this and also allowed me to add a check for codes going higher than `u16::MAX`. I could have kept it in 16 bits and used `wrapping_add` instead, but then I wouldn't be able to check if the overflowed codes were used or not.

### 30-05-2020

I wrote my own LSB-first bit reader, it's naive and will need to be optimized later. The code length array is now looking reasonable and I can convert it into a Huffman table by using the algorithm written in the Deflate spec.

### 29-05-2020

I had only half an hour today, but managed to figure it out. Turns out you can read bits in two ways, MSB-first or LSB-first. To get some intuition for this, I highly recommend [Reading bits in far too many ways (part 1)](https://fgiesen.wordpress.com/2018/02/19/reading-bits-in-far-too-many-ways-part-1/) by Fabian Giesen. I blindly used a create which silently assumed MSB-first. Silly me.

What made the whole situation worse is that the data is tightly packed into only as many bits as needed to represent all valid values. If you read something from a wrong offset or if you even reverse the bits, you still get reasonably looking value. I have to be careful about this when dealing with compression. I can't rely on getting obviously wrong values when I make a mistake, I need to validate the data in more sophisticated ways.

### 28-05-2020

Wrapping my head around Huffman tables (Section 6.0 of the spec). I don't have experience with Huffman coding or Deflate, so I kind of expected this would be the difficult part. I got a pretty good understanding of Huffman coding after reading the [Huffman coding section of Modern LZ Compression](https://glinscott.github.io/lz/index.html#toc3) and the [section 3.2. of Deflate - RFC 1951](https://tools.ietf.org/html/rfc1951) (linked from the Basis spec).

Sadly my understanding did not translate well into practice. I got already stuck on reading the first array of code lengths. The bit stream contains a variable number of code lengths, plus they are sent in a different order than the order used for creating the Huffman table. This confused me a lot and I had to take a look at the [transcoder implementation](https://github.com/BinomialLLC/basis_universal/blob/6ef114ac1e0665b233c04fcb2e1249400ec65044/contrib/previewers/lib/basisu_transcoder.h#L919) to figure out what all of this means. Including the code to read this array in the spec would help a lot.

I finally managed to read the first code length array, but the lengths did not represent a valid Huffman table. This created more confusion. Am I reading at the right offset? Are these really code lenghts, or do I need to decode these somehow to get valid code lengths?

### 27-05-2020

Reading the file header and slice descriptions was pretty straighforward. The only surprise was the use of 24-bit integers, Rust does not have 24-bit primitive types, luckily `ByteOrder` crate I used can read them into `u32`. I could have used `packed_struct` or something similar instead of reading all the fields manually, but for now I don't think it's worth the complexity.

Implementing CRC-16 was super easy, the code provided in the spec is short, clear, and easy to convert. Worked on the first try.
