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
- [ ] Decoding Huffman tables

## Log

Here I'm writing a log of what I did, problems I encountered, and what I learned. Have anything to say or discuss? I'd be happy to hear from you, please send me a DM or @ me on Twitter [@JakubValtar](https://twitter.com/jakubvaltar).

### 30-05-2020

I wrote my own LSB-first bit reader, it's naive and will need to be optimized later. The code length array is now looking reasonable and I can convert it into a Huffman table by using the algorithm written in the Deflate spec.

### 29-05-2020

I had only half an hour today, but managed to figure it out. Turns out you can read bits in two ways, MSB-first or LSB-first. To get some intuition for this, I highly recommend [Reading bits in far too many ways (part 1)](https://fgiesen.wordpress.com/2018/02/19/reading-bits-in-far-too-many-ways-part-1/) by Fabian Giesen. I blindly used a create which silently assumed MSB-first. Silly me.

What made the whole situation worse is that the data is tightly packed into only as many bits as needed to represent all valid values. If you read something from a wrong offset or if you even reverse the bits, you still get reasonably looking value. I have to be careful about this when dealing with compression. I can't rely on getting obviously wrong values when I make a mistake, I need to validate the data in more sophisticated ways.

### 28-05-2020

Wrapping my head around Huffman tables (Section 6.0 of the spec). I don't have experience with Huffman coding or Deflate, so I kind of expected this would be the difficult part. I got a pretty good understanding of Huffman coding after reading the [section 3.2. of Deflate - RFC 1951](https://tools.ietf.org/html/rfc1951) (the link is provided in the Basis spec).

Sadly my understanding did not translate well into practice. I got already stuck on reading the first array of code lengths. The bit stream contains a variable number of code lengths, plus they are sent in a different order than the order used for creating the Huffman table. This confused me a lot and I had to take a look at the [transcoder implementation](https://github.com/BinomialLLC/basis_universal/blob/6ef114ac1e0665b233c04fcb2e1249400ec65044/contrib/previewers/lib/basisu_transcoder.h#L919) to figure out what all of this means. Including the code to read this array in the spec would help a lot.

I finally managed to read the first code length array, but the lengths did not represent a valid Huffman table. This created more confusion. Am I reading at the right offset? Are these really code lenghts, or do I need to decode these somehow to get valid code lengths?

### 27-05-2020 

Reading the file header and slice descriptions was pretty straighforward. The only surprise was the use of 24-bit integers, Rust does not have 24-bit primitive types, luckily `ByteOrder` crate I used can read them into `u32`. I could have used `packed_struct` or something similar instead of reading all the fields manually, but for now I don't think it's worth the complexity.

Implementing CRC-16 was super easy, the code provided in the spec is short, clear, and easy to convert. Worked on the first try.
