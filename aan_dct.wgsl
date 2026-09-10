// DCT based on the C++ example here: https://unix4lyfe.org/dct-1d/

@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

//@group(0) @binding(1)
//var tex_y: texture_storage_2d<r32float, write>;

// Per-block scratch RLE. Each u32 slot holds:
//   first  16 bits = preceding zero count
//   second 16 bits = quantized coefficient value (signed)
// Block b's pairs sit at [b*64, b*64 + n_non_zeros[b]); the tail of the 64-slot range is unused.
// The compact pass later copies these slices contiguously into zle_packed using block_starts[]
// from the prefix-sum scan over n_non_zeros.
@group(0) @binding(1)
var<storage, read_write> per_block_rle: array<u32>;

@group(0) @binding(2)
var<storage, read_write> n_non_zeros: array<u32>;


const LUMA_WEIGHTS = vec4f(0.299, 0.587, 0.114, 0);

// for some reason pycharm complains if there is no underscore in constants
// even though the wgsl compiler does not complain, pycharm is just annoyingly buggy these days...
const PI_VALUE: f32 = 3.1415927;

// apparently you can't use muliply into a sqrt function expression
// i.e., this is not valid wgsl: 3 * sqrt(2)
const SQRT_2: f32 = 1.4142135;
const SQRT_HALF: f32 = 0.70710677;

// copy pasted from the C++ code but some things changed to make wgsl happy
const a1: f32 = SQRT_2;
const a2: f32 = 0.27059805; // wgsl doesn't like to multiply into cosine func SQRT_2 * cos(3 / 16 * 2 * PI_VALUE);
const a3: f32 = a1;
const a4: f32 = 0.27059803407241206;
const a5: f32 = 0.3826834323650898;  // cos(3*pi/16)

// copy pasted from the C++ code but some things changed to make wgsl happy
const s0: f32 = (1.0 * SQRT_2/2)/(1       );  // 0.353553
const s1: f32 = (cos(1.*PI_VALUE/16)/2)/(-a5+a4+1);  // 0.254898
const s2: f32 = (cos(2.*PI_VALUE/16)/2)/(a1+1    );  // 0.270598
const s3: f32 = (cos(3.*PI_VALUE/16)/2)/(a5+1    );  // 0.300672
const s4: f32 = s0;  // (cos(4.*M_PI/16)/2)/(1       );
const s5: f32 = (cos(5.*PI_VALUE/16)/2)/(1-a5    );  // 0.449988
const s6: f32 = (cos(6.*PI_VALUE/16)/2)/(1-a1    );  // 0.653281
const s7: f32 = (cos(7.*PI_VALUE/16)/2)/(a5-a4+1 );  // 1.281458


const ZZ_INDEX = array<vec2u, 64>(
    vec2u(0, 0),
    vec2u(1, 0),
    vec2u(0, 1),
    vec2u(0, 2),
    vec2u(1, 1),
    vec2u(2, 0),
    vec2u(3, 0),
    vec2u(2, 1),
    vec2u(1, 2),
    vec2u(0, 3),
    vec2u(0, 4),
    vec2u(1, 3),
    vec2u(2, 2),
    vec2u(3, 1),
    vec2u(4, 0),
    vec2u(5, 0),
    vec2u(4, 1),
    vec2u(3, 2),
    vec2u(2, 3),
    vec2u(1, 4),
    vec2u(0, 5),
    vec2u(0, 6),
    vec2u(1, 5),
    vec2u(2, 4),
    vec2u(3, 3),
    vec2u(4, 2),
    vec2u(5, 1),
    vec2u(6, 0),
    vec2u(7, 0),
    vec2u(6, 1),
    vec2u(5, 2),
    vec2u(4, 3),
    vec2u(3, 4),
    vec2u(2, 5),
    vec2u(1, 6),
    vec2u(0, 7),
    vec2u(1, 7),
    vec2u(2, 6),
    vec2u(3, 5),
    vec2u(4, 4),
    vec2u(5, 3),
    vec2u(6, 2),
    vec2u(7, 1),
    vec2u(7, 2),
    vec2u(6, 3),
    vec2u(5, 4),
    vec2u(4, 5),
    vec2u(3, 6),
    vec2u(2, 7),
    vec2u(3, 7),
    vec2u(4, 6),
    vec2u(5, 5),
    vec2u(6, 4),
    vec2u(7, 3),
    vec2u(7, 4),
    vec2u(6, 5),
    vec2u(5, 6),
    vec2u(4, 7),
    vec2u(5, 7),
    vec2u(6, 6),
    vec2u(7, 5),
    vec2u(7, 6),
    vec2u(6, 7),
    vec2u(7, 7),
);


const QTABLE_MEDIUM_GRAY = array<f32, 64>(
    16, 12, 11, 10, 12, 14, 14, 13,
    14, 16, 24, 19, 16, 17, 18, 24,
    22, 22, 24, 26, 40, 51, 58, 40,
    29, 37, 35, 49, 72, 64, 55, 56,
    51, 57, 60, 61, 55, 69, 87, 68,
    64, 78, 92, 95, 87, 81, 109, 80,
    56, 62, 103, 104, 103, 98, 112, 121,
    113, 77, 92, 120, 100, 103, 101, 99
);


var<workgroup> wg_mem: array<f32, 64>;


@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let flat_lid = lid.y * 8u + lid.x;

    // flat index of this block within the full image; derived from image dims so
    // it works for any input size up to 4096x4096 (n_blocks_x up to 512).
    let n_blocks_x = textureDimensions(tex_rgba).x / 8u;
    let block_index = wid.y * n_blocks_x + wid.x;

    let start = wid.xy * vec2u(8, 8);

    let pixel_pos = start + lid.xy;
    let rgba_val = textureLoad(tex_rgba, pixel_pos, 0);
    let luma = (dot(rgba_val, LUMA_WEIGHTS) - 0.5) * 255.0;
    wg_mem[flat_lid] = luma;

    workgroupBarrier();

    // laster can use templating here maybe instead of copy pasted
    // rows
    if (lid.x == 0u) {
        let index = lid.y * 8u;
        // luma vals
        let i0 = wg_mem[index + 0u];
        let i1 = wg_mem[index + 1u];
        let i2 = wg_mem[index + 2u];
        let i3 = wg_mem[index + 3u];
        let i4 = wg_mem[index + 4u];
        let i5 = wg_mem[index + 5u];
        let i6 = wg_mem[index + 6u];
        let i7 = wg_mem[index + 7u];

        let b0 = i0 + i7;
        let b1 = i1 + i6;
        let b2 = i2 + i5;
        let b3 = i3 + i4;
        let b4 = -i4 + i3;
        let b5 = -i5 + i2;
        let b6 = -i6 + i1;
        let b7 = -i7 + i0;

        let c0 = b0 + b3;
        let c1 = b1 + b2;
        let c2 = -b2 + b1;
        let c3 = -b3 + b0;
        let c4 = -b4 - b5;
        let c5 = b5 + b6;
        let c6 = b6 + b7;
        let c7 = b7;

        let d0 = c0 + c1;
        let d1 = -c1 + c0;
        let d2 = c2 + c3;
        let d3 = c3;
        let d4 = c4;
        let d5 = c5;
        let d6 = c6;
        let d7 = c7;

        let d8 = (d4 + d6) * a5;

        let e0 = d0;
        let e1 = d1;
        let e2 = d2 * a1;
        let e3 = d3;
        let e4 = -d4 * a2 - d8;
        let e5 = d5 * a3;
        let e6 = d6 * a4 - d8;
        let e7 = d7;

        let f0 = e0;
        let f1 = e1;
        let f2 = e2 + e3;
        let f3 = e3 - e2;
        let f4 = e4;
        let f5 = e5 + e7;
        let f6 = e6;
        let f7 = e7 - e5;

        let g0 = f0;
        let g1 = f1;
        let g2 = f2;
        let g3 = f3;
        let g4 = f4 + f7;
        let g5 = f5 + f6;
        let g6 = -f6 + f5;
        let g7 = f7 - f4;

        // IDK what the C++ code is doing in this block
        wg_mem[index + 0u] = g0 * s0;
        wg_mem[index + 4u] = g1 * s4;
        wg_mem[index + 2u] = g2 * s2;
        wg_mem[index + 6u] = g3 * s6;
        wg_mem[index + 5u] = g4 * s5;
        wg_mem[index + 1u] = g5 * s1;
        wg_mem[index + 7u] = g6 * s7;
        wg_mem[index + 3u] = g7 * s3;

    }

    workgroupBarrier();

    // transpose
    if (lid.x < lid.y) {
        // see if wgsl has a function for this
        let index_a = lid.y * 8u + lid.x;
        let index_b = lid.x * 8u + lid.y;
        let to_swap = wg_mem[index_a];
        wg_mem[index_a] = wg_mem[index_b];
        wg_mem[index_b] = to_swap;
    }

    workgroupBarrier();

    // columns
    if (lid.x == 0u) {
                let index = lid.y * 8u;
        // luma vals
        let i0 = wg_mem[index + 0u];
        let i1 = wg_mem[index + 1u];
        let i2 = wg_mem[index + 2u];
        let i3 = wg_mem[index + 3u];
        let i4 = wg_mem[index + 4u];
        let i5 = wg_mem[index + 5u];
        let i6 = wg_mem[index + 6u];
        let i7 = wg_mem[index + 7u];

        let b0 = i0 + i7;
        let b1 = i1 + i6;
        let b2 = i2 + i5;
        let b3 = i3 + i4;
        let b4 = -i4 + i3;
        let b5 = -i5 + i2;
        let b6 = -i6 + i1;
        let b7 = -i7 + i0;

        let c0 = b0 + b3;
        let c1 = b1 + b2;
        let c2 = -b2 + b1;
        let c3 = -b3 + b0;
        let c4 = -b4 - b5;
        let c5 = b5 + b6;
        let c6 = b6 + b7;
        let c7 = b7;

        let d0 = c0 + c1;
        let d1 = -c1 + c0;
        let d2 = c2 + c3;
        let d3 = c3;
        let d4 = c4;
        let d5 = c5;
        let d6 = c6;
        let d7 = c7;

        let d8 = (d4 + d6) * a5;

        let e0 = d0;
        let e1 = d1;
        let e2 = d2 * a1;
        let e3 = d3;
        let e4 = -d4 * a2 - d8;
        let e5 = d5 * a3;
        let e6 = d6 * a4 - d8;
        let e7 = d7;

        let f0 = e0;
        let f1 = e1;
        let f2 = e2 + e3;
        let f3 = e3 - e2;
        let f4 = e4;
        let f5 = e5 + e7;
        let f6 = e6;
        let f7 = e7 - e5;

        let g0 = f0;
        let g1 = f1;
        let g2 = f2;
        let g3 = f3;
        let g4 = f4 + f7;
        let g5 = f5 + f6;
        let g6 = -f6 + f5;
        let g7 = f7 - f4;

        // IDK what the C++ code is doing in this block
        wg_mem[index + 0u] = g0 * s0;
        wg_mem[index + 4u] = g1 * s4;
        wg_mem[index + 2u] = g2 * s2;
        wg_mem[index + 6u] = g3 * s6;
        wg_mem[index + 5u] = g4 * s5;
        wg_mem[index + 1u] = g5 * s1;
        wg_mem[index + 7u] = g6 * s7;
        wg_mem[index + 3u] = g7 * s3;

    }

    workgroupBarrier();

    // zz path -> quantize -> RLE pack. Inherently serial within a block (each pair's
    // zigzag position depends on the prior pair's zero_run), so a single invocation
    // handles it; the other 63 threads exit.
    if flat_lid != 0u {
        return;
    }

    let base = block_index * 64u;
    var zero_run: u32 = 0u;
    var out: u32 = 0u;

    for (var i: u32 = 0u; i < 64u; i++) {
        let zz = ZZ_INDEX[i];
        let src = zz.y * 8u + zz.x;
        let q = i32(round(wg_mem[src] / QTABLE_MEDIUM_GRAY[src]));
        if q == 0 {
            zero_run++;
        } else {
            per_block_rle[base + out] = (zero_run & 0xFFFFu) | ((bitcast<u32>(q) & 0xFFFFu) << 16u);
            out++;
            zero_run = 0u;
        }
    }

    n_non_zeros[block_index] = out;
}
