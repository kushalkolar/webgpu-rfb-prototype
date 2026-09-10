// Decode ZLE -> 2D inverse DCT (AAN butterfly inverse, separable) -> luma-to-rgba pixel.
//
// One workgroup processes a 2x2 grid of 8x8 blocks = 4 blocks = 16x16 pixels.
// workgroup_size = (8, 4): 32 threads.
//   - lid.y selects the sub-block (b_in_wg, 0..3).
//   - lid.x is the col index in phase 1 and the row index in phase 2.
//
// Each thread runs exactly one 1D inverse butterfly per phase (32 threads * 8 directions
// * 4 blocks / 8 threads-per-block-per-direction = 32 active threads per phase).
//
// The AAN forward in aan_dct.wgsl maps spatial -> coef via a 6-stage butterfly followed
// by a per-frequency scale by s_k. The inverse is: pre-divide by s_k (here absorbed
// into the dequantize step as INV_S[u]*INV_S[v]), then run those 6 stages backwards.


@group(0) @binding(0)
var<storage, read> n_non_zeros: array<u32>;

@group(0) @binding(1)
var<storage, read> block_starts: array<u32>;

@group(0) @binding(2)
var<storage, read> zle_packed: array<u32>;

@group(0) @binding(3)
var tex_out: texture_storage_2d<rgba8unorm, write>;


const PI_VALUE: f32 = 3.1415927;
const SQRT_2: f32 = 1.4142135;

// AAN constants - identical to the forward shader so the inverse exactly cancels.
const a1: f32 = SQRT_2;
const a2: f32 = 0.27059805;
const a3: f32 = a1;
const a4: f32 = 0.27059803407241206;
const a5: f32 = 0.3826834323650898;

const s0: f32 = (1.0 * SQRT_2/2)/(1       );
const s1: f32 = (cos(1.*PI_VALUE/16)/2)/(-a5+a4+1);
const s2: f32 = (cos(2.*PI_VALUE/16)/2)/(a1+1    );
const s3: f32 = (cos(3.*PI_VALUE/16)/2)/(a5+1    );
const s4: f32 = s0;
const s5: f32 = (cos(5.*PI_VALUE/16)/2)/(1-a5    );
const s6: f32 = (cos(6.*PI_VALUE/16)/2)/(1-a1    );
const s7: f32 = (cos(7.*PI_VALUE/16)/2)/(a5-a4+1 );

// 1/s_k - applied during decode so the butterfly itself has no per-frequency scaling.
const INV_S = array<f32, 8>(1.0/s0, 1.0/s1, 1.0/s2, 1.0/s3, 1.0/s4, 1.0/s5, 1.0/s6, 1.0/s7);

const INV_A1: f32 = 1.0 / a1;
const INV_A3: f32 = 1.0 / a3;

// 2x2 inverse of the (d4, d6) -> (e4, e6) stage:
//   e4 = -(a2+a5)*d4 - a5*d6
//   e6 = -a5*d4 + (a4-a5)*d6
// det = a5*(a2-a4) - a2*a4  (=~ -a2*a4 since a2 =~ a4 in the user's forward).
const D44: f32 = (a5 - a4) / (a2 * a4);
const D46: f32 = -a5       / (a2 * a4);
const D66: f32 = (a2 + a5) / (a2 * a4);


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


const ZZ_INDEX = array<vec2u, 64>(
    vec2u(0, 0), vec2u(1, 0), vec2u(0, 1), vec2u(0, 2),
    vec2u(1, 1), vec2u(2, 0), vec2u(3, 0), vec2u(2, 1),
    vec2u(1, 2), vec2u(0, 3), vec2u(0, 4), vec2u(1, 3),
    vec2u(2, 2), vec2u(3, 1), vec2u(4, 0), vec2u(5, 0),
    vec2u(4, 1), vec2u(3, 2), vec2u(2, 3), vec2u(1, 4),
    vec2u(0, 5), vec2u(0, 6), vec2u(1, 5), vec2u(2, 4),
    vec2u(3, 3), vec2u(4, 2), vec2u(5, 1), vec2u(6, 0),
    vec2u(7, 0), vec2u(6, 1), vec2u(5, 2), vec2u(4, 3),
    vec2u(3, 4), vec2u(2, 5), vec2u(1, 6), vec2u(0, 7),
    vec2u(1, 7), vec2u(2, 6), vec2u(3, 5), vec2u(4, 4),
    vec2u(5, 3), vec2u(6, 2), vec2u(7, 1), vec2u(7, 2),
    vec2u(6, 3), vec2u(5, 4), vec2u(4, 5), vec2u(3, 6),
    vec2u(2, 7), vec2u(3, 7), vec2u(4, 6), vec2u(5, 5),
    vec2u(6, 4), vec2u(7, 3), vec2u(7, 4), vec2u(6, 5),
    vec2u(5, 6), vec2u(4, 7), vec2u(5, 7), vec2u(6, 6),
    vec2u(7, 5), vec2u(7, 6), vec2u(6, 7), vec2u(7, 7),
);


// 4 blocks * 64 freq coefs (pre-divided by s_u*s_v).
var<workgroup> freq_coef: array<array<f32, 64>, 4>;
// 4 blocks * 64 intermediate after the 1D inverse butterfly on cols (phase 1).
var<workgroup> temp_2d: array<array<f32, 64>, 4>;


// Reverse butterfly of aan_dct.wgsl's 1D forward. Inputs are coefs at frequency
// positions 0..7, ALREADY pre-divided by s_k (decode does that).
fn aan_idct_1d(
    c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32,
) -> array<f32, 8> {
    // The forward writes g_i to permuted positions; recover them from coefs.
    let g0 = c0;
    let g5 = c1;
    let g2 = c2;
    let g7 = c3;
    let g1 = c4;
    let g4 = c5;
    let g3 = c6;
    let g6 = c7;

    // g -> f  (inverse of forward's f -> g)
    let f0 = g0;
    let f1 = g1;
    let f2 = g2;
    let f3 = g3;
    let f4 = (g4 - g7) * 0.5;
    let f5 = (g5 + g6) * 0.5;
    let f6 = (g5 - g6) * 0.5;
    let f7 = (g4 + g7) * 0.5;

    // f -> e  (inverse of forward's e -> f)
    let e0 = f0;
    let e1 = f1;
    let e2 = (f2 - f3) * 0.5;
    let e3 = (f2 + f3) * 0.5;
    let e4 = f4;
    let e5 = (f5 - f7) * 0.5;
    let e6 = f6;
    let e7 = (f5 + f7) * 0.5;

    // e -> d  (inverse of forward's d -> e; (d4, d6) via the 2x2 inverse)
    let d0 = e0;
    let d1 = e1;
    let d2 = e2 * INV_A1;
    let d3 = e3;
    let d5 = e5 * INV_A3;
    let d7 = e7;
    let d4 = e4 * D44 + e6 * D46;
    let d6 = e4 * D46 + e6 * D66;

    // d -> c  (inverse of forward's c -> d)
    let cc0 = (d0 + d1) * 0.5;
    let cc1 = (d0 - d1) * 0.5;
    let cc2 = d2 - d3;
    let cc3 = d3;
    let cc4 = d4;
    let cc5 = d5;
    let cc6 = d6;
    let cc7 = d7;

    // c -> b  (inverse of forward's b -> c)
    let b0 = (cc0 + cc3) * 0.5;
    let b3 = (cc0 - cc3) * 0.5;
    let b1 = (cc1 + cc2) * 0.5;
    let b2 = (cc1 - cc2) * 0.5;
    let b7 = cc7;
    let b6 = cc6 - cc7;
    let b5 = cc5 - cc6 + cc7;
    let b4 = -cc4 - cc5 + cc6 - cc7;

    // b -> spatial  (inverse of forward's i -> b)
    var r: array<f32, 8>;
    r[0] = (b0 + b7) * 0.5;
    r[7] = (b0 - b7) * 0.5;
    r[1] = (b1 + b6) * 0.5;
    r[6] = (b1 - b6) * 0.5;
    r[2] = (b2 + b5) * 0.5;
    r[5] = (b2 - b5) * 0.5;
    r[3] = (b3 + b4) * 0.5;
    r[4] = (b3 - b4) * 0.5;
    return r;
}


@compute @workgroup_size(8, 4, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n_blocks_x = textureDimensions(tex_out).x / 8u;
    let block_in_wg = lid.y;
    let sub_x = block_in_wg & 1u;
    let sub_y = block_in_wg >> 1u;
    let block_x = wid.x * 2u + sub_x;
    let block_y = wid.y * 2u + sub_y;
    let block_index = block_y * n_blocks_x + block_x;

    // Decode: 4 threads (lid.x == 0) each handle one sub-block, dequantize and
    // pre-divide by s_u * s_v so the inverse butterfly itself has no s_k scaling.
    if lid.x == 0u {
        for (var k = 0u; k < 64u; k++) {
            freq_coef[block_in_wg][k] = 0.0;
        }
        let n = n_non_zeros[block_index];
        let zle_base = block_starts[block_index];
        var pos: u32 = 0u;
        for (var j = 0u; j < n; j++) {
            let packed = zle_packed[zle_base + j];
            let zeros = packed & 0xFFFFu;
            let q = bitcast<i32>(packed) >> 16u;
            pos = pos + zeros;
            let zz = ZZ_INDEX[pos];
            let inv_s = INV_S[zz.y] * INV_S[zz.x];
            freq_coef[block_in_wg][zz.y * 8u + zz.x] =
                f32(q) * QTABLE_MEDIUM_GRAY[zz.y * 8u + zz.x] * inv_s;
            pos = pos + 1u;
        }
    }
    workgroupBarrier();

    // Phase 1: read col v of freq_coef, apply inverse_1d (= M_orth^T), write as row v
    // of temp_2d. The forward shader's row_pass + transpose + row_pass makes the stored
    // DCT equal to (M X^T M^T), so the inverse is X = M_orth^T G^T M_orth where
    // G = freq_coef; this step computes A[v][:] = M_orth^T @ (col v of G).
    let v = lid.x;
    let r1 = aan_idct_1d(
        freq_coef[block_in_wg][0u * 8u + v],
        freq_coef[block_in_wg][1u * 8u + v],
        freq_coef[block_in_wg][2u * 8u + v],
        freq_coef[block_in_wg][3u * 8u + v],
        freq_coef[block_in_wg][4u * 8u + v],
        freq_coef[block_in_wg][5u * 8u + v],
        freq_coef[block_in_wg][6u * 8u + v],
        freq_coef[block_in_wg][7u * 8u + v],
    );
    temp_2d[block_in_wg][v * 8u + 0u] = r1[0];
    temp_2d[block_in_wg][v * 8u + 1u] = r1[1];
    temp_2d[block_in_wg][v * 8u + 2u] = r1[2];
    temp_2d[block_in_wg][v * 8u + 3u] = r1[3];
    temp_2d[block_in_wg][v * 8u + 4u] = r1[4];
    temp_2d[block_in_wg][v * 8u + 5u] = r1[5];
    temp_2d[block_in_wg][v * 8u + 6u] = r1[6];
    temp_2d[block_in_wg][v * 8u + 7u] = r1[7];

    workgroupBarrier();

    // Phase 2: read col j of temp_2d, apply inverse_1d, write as col j of pixel output.
    // This computes col j of X = M_orth^T @ (col j of temp_2d).
    let j = lid.x;
    let r2 = aan_idct_1d(
        temp_2d[block_in_wg][0u * 8u + j],
        temp_2d[block_in_wg][1u * 8u + j],
        temp_2d[block_in_wg][2u * 8u + j],
        temp_2d[block_in_wg][3u * 8u + j],
        temp_2d[block_in_wg][4u * 8u + j],
        temp_2d[block_in_wg][5u * 8u + j],
        temp_2d[block_in_wg][6u * 8u + j],
        temp_2d[block_in_wg][7u * 8u + j],
    );

    // Luma in [-127.5, 127.5] back to rgba8unorm. r2[i] = X[i][j]: write a vertical
    // 8-pixel stripe at col j of the block.
    let pixel_x = block_x * 8u + j;
    let pixel_y_base = block_y * 8u;
    for (var i = 0u; i < 8u; i++) {
        let pixel = clamp(r2[i] / 255.0 + 0.5, 0.0, 1.0);
        textureStore(tex_out, vec2u(pixel_x, pixel_y_base + i), vec4f(pixel, pixel, pixel, 1.0));
    }
}
