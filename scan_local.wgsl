// Tile-local exclusive prefix scan of n_non_zeros into block_starts, plus one per-tile
// total written to wg_totals[wid.x]. Thread-coarsened Kogge-Stone:
//   each thread serial-scans K=8 contiguous u32s in private registers,
//   then a workgroup-level Kogge-Stone runs over the W=64 thread-totals,
//   log2(W) = 6 barriered stages with all threads active throughout.
// Tile size TILE = W*K = 512. Two-level scan (this + scan_totals) supports up to
// TILE*TILE = 262144 blocks (4096x4096 luma).

@group(0) @binding(0)
var<storage, read> n_non_zeros: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_starts: array<u32>;

@group(0) @binding(2)
var<storage, read_write> wg_totals: array<u32>;


const W: u32 = 64u;
const K: u32 = 8u;
const TILE: u32 = 512u;


var<workgroup> sm: array<array<u32, W>, 2>;


@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = lid.x;
    let base = wid.x * TILE + i * K;

    // Load K elements per thread, serial-scan in registers.
    var r: array<u32, 8>;
    r[0] = n_non_zeros[base + 0u];
    r[1] = n_non_zeros[base + 1u];
    r[2] = n_non_zeros[base + 2u];
    r[3] = n_non_zeros[base + 3u];
    r[4] = n_non_zeros[base + 4u];
    r[5] = n_non_zeros[base + 5u];
    r[6] = n_non_zeros[base + 6u];
    r[7] = n_non_zeros[base + 7u];

    let s0 = r[0];
    let s1 = s0 + r[1];
    let s2 = s1 + r[2];
    let s3 = s2 + r[3];
    let s4 = s3 + r[4];
    let s5 = s4 + r[5];
    let s6 = s5 + r[6];
    let thread_total = s6 + r[7];

    r[0] = 0u; r[1] = s0; r[2] = s1; r[3] = s2;
    r[4] = s3; r[5] = s4; r[6] = s5; r[7] = s6;

    // Workgroup Kogge-Stone over thread_totals.
    var src: u32 = 0u;
    var dst: u32 = 1u;
    sm[src][i] = thread_total;
    workgroupBarrier();

    for (var step: u32 = 1u; step < W; step = step << 1u) {
        var add: u32 = 0u;
        if i >= step {
            add = sm[src][i - step];
        }
        sm[dst][i] = sm[src][i] + add;
        let t = src; src = dst; dst = t;
        workgroupBarrier();
    }

    let thread_prefix = sm[src][i] - thread_total;

    block_starts[base + 0u] = r[0] + thread_prefix;
    block_starts[base + 1u] = r[1] + thread_prefix;
    block_starts[base + 2u] = r[2] + thread_prefix;
    block_starts[base + 3u] = r[3] + thread_prefix;
    block_starts[base + 4u] = r[4] + thread_prefix;
    block_starts[base + 5u] = r[5] + thread_prefix;
    block_starts[base + 6u] = r[6] + thread_prefix;
    block_starts[base + 7u] = r[7] + thread_prefix;

    if i == W - 1u {
        wg_totals[wid.x] = sm[src][i];
    }
}
