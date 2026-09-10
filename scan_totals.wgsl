// Single-workgroup in-place exclusive scan of wg_totals (length TILE = 512). Same
// thread-coarsened Kogge-Stone as scan_local; no per-tile output to emit.

@group(0) @binding(0)
var<storage, read_write> wg_totals: array<u32>;


const W: u32 = 64u;
const K: u32 = 8u;
const TILE: u32 = 512u;


var<workgroup> sm: array<array<u32, W>, 2>;


@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let base = i * K;

    var r: array<u32, 8>;
    r[0] = wg_totals[base + 0u];
    r[1] = wg_totals[base + 1u];
    r[2] = wg_totals[base + 2u];
    r[3] = wg_totals[base + 3u];
    r[4] = wg_totals[base + 4u];
    r[5] = wg_totals[base + 5u];
    r[6] = wg_totals[base + 6u];
    r[7] = wg_totals[base + 7u];

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

    wg_totals[base + 0u] = r[0] + thread_prefix;
    wg_totals[base + 1u] = r[1] + thread_prefix;
    wg_totals[base + 2u] = r[2] + thread_prefix;
    wg_totals[base + 3u] = r[3] + thread_prefix;
    wg_totals[base + 4u] = r[4] + thread_prefix;
    wg_totals[base + 5u] = r[5] + thread_prefix;
    wg_totals[base + 6u] = r[6] + thread_prefix;
    wg_totals[base + 7u] = r[7] + thread_prefix;
}
