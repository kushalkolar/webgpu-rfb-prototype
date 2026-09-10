// Add each tile's scanned prefix from wg_totals to every block_starts slot in that tile.
// After this pass block_starts is the global exclusive prefix sum of n_non_zeros.

@group(0) @binding(0)
var<storage, read> wg_totals: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_starts: array<u32>;


const TILE: u32 = 512u;
const K: u32 = 8u;


@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let offset = wg_totals[wid.x];
    let base = wid.x * TILE + lid.x * K;

    block_starts[base + 0u] = block_starts[base + 0u] + offset;
    block_starts[base + 1u] = block_starts[base + 1u] + offset;
    block_starts[base + 2u] = block_starts[base + 2u] + offset;
    block_starts[base + 3u] = block_starts[base + 3u] + offset;
    block_starts[base + 4u] = block_starts[base + 4u] + offset;
    block_starts[base + 5u] = block_starts[base + 5u] + offset;
    block_starts[base + 6u] = block_starts[base + 6u] + offset;
    block_starts[base + 7u] = block_starts[base + 7u] + offset;
}
