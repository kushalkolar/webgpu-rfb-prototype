// Copy each block's packed RLE pairs from the per-block scratch into the contiguous
// zle_packed buffer at block_starts[b]..block_starts[b]+n_non_zeros[b].
//
// 2D dispatch (n_blocks_x, n_blocks_y) to stay under maxComputeWorkgroupsPerDimension
// at 4K; flat block index reconstructed via num_workgroups.

@group(0) @binding(0)
var<storage, read> per_block_rle: array<u32>;

@group(0) @binding(1)
var<storage, read> n_non_zeros: array<u32>;

@group(0) @binding(2)
var<storage, read> block_starts: array<u32>;

@group(0) @binding(3)
var<storage, read_write> zle_packed: array<u32>;


@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let b = wid.y * nwg.x + wid.x;
    let k = lid.x;

    if k < n_non_zeros[b] {
        zle_packed[block_starts[b] + k] = per_block_rle[b * 64u + k];
    }
}
