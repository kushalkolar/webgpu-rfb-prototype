@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

// TODO: once this is working, use uint8 for quantized output
@group(0) @binding(1)
var tex_y_dct: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var tex_cbcr: texture_storage_2d<rg32float, write>;

@group(0) @binding(3)
var chroma_sampler: sampler;

// DCT basis, shape from numpy is [64, 8, 8], IN ZIGZAG ORDER!, [b, m, n] which becomes [z, y, x] in the shader
@group(0) @binding(4)
var dct_basis: texture_3d<f32>;

// block size
override group_size_x: u32;
override group_size_y: u32;


// the zigzag indices that set the order of the basis
// used for indexing the texture
// note that each vector in this array is [x, y] which corresponds to [col, row] in numpy! this is flipped.
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


@compute @workgroup_size(group_size_x, group_size_y)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    // wid.xy is the workgroup invocation ID
    // to get the starting (x, y) texture coordinate for a given (8, 8) block we must multiply by workgroup size
    // Example:
    //  workgroup invocation id (0, 0) becomes texture coord (0, 0)
    //  workgroup invocation id (1, 0) becomes texture coord (8, 0) (block size is 8x8)
    //  workgroup invocation id (1, 1) becomes texture coord (8, 8)
    // We can iterate through pixels within this block by just adding to this starting (x, y) position
    // upto the max position which is (start_x + group_size_x, start_y + group_size_y)

    // start and stop indices for this block
    let start = wid.xy * vec2u(group_size_x, group_size_y);
    let stop = start + vec2u(group_size_x, group_size_y);

    // write luminance for each pixel in this block
    // go in the order of the basis
    for (var basis_index: u32 = 0; basis_index < 64; basis_index++) {
        let pos_basis = ZZ_INDEX[basis_index];

        // used to accumulate the weight for the current basis iteration from the image block
        var basis_weight: f32 = 0.0;

        // go through each pixel in the block
        var block_x_index: u32 = 0;
        for (var x: u32 = start.x; x < stop.x; x++) {
            var block_y_index: u32 = 0;
            for (var y: u32 = start.y; y < stop.y; y++) {
                let pos = vec2u(x, y);

                // read array element i.e. "pixel" value
                var px: vec4f = textureLoad(tex_rgba, pos, 0);

                // get luma value for this pixel
                var luma: f32 = (0.299 * px.r + 0.587 * px.g + 0.114 * px.b) * 255;

                // JPEG DCT uses Hadamard multiply, so multiply the DCT at this basis at the corresponding pixel index
                // textureLoad always returns vec4f32 even if it's just a 2D array
                var dct_element: vec4<f32> = textureLoad(dct_basis, vec3u(block_x_index, block_y_index, basis_index), 0);
                basis_weight += dct_element[0] * luma;

                block_y_index += 1;
            }
            block_x_index += 1;
        }

        // store the weight in this basis for the block
        // TODO: Quantize
        textureStore(tex_y_dct, start + pos_basis, vec4<f32>(basis_weight, 0, 0, 0));
    }

    // TODO: dct for chroma
    // chroma subsampling
    for (var x: u32 = start.x; x < stop.x; x += 2) {
        for (var y: u32 = start.y; y < stop.y; y += 2) {
            // convert to normalized uv coords for sampler
            let coords_sample: vec2f = (vec2f(f32(x), f32(y)) + 0.5) / vec2f(textureDimensions(tex_rgba).xy);

            var px_sample: vec4f = textureSampleLevel(tex_rgba, chroma_sampler, coords_sample, 0.0);

            // create cb, cr channels
            var cb: f32 = (-0.1687 * px_sample.r - 0.3313 * px_sample.g + 0.5 * px_sample.b) + 0.5;
            var cr: f32 = (0.5 * px_sample.r - 0.4187 * px_sample.g - 0.0813 * px_sample.b) + 0.5;
            let pos_out: vec2u = vec2u(x / 2, y / 2);
            textureStore(tex_cbcr, pos_out.xy, vec4<f32>(cb, cr, 0, 0));
        }
    }
}
