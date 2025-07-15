// does 881 luma then one basis per z-direction workgroup
// takes ~51 seconds on my AMD GPU

@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y: texture_storage_2d<r32float, write>;

// weights to convert from RGB -> YCbCr
const LUMA_WEIGHTS = vec4f(0.299, 0.587, 0.114, 0);
const CB_WEIGHTS = vec4f(-0.1687, -0.3313, 0.5, 0);
const CR_WEIGHTS = vec4f(0.5, -0.4187, -0.0813, 0);

const DCT_BASIS: array<array<array<f32, 8>, 8>, 64> = array(
    $$ for basis_index in range(64)
        array(
        $$ for i in range(8)
            array(
            $$ for j in range(8)
                {{ dct_basis[basis_index, i, j] }},
            $$ endfor
            ),
        $$ endfor
        ),
    $$ endfor
);

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


var<workgroup> wg_luma: array<array<f32, 8>, 8>;
var<private> current_weight: f32 = 0.0;

@compute @workgroup_size(8, 1, 64)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>
    ) {
        let start = wid.xy * vec2u(8, 8);
        
        var pos_load: vec2u;
        var rgba_val: vec4f;
        
        $$ for y_i in range(8)
            // store the luma in the workgroup memory
            pos_load = start + vec2u(lid.x, {{ y_i }});
            
            rgba_val = textureLoad(
                tex_rgba,
                pos_load,
                0
            );
            
            wg_luma[lid.x][{{ y_i }}] = (dot(rgba_val, LUMA_WEIGHTS) - 0.5) * 255;
        $$ endfor

        // block until all luma values have been written for this block
        workgroupBarrier();

        // we no longer need 8 invocations along x
        if lid.x > 0 {
            return;
        }

        let basis_index = lid.z;
        // seems like there is no performance difference if we do this or a private location invocation var
//        var current_weight: f32 = 0.0;

        for (var x: u32 = 0; x < 8; x++) {
          $$ for y in range(8)
            //for (var y: u32 = 0; y < 8; y++) {
//                current_weight += DCT_BASIS[basis_index][x][y] * wg_luma[x][y];
                // should benchmark if fma is faster
             //current_weight = fma(DCT_BASIS[basis_index][x][y], wg_luma[x][y], current_weight);
     // {#      current_weight = fma(DCT_BASIS[basis_index][x][{{ y }}], wg_luma[x][{{ y }}], current_weight); #}
               current_weight += DCT_BASIS[basis_index][x][{{ y }}] * wg_luma[x][{{ y }}];
// {#           current_weight[chunk_index] += DCT_BASIS[basis_index][{{ x }}][{{ y }}] * wg_luma[{{ x }}][{{ y }}];   #}
          $$ endfor
            //}
        }

        let value: f32 = current_weight / QTABLE_MEDIUM_GRAY[basis_index];

//        if abs(value) > 1.0 {

        let pos_dct_output = vec2u((wid.x * 8) + ZZ_INDEX[basis_index].x, (wid.y * 8) + ZZ_INDEX[basis_index].y);

            textureStore(
                tex_y,
                pos_dct_output,
                vec4<f32>(current_weight, 0, 0, 0)
            );
//        }
}
