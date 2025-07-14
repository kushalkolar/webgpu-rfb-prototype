// This is the slowest and takes ~10.5ms on my AMD GPU


@group(0) @binding(0)
var tex_rgba: texture_2d<f32>;

@group(0) @binding(1)
var tex_y_dct: texture_storage_2d<r32float, write>;

// weights to convert from RGB -> YCbCr
const LUMA_WEIGHTS = vec4f(0.299, 0.587, 0.114, 0);
const CB_WEIGHTS = vec4f(-0.1687, -0.3313, 0.5, 0);
const CR_WEIGHTS = vec4f(0.5, -0.4187, -0.0813, 0);


@compute @workgroup_size(1, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>
    ) {
        let start: vec2u = wid.xy * vec2u(8, 8);
        var pixel_pos = vec2u(0, 0);

        $$ for x in range(8)
            $$ for y in range(8)
                pixel_pos = vec2u(start.x + {{ x }}, start.y + {{ y }});

                textureStore(
                tex_y_dct,
                pixel_pos,
                vec4f((dot(textureLoad(tex_rgba, pixel_pos, 0), LUMA_WEIGHTS) - 0.5) * 255, 0, 0, 0)
                );
            $$ endfor
        $$ endfor
}
