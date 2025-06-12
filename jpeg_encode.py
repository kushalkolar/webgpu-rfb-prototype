import numpy as np
import fastplotlib as fpl
import imageio.v3 as iio

import wgpu
from utils import Texture, DEVICE, make_bindings
from jpeg_utils import block_size, dct_basis


# get example image, add alpha channel of all ones
image = iio.imread("imageio:astronaut.png")#[::4, ::4]
image_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
image_rgba[..., :-1] = image
image_rgba[..., -1] = 255

#%% setup textures
texture_rgba = Texture(image_rgba, label="rgba_input", usage="read")
texture_y_dct = Texture(
    image.shape[:2],
    label="y",
    usage="write",
    format=wgpu.TextureFormat.r32float,
)

texture_cbcr = Texture(
    image[::2, ::2, :2].shape,
    label="cbcr",
    usage="write",
    format=wgpu.TextureFormat.rg32float,
)

# sample we will use to generate the CbCr channels
chroma_sampler = DEVICE.create_sampler(
    # I don't think min filtering actually occurs for chroma sampling
    # since we are always sampling from the center of 4 pixels to create 1 subsampled new CbCr pixel
    min_filter=wgpu.FilterMode.linear,
    mag_filter=wgpu.FilterMode.linear,
)

texture_dct_basis = Texture(
    dct_basis,
    dim=3,
    label="dct_basis",
    usage="read",
    format=wgpu.TextureFormat.r32float,
)

resources = [
    texture_rgba.texture.create_view(),
    texture_y_dct.texture.create_view(),
    texture_cbcr.texture.create_view(),
    chroma_sampler,
    texture_dct_basis.texture.create_view(),
]

bindings = make_bindings(resources)

#%% visualization

Y = texture_y_dct.read()
CbCr = texture_cbcr.read()

iw = fpl.ImageWidget(
    [Y, CbCr[..., 0], CbCr[..., 1]],
    names=["Y", "Cb", "Cr"],
    figure_shape=(1, 3),
    figure_kwargs={"size": (1000, 400), "controller_ids": None},
    cmap="viridis",
)

iw.show()

#%% run shader continously and update the imagewidget

def run_shader():
    with open("./jpeg_encode.wgsl", "r") as f:
        shader_src = f.read()

    shader_module = DEVICE.create_shader_module(code=shader_src)

    workgroup_size_constants = {
        "group_size_x": block_size,
        "group_size_y": block_size,
    }

    # create compute pipeline
    pipeline: wgpu.GPUComputePipeline = DEVICE.create_compute_pipeline(
        layout=wgpu.AutoLayoutMode.auto,
        compute={
            "module": shader_module,
            "entry_point": "main",
            "constants": workgroup_size_constants,
        },
    )


    # set layout
    layout = pipeline.get_bind_group_layout(0)
    bind_group = DEVICE.create_bind_group(layout=layout, entries=bindings)

    # make sure we have enough workgroups to process all blocks of the input image
    # each workgroup will process the pixels within one 8x8 block
    # the blocks are non-overlapping
    workgroups = np.ceil(np.asarray(image.shape[:2]) / block_size).astype(int)

    # encode, submit
    command_encoder = DEVICE.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(*workgroups, 1)
    compute_pass.end()
    DEVICE.queue.submit([command_encoder.finish()])

    Y = texture_y_dct.read()
    CbCr = texture_cbcr.read()

    iw.set_data([Y, CbCr[..., 0], CbCr[..., 1]])

iw.figure.show_tooltips = True
run_shader()
fpl.loop.run()
