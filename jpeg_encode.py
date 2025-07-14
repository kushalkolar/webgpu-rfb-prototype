from time import perf_counter
import numpy as np
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import imageio.v3 as iio
from skimage.transform import rescale

import wgpu
from utils import Texture, DEVICE, make_bindings
from jpeg_utils import block_size, dct_basis
from pygfx.renderers.wgpu.shader.templating import apply_templating

adapter_ix = 0
# adapter_ix = 2

print(fpl.enumerate_adapters()[adapter_ix].summary)

fpl.select_adapter(fpl.enumerate_adapters()[adapter_ix])

# get example image, add alpha channel of all ones
image = iio.imread("imageio:astronaut.png")#[::4, ::4]

RIF = 6

image = rescale(image, scale=(RIF, RIF, 1), preserve_range=True)
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
    # texture_cbcr.texture.create_view(),
    # chroma_sampler,
    # texture_dct_basis.texture.create_view(),
]

bindings = make_bindings(resources)

#%% visualization

Y = texture_y_dct.read()
CbCr = texture_cbcr.read()

iw = fpl.ImageWidget(
    [Y, CbCr[..., 0], CbCr[..., 1]],
    names=["Y", "Cb", "Cr"],
    figure_shape=(3, 1),
    figure_kwargs={"size": (900, 1800), "controller_ids": None},
    cmap="viridis",
)

iw.show()

templating_values = {
    # "dct_basis": dct_basis
}

with open("./to_luma_881.wgsl", "r") as f:
    shader_src = f.read()

composed_shader = apply_templating(shader_src, **templating_values)
print(composed_shader)
shader_module = DEVICE.create_shader_module(code=composed_shader)

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


def run_shader():
    # encode, submit
    t0 = perf_counter()
    command_encoder = DEVICE.create_command_encoder()

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(
        int((64 * RIF)),
        int((64 * RIF))
        , 1
    )

    compute_pass.end()
    #
    # compute_pass2 = command_encoder.begin_compute_pass()
    # compute_pass2.set_pipeline(pipeline)
    # compute_pass2.set_bind_group(0, bind_group)
    # compute_pass2.dispatch_workgroups(int((64 * RIF)), int((64 * RIF)), 1)
    #
    # compute_pass2.end()

    DEVICE.queue.submit([command_encoder.finish()])
    DEVICE._poll_wait()  # wait for the GPU to finish
    t1 = perf_counter()
    what = f"Computing"
    print(f"{what} took {(t1 - t0) * 1000:0.3f} ms")

    Y = texture_y_dct.read()
    CbCr = texture_cbcr.read()

    iw.set_data([Y, CbCr[..., 0], CbCr[..., 1]])

iw.figure.show_tooltips = True


class GUI(EdgeWindow):
    def __init__(self, figure, title="gui", size=150):
        super().__init__(figure=figure, title=title, size=size, location="right")

    def update(self):
        if imgui.button("Rerun shader"):
            run_shader()


iw.figure.add_gui(GUI(iw.figure))

run_shader()
fpl.loop.run()
