from time import perf_counter
import numpy as np
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import imageio.v3 as iio
from skimage.transform import rescale

import wgpu
from utils import Texture, make_device, make_bindings
from jpeg_utils import block_size, dct_basis
from pygfx.renderers.wgpu.shader.templating import apply_templating

DEVICE = make_device(0)

image = iio.imread("imageio:astronaut.png")#[::4, ::4]

RIF = 6

image = rescale(image, scale=(RIF, RIF, 1), preserve_range=True)
image_rgba = np.zeros((*image.shape[:-1], 4), dtype=np.uint8)
image_rgba[..., :-1] = image
image_rgba[..., -1] = 255


class ComputePass:
    def __init__(self, name, bind_group, pipeline, workgroups):
        self.name = name
        self.bind_group = bind_group
        self.pipeline = pipeline
        self.workgroups = workgroups


class MultiPass:
    def __init__(self):
        self.steps = list()

    def add(
        self,
        name: str,
        resources: list,
        shader_path: str,
        workgroups: tuple[int, int, int],
        templating_values: dict = None,
    ):
        bindings = make_bindings(resources)

        with open(shader_path, "r") as f:
            shader_src = f.read()

        if templating_values is None:
            templating_values = dict()

        composed_shader = apply_templating(shader_src, **templating_values)
        shader_module = DEVICE.create_shader_module(code=composed_shader)

        # create compute pipeline
        pipeline: wgpu.GPUComputePipeline = DEVICE.create_compute_pipeline(
            layout=wgpu.AutoLayoutMode.auto,
            compute={
                "module": shader_module,
                "entry_point": "main",
            },
        )

        # set layout
        layout = pipeline.get_bind_group_layout(0)
        bind_group = DEVICE.create_bind_group(layout=layout, entries=bindings)

        compute_pass = ComputePass(
            name,
            bind_group,
            pipeline,
            workgroups,
        )

        self.steps.append(compute_pass)

    def execute_all(self):
        command_encoder = DEVICE.create_command_encoder()

        for step in self.steps:
            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(step.pipeline)
            compute_pass.set_bind_group(0, step.bind_group)
            compute_pass.dispatch_workgroups(*step.workgroups)

            compute_pass.end()

        DEVICE.queue.submit([command_encoder.finish()])
        DEVICE._poll_wait()  # wait for the GPU to finish
        
    def run_stats(self, n: int = 1_000):
        timings = np.zeros(n)
        
        for i in range(n):
            t0 = perf_counter()
            
            self.execute_all()
            
            timings[i] = perf_counter() - t0
        
        timings *= 1000
        return timings


# %% setup textures
texture_rgba = Texture(image_rgba, label="rgba_input", usage="read")

texture_y = Texture(
    image.shape[:2],
    label="y",
    usage="write",
    format=wgpu.TextureFormat.r32float,
)

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

resources_to_ycbcr = [
    texture_rgba.texture.create_view(),
    texture_y.texture.create_view(),
    # texture_cbcr.texture.create_view(),
    # chroma_sampler,
]

resources_dct = [
    # texture_rgba.texture.create_view(),
    texture_y.texture.create_view(format=wgpu.TextureFormat.r32float),
    texture_y_dct.texture.create_view(),
    # texture_cbcr.texture.create_view(),
    # chroma_sampler,
]

#%% visualization

Y = texture_y_dct.read()
CbCr = texture_cbcr.read()

iw = fpl.ImageWidget(
    [Y, CbCr[..., 0], CbCr[..., 1]],
    names=["Y", "Cb", "Cr"],
    figure_shape=(3, 1),
    figure_kwargs={"size": (900, 1800), "controller_ids": None, "canvas": "qt"},
    cmap="viridis",
)

iw.show()

MULTIPASS: MultiPass = None

#%%

def create_multipass(shader_path, workgroups):
    templating_dct_step = {"dct_basis": dct_basis}

    global MULTIPASS

    MULTIPASS = MultiPass()

    MULTIPASS.add(
        name="ycbcr",
        resources=resources_to_ycbcr,
        shader_path=shader_path,
        workgroups=workgroups,
        templating_values=templating_dct_step,
    )

create_multipass(
    "./luma_dct_8-1-64.wgsl",
    workgroups=(int(64 * RIF), int(64 * RIF), 1)
)


def run_shader():
    global MULTIPASS
    MULTIPASS.execute_all()

    Y = texture_y.read()
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
