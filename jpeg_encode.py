from time import perf_counter
import numpy as np
import fastplotlib as fpl
from fastplotlib.ui import EdgeWindow
from imgui_bundle import imgui
import imageio.v3 as iio
from skimage.transform import rescale

import wgpu
from utils import Texture, StorageBuffer, make_device, make_bindings
from jpeg_utils import block_size, dct_basis
from pygfx.renderers.wgpu.shader.templating import apply_templating
from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object
from fastplotlib.graphics.features import TextureArray

DEVICE = make_device(1)
print(DEVICE.adapter_info)


image = iio.imread("imageio:astronaut.png")
#
# fig = fpl.Figure(size=(1000, 1000))
#
# fig[0, 0].add_image(image)
#
# fig.renderer.pixel_ratio = 1.0
#
# fig.show()
#
# image_rgba = fig.export_numpy(rgb=False)

RIF = 4

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
        # populated by execute_all when timestamp_query is enabled on the device
        self.timings_ms: dict[str, float] = {}
        self._timing_enabled = "timestamp-query" in DEVICE.features
        self._query_set = None
        self._resolve_buf = None
        # (buffer, zero_bytes) pairs zeroed via queue.write_buffer at the start of every execute_all
        self._frame_zero_buffers: list[tuple[wgpu.GPUBuffer, bytes]] = []

    def zero_each_frame(self, buffer: wgpu.GPUBuffer, n_bytes: int):
        """Register `buffer` to be zeroed (first `n_bytes` bytes) before every dispatch."""
        self._frame_zero_buffers.append(
            (buffer, np.zeros(n_bytes, dtype=np.uint8).tobytes())
        )

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
        # step count changed; previously sized timing resources are stale
        self._query_set = None
        self._resolve_buf = None

    def _check_timing_resources(self):
        if self._query_set is not None:
            return
        n = len(self.steps)
        # one timestamp at begin, one at end of each compute pass
        self._query_set = DEVICE.create_query_set(
            label="multipass-timing",
            type=wgpu.QueryType.timestamp,
            count=2 * n,
        )
        self._resolve_buf = DEVICE.create_buffer(
            label="multipass-timing-resolve",
            size=2 * n * 8,  # u64 nanoseconds per slot
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
        )

    def execute_all(self):
        if self._timing_enabled:
            self._check_timing_resources()

        # queue.write_buffer is ordered before the subsequent submit on the same queue,
        # so registered buffers are zeroed before any compute pass reads them
        for buf, zeros in self._frame_zero_buffers:
            DEVICE.queue.write_buffer(buf, 0, zeros)

        command_encoder = DEVICE.create_command_encoder()

        for i, step in enumerate(self.steps):
            ts_writes = None
            if self._timing_enabled:
                ts_writes = {
                    "query_set": self._query_set,
                    "beginning_of_pass_write_index": 2 * i,
                    "end_of_pass_write_index": 2 * i + 1,
                }
            compute_pass = command_encoder.begin_compute_pass(timestamp_writes=ts_writes)
            compute_pass.set_pipeline(step.pipeline)
            compute_pass.set_bind_group(0, step.bind_group)
            compute_pass.dispatch_workgroups(*step.workgroups)

            compute_pass.end()

        if self._timing_enabled:
            command_encoder.resolve_query_set(
                self._query_set, 0, 2 * len(self.steps), self._resolve_buf, 0,
            )

        DEVICE.queue.submit([command_encoder.finish()])
        # no _poll_wait: the GPU runs async, sequential submits serialize on the queue,
        # and read_timings()/read_buffer() do their own sync when called.

    def read_timings(self):
        """Download the resolved timestamps from the most recent execute_all and
        populate self.timings_ms. Pulled out of execute_all because the staging-buffer
        roundtrip adds milliseconds and otherwise dominates the wall-time measurement.
        """
        if not self._timing_enabled:
            return
        raw = DEVICE.queue.read_buffer(self._resolve_buf)
        ts = np.frombuffer(raw, dtype=np.uint64)
        self.timings_ms = {
            step.name: (int(ts[2 * i + 1]) - int(ts[2 * i])) / 1e6
            for i, step in enumerate(self.steps)
        }

    def run_stats(self, n: int = 1_000):
        timings = np.zeros(n)
        per_pass = {step.name: np.zeros(n) for step in self.steps}

        for i in range(n):
            t0 = perf_counter()

            self.execute_all()
            # execute_all no longer blocks, so add an explicit fence wait here to capture
            # actual completion in the wall-time measurement (the interactive render path
            # doesn't need this; fpl's frame-driven submits pipeline naturally).
            DEVICE._poll_wait()

            timings[i] = perf_counter() - t0
            # read outside the timed window so the staging-buffer roundtrip
            # doesn't contaminate the wall time we're trying to measure
            self.read_timings()
            for name, ms in self.timings_ms.items():
                per_pass[name][i] = ms

        timings *= 1000
        return timings, per_pass


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


# dct_buffer = StorageBuffer(dct_basis)

n_blocks_x = image_rgba.shape[1] // 8
n_blocks_y = image_rgba.shape[0] // 8
n_image_blocks = n_blocks_x * n_blocks_y

# scan tile params: W threads per workgroup, K elements per thread.
# Two-level scan supports n_image_blocks up to TILE * TILE = 262144 (4096x4096 luma).
W = 64
K = 8
TILE = W * K
n_blocks_padded = ((n_image_blocks + TILE - 1) // TILE) * TILE
n_tiles = n_blocks_padded // TILE

# per_block_rle: per-block scratch; block b's packed pairs at [b*64, b*64 + n_non_zeros[b])
# n_non_zeros / block_starts: padded to a multiple of TILE so the scan kernels have full tiles
# wg_totals: sized TILE so scan_totals operates as a single full tile
# zle_packed: contiguous output; host downloads only the populated prefix
per_block_rle = StorageBuffer(np.zeros(n_image_blocks * 64, dtype=np.uint32), label="per_block_rle")
n_non_zeros   = StorageBuffer(np.zeros(n_blocks_padded,     dtype=np.uint32), label="n_non_zeros")
block_starts  = StorageBuffer(np.zeros(n_blocks_padded,     dtype=np.uint32), label="block_starts")
wg_totals     = StorageBuffer(np.zeros(TILE,                dtype=np.uint32), label="wg_totals")
zle_packed    = StorageBuffer(np.zeros(n_image_blocks * 64, dtype=np.uint32), label="zle_packed")

resources_dct = [texture_rgba, per_block_rle, n_non_zeros]

# Reconstruction output - GPU-only TextureArray shared with pygfx so the renderer samples
# the same wgpu texture our compute shader writes into.
texture_recon_jpeg_array = TextureArray(
    data=np.zeros((image_rgba.shape[0], image_rgba.shape[1], 4), dtype=np.uint8),
    cpu_buffer=False,
    usage=(
        wgpu.TextureUsage.STORAGE_BINDING
        | wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.COPY_SRC
    ),
)
texture_recon_jpeg = ensure_wgpu_object(texture_recon_jpeg_array.buffer[0, 0])

#%% visualization

# Y = texture_y_dct.read()
# CbCr = texture_cbcr.read()
#
# iw = fpl.ImageWidget(
#     [Y, CbCr[..., 0], CbCr[..., 1]],
#     names=["Y", "Cb", "Cr"],
#     figure_shape=(3, 1),
#     figure_kwargs={"size": (900, 1800), "controller_ids": None, "canvas": "qt"},
#     cmap="viridis",
# )
#
# iw.show()

MULTIPASS: MultiPass = None

#%%

def create_multipass():
    global MULTIPASS

    MULTIPASS = MultiPass()

    MULTIPASS.add(
        name="dct_zle",
        resources=resources_dct,
        shader_path="./aan_dct.wgsl",
        workgroups=(n_blocks_x, n_blocks_y, 1),
    )

    MULTIPASS.add(
        name="scan_local",
        resources=[n_non_zeros, block_starts, wg_totals],
        shader_path="./scan_local.wgsl",
        workgroups=(n_tiles, 1, 1),
    )

    MULTIPASS.add(
        name="scan_totals",
        resources=[wg_totals],
        shader_path="./scan_totals.wgsl",
        workgroups=(1, 1, 1),
    )

    MULTIPASS.add(
        name="scan_add",
        resources=[wg_totals, block_starts],
        shader_path="./scan_add.wgsl",
        workgroups=(n_tiles, 1, 1),
    )

    MULTIPASS.add(
        name="compact",
        resources=[per_block_rle, n_non_zeros, block_starts, zle_packed],
        shader_path="./compact.wgsl",
        workgroups=(n_blocks_x, n_blocks_y, 1),
    )

    MULTIPASS.add(
        name="reconstruct",
        resources=[
            n_non_zeros,
            block_starts,
            zle_packed,
            texture_recon_jpeg.create_view(),
        ],
        shader_path="./reconstruct.wgsl",
        # 4 blocks per workgroup (2x2 spatial), so dispatch is halved on each axis
        workgroups=(n_blocks_x // 2, n_blocks_y // 2, 1),
    )


create_multipass()


figure = fpl.Figure(size=(900, 900))
figure[0, 0].add_image(texture_recon_jpeg_array, vmin=0, vmax=255)


def animate(_figure):
    MULTIPASS.execute_all()


figure.add_animations(animate)
figure.show()
fpl.loop.run()

stats, per_pass = MULTIPASS.run_stats(n=100)
# print("nvidia RTX 3080, 8k")
# print("nvidia RTX 3080, 4k")
print(f"\tmean: {stats.mean():0.3f} ms")
print(f"\tmedian: {np.median(stats):0.3f} ms")

if per_pass:
    print("per-pass GPU time (median ms):")
    for name, samples in per_pass.items():
        print(f"\t{name:>12s}: {np.median(samples):0.3f} ms")

# Only the first n_image_blocks slots are meaningful; the rest are padding for the scan tiles.
n_non_zeros_arr = np.frombuffer(
    DEVICE.queue.read_buffer(n_non_zeros.buffer, 0, n_image_blocks * 4),
    dtype=np.uint32,
)
block_starts_arr = np.frombuffer(
    DEVICE.queue.read_buffer(block_starts.buffer, 0, n_image_blocks * 4),
    dtype=np.uint32,
)
# block_starts is the exclusive prefix sum of n_non_zeros, so the last block's end is the total.
total = int(block_starts_arr[-1] + n_non_zeros_arr[-1])

# Only download the populated prefix of zle_packed. Each u32 slot packs:
#   first  16 bits = preceding zero count
#   second 16 bits = quantized coefficient value (signed)
# View as a (N, 2) u16 array, reinterpret column 1 as i16 to recover the signed value.
zle_raw = DEVICE.queue.read_buffer(zle_packed.buffer, 0, total * 4)
shorts = np.frombuffer(zle_raw, dtype=np.uint16).reshape(-1, 2)
zle_zeros = shorts[:, 0]
zle_values = shorts[:, 1].view(np.int16)


def block_zle(b: int) -> tuple[np.ndarray, np.ndarray]:
    s, n = int(block_starts_arr[b]), int(n_non_zeros_arr[b])
    return zle_zeros[s:s + n], zle_values[s:s + n]


print(f"total ZLE entries: {total}  ({total * 4} bytes downloaded)")
print(f"block 0: n={n_non_zeros_arr[0]} start={block_starts_arr[0]} pairs={block_zle(0)}")
