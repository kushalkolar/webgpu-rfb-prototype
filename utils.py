from typing import *
import numpy as np

import wgpu


DEVICE: wgpu.GPUDevice = wgpu.utils.get_default_device()


def size_from_shape(shape, dim):
    # copied from pygfx
    # Check if shape matches dimension
    if len(shape) not in (dim, dim + 1):
        raise ValueError(
            f"Can't map shape {shape} on {dim}D tex. Maybe also specify size?"
        )
    # Determine size based on dim and shape
    if dim == 1:
        return shape[0], 1, 1
    elif dim == 2:
        return shape[1], shape[0], 1
    else:  # dim == 3:
        return shape[2], shape[1], shape[0]


class Texture:
    def __init__(
            self,
            data_or_shape: np.ndarray | tuple[int, ...] = None,
            # refers to read or write only in the shader
            usage: Literal["read", "write"] = "read",
            format: wgpu.TextureFormat = wgpu.TextureFormat.rgba8unorm,
            dim: int = 2,
            label: str = "",
    ):
        match usage:
            case "read":
                usage = wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
            case "write":
                usage = wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.STORAGE_BINDING
            case _:
                raise ValueError

        if isinstance(data_or_shape, np.ndarray):
            shape = data_or_shape.shape
        else:
            shape = data_or_shape

        self._texture = DEVICE.create_texture(
            label=label,
            size=size_from_shape(shape, dim),
            usage=usage,
            dimension=getattr(wgpu.TextureDimension, f"d{dim}"),
            format=format,
            mip_level_count=1,
            sample_count=1,
        )

        self._shape = shape

        if format == wgpu.TextureFormat.rgba8unorm:
            self._bytes_per_row = shape[1] * 4
        elif format == wgpu.TextureFormat.r32float:
            # 4 bytes for 32 bit float pixel
            self._bytes_per_row = shape[1] * 4
        elif format == wgpu.TextureFormat.rg32float:
            # 4 bytes for 32 bit float pixel, 2 channels
            self._bytes_per_row = shape[1] * 4 * 2

        if isinstance(data_or_shape, np.ndarray):
            self.write(data_or_shape)

    @property
    def texture(self) -> wgpu.GPUTexture:
        return self._texture

    def write(self, array: np.ndarray):
        """write array to texture so it can be read in the shader"""
        if array.shape != self._shape:
            raise ValueError

        DEVICE.queue.write_texture(
            {
                "texture": self.texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            array,
            {
                "offset": 0,
                "bytes_per_row": self._bytes_per_row,
            },
            self.texture.size,
        )

    def read(self) -> np.ndarray:
        """read data from the GPU that was written in the shader"""
        buffer = DEVICE.queue.read_texture(
            source={
                "texture": self.texture,
                "origin": (0, 0, 0),
                "mip_level": 0,
            },
            data_layout={
                "offset": 0,
                "bytes_per_row": self._bytes_per_row,
            },
            size=self.texture.size,
        ).cast("f")

        return np.frombuffer(buffer, dtype=np.float32).reshape(self._shape)


def make_bindings(resources) -> list[dict]:
    bindings = list()

    for i, r in enumerate(resources):
        bindings.append(
            {
                "binding": i,
                "resource": r,
            }
        )

    return bindings