"""Torch-only 3-D Hilbert, Morton (Z-order), and hybrid HZ serialization."""

import torch


def z_order_codes(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    """Interleave XYZ bits (X:3i+2, Y:3i+1, Z:3i) into Morton codes."""
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError('Z-order coordinates must have shape (N, 3)')
    if not 1 <= depth <= 21:
        raise ValueError('3-D Z-order depth must be between 1 and 21')
    grid = grid_coord.long()
    if grid.numel() and (torch.any(grid < 0) or torch.any(grid >= 1 << depth)):
        raise ValueError('Z-order coordinates are outside the encoding cube')
    code = torch.zeros(grid.shape[0], dtype=torch.long, device=grid.device)
    for bit in range(depth):
        for axis in range(3):
            code |= ((grid[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return code


@torch.no_grad()
def z_order_sort(xyz: torch.Tensor, grid_size: float = 0.2,
                 transposed: bool = False) -> torch.Tensor:
    """Return stable Morton order; transposed swaps the X/Y grid axes."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Z-order positions must have shape (N, 3)')
    if grid_size <= 0:
        raise ValueError('Z-order grid size must be positive')
    if not torch.isfinite(xyz).all():
        raise ValueError('Z-order positions must be finite')
    if xyz.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    grid = torch.floor((xyz - xyz.amin(dim=0)) / grid_size).long()
    depth = max(1, int(grid.max()).bit_length())
    if transposed:
        grid = grid[:, [1, 0, 2]]
    return torch.argsort(z_order_codes(grid, depth), stable=True)


def hilbert_codes(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    """Encode integer XYZ in [0, 2**depth) as signed-int64 Hilbert codes."""
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError('Hilbert coordinates must have shape (N, 3)')
    if not 1 <= depth <= 21:
        raise ValueError('3-D Hilbert depth must be between 1 and 21')
    x = grid_coord.long().clone()
    if x.numel() and (torch.any(x < 0) or torch.any(x >= 1 << depth)):
        raise ValueError('Hilbert coordinates are outside the encoding cube')

    # Undo excess work (inverse of the Hilbert exchange/invert operations).
    q = 1 << (depth - 1)
    while q > 1:
        p = q - 1
        for axis in range(3):
            invert = (x[:, axis] & q) != 0
            exchange = (x[:, 0] ^ x[:, axis]) & p
            x[:, 0] ^= torch.where(invert, p, exchange)
            x[:, axis] ^= torch.where(invert, 0, exchange)
        q >>= 1

    # Gray encode, then interleave the transpose bits in XYZ order.
    for axis in range(1, 3):
        x[:, axis] ^= x[:, axis - 1]
    correction = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    q = 1 << (depth - 1)
    while q > 1:
        correction ^= torch.where((x[:, 2] & q) != 0, q - 1, 0)
        q >>= 1
    x ^= correction[:, None]
    code = torch.zeros_like(correction)
    for bit in range(depth - 1, -1, -1):
        for axis in range(3):
            code = (code << 1) | ((x[:, axis] >> bit) & 1)
    return code


@torch.no_grad()
def hilbert_sort(xyz: torch.Tensor, grid_size: float = 0.2,
                 transposed: bool = False) -> torch.Tensor:
    """Return a stable Hilbert permutation; transposed swaps the X/Y axes.

    Translate each scene to its minimum XYZ and quantize on an isotropic grid.
    The original point/query indices are retained for ties in a grid cell.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Hilbert positions must have shape (N, 3)')
    if grid_size <= 0:
        raise ValueError('Hilbert grid size must be positive')
    if not torch.isfinite(xyz).all():
        raise ValueError('Hilbert positions must be finite')
    if xyz.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    grid = torch.floor((xyz - xyz.amin(dim=0)) / grid_size).long()
    depth = max(1, int(grid.max()).bit_length())
    if transposed:
        grid = grid[:, [1, 0, 2]]
    return torch.argsort(hilbert_codes(grid, depth), stable=True)


def morton3d(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    """HZ convention: interleave XYZ with X as the least significant axis."""
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError('Morton coordinates must have shape (N, 3)')
    # Reverse the axes to reuse the existing X-most-significant encoder.
    return z_order_codes(grid_coord[:, [2, 1, 0]], depth)


def hilbert3d(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    """Skilling Hilbert transform, matching the supplied HZ reference."""
    return hilbert_codes(grid_coord, depth)


def hz_encode(grid_coord: torch.Tensor, depth: int = 10,
              mode: str = 'hz', coarse_bits: int = 1) -> torch.Tensor:
    """Coarse Morton cells with parity-selected local Hilbert/Morton codes.

    Local Hilbert is selected when ((coarse_X ^ coarse_Y) & 1) is nonzero.
    hz-swap exchanges X/Y before computing both coarse and local codes.
    """
    if mode not in ('hz', 'hz-swap'):
        raise ValueError(f'Unsupported HZ mode: {mode!r}')
    if not 2 <= depth <= 21:
        raise ValueError('HZ depth must be between 2 and 21')
    if not isinstance(coarse_bits, int) or not 1 <= coarse_bits < depth:
        raise ValueError('coarse_bits must satisfy 1 <= coarse_bits < depth')
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise ValueError('grid_coord must have shape (N, 3)')
    coord = grid_coord.long()
    if torch.any(coord < 0) or torch.any(coord >= (1 << depth)):
        raise ValueError(f'Coordinates must be in [0, {1 << depth})')
    if mode == 'hz-swap':
        coord = coord[:, [1, 0, 2]]
    fine_bits = depth - coarse_bits
    high = coord >> fine_bits
    low = coord & ((1 << fine_bits) - 1)
    high_code = morton3d(high, coarse_bits)
    hilbert_code = hilbert3d(low, fine_bits)
    morton_code = morton3d(low, fine_bits)
    use_hilbert = ((high[:, 0] ^ high[:, 1]) & 1).bool()
    local_code = torch.where(use_hilbert, hilbert_code, morton_code)
    return (high_code << (3 * fine_bits)) | local_code


@torch.no_grad()
def hz_sort(xyz: torch.Tensor, grid_size: float = 0.2,
            mode: str = 'hz', coarse_bits: int = 1) -> torch.Tensor:
    """Stable HZ order on a scene-relative grid with adaptive encoding depth.

    Depth covers the scene and leaves at least one fine bit below the coarse
    bits. Direct hz_encode callers can instead specify a fixed depth.
    """
    if mode not in ('hz', 'hz-swap'):
        raise ValueError(f'Unsupported HZ mode: {mode!r}')
    if not isinstance(coarse_bits, int) or not 1 <= coarse_bits <= 20:
        raise ValueError('HZ coarse_bits must be between 1 and 20')
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('HZ positions must have shape (N, 3)')
    if grid_size <= 0:
        raise ValueError('HZ grid size must be positive')
    if not torch.isfinite(xyz).all():
        raise ValueError('HZ positions must be finite')
    if xyz.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    grid = torch.floor((xyz - xyz.amin(dim=0)) / grid_size).long()
    depth = max(2, coarse_bits + 1, int(grid.max()).bit_length())
    return torch.argsort(hz_encode(grid, depth, mode, coarse_bits), stable=True)
