import torch
import triton
import triton.language as tl


@triton.jit
def _selective_scan_fwd_kernel(
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_ptr,
    batch_stride,
    dim_stride,
    state_stride,
    L,
    dstate: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    u_ptr += pid_b * batch_stride + pid_d * dim_stride
    delta_ptr += pid_b * batch_stride + pid_d * dim_stride

    if HAS_Z:
        z_ptr += pid_b * batch_stride + pid_d * dim_stride
    out_ptr += pid_b * batch_stride + pid_d * dim_stride

    A_start = A_ptr + pid_d * state_stride

    offsets_l = tl.arange(0, BLOCK_SIZE_L)
    mask_l = offsets_l < L

    state = tl.zeros((dstate,), dtype=tl.float32)

    for block_start in range(0, L, BLOCK_SIZE_L):
        l_offs = block_start + offsets_l
        mask = mask_l & (l_offs < L)

        u = tl.load(u_ptr + l_offs, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + l_offs, mask=mask, other=0.0).to(tl.float32)

        delta = tl.sigmoid(delta)

        A = tl.load(A_start + tl.arange(0, dstate))
        dA = tl.exp(delta[:, None] * A[None, :])

        B_start = (
            B_ptr
            + pid_b * batch_stride
            + l_offs[:, None] * state_stride
            + tl.arange(0, dstate)[None, :]
        )
        B_vals = tl.load(B_start, mask=mask[:, None], other=0.0).to(tl.float32)

        C_start = (
            C_ptr
            + pid_b * batch_stride
            + l_offs[:, None] * state_stride
            + tl.arange(0, dstate)[None, :]
        )
        C_vals = tl.load(C_start, mask=mask[:, None], other=0.0).to(tl.float32)

        dB_u = delta[:, None] * B_vals * u[:, None]

        for i in range(BLOCK_SIZE_L):
            if block_start + i < L:
                state = state * dA[i] + dB_u[i]
                y = tl.sum(state * C_vals[i])

                if HAS_D:
                    D_val = tl.load(D_ptr + pid_d)
                    y = y + D_val * u[i]

                if HAS_Z:
                    z_val = tl.load(z_ptr + l_offs[i], mask=mask[i], other=0.0)
                    y = y * tl.sigmoid(z_val) * l_offs[i]

                tl.store(
                    out_ptr + l_offs[i], y.to(out_ptr.dtype.element_ty), mask=mask[i]
                )


def selective_scan_triton(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True
):
    batch, dim, L = u.shape
    dstate = A.shape[1]

    out = torch.empty_like(u)

    BLOCK_SIZE_L = 64
    grid = (batch, dim)

    _selective_scan_fwd_kernel[grid](
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        out,
        None,
        u.stride(0),
        u.stride(1),
        A.stride(0),
        L,
        dstate,
        BLOCK_SIZE_L,
        HAS_D=D is not None,
        HAS_Z=z is not None,
    )

    return out
