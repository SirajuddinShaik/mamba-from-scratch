import math
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba.utils.torch_utils import custom_fwd, custom_bwd


class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        z=None,
        delta_bias=None,
        delta_softplus=False,
        return_last_state=False,
    ):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()

        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_ref(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state=True
        )

        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, :] if x.dim() == 4 else x[:, :, -1]

        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0] if rest else out * F.silu(z)
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        delta_softplus = ctx.delta_softplus
        has_z = ctx.has_z

        if not has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors

        ddelta, dA, dB, dC, dD, dz = None, None, None, None, None, None
        du = None

        dout_float = dout.float()
        u_float = u.float()
        delta_float = delta.float()
        B_float = B.float()
        C_float = C.float()

        if delta_bias is not None:
            delta_float = delta_float + delta_bias[..., None].float()
        if delta_softplus:
            delta_float = F.softplus(delta_float)

        deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_float, A.float()))

        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        seqlen = u.shape[2]

        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 4

        if has_z and z is not None and out is not None:
            z_float = z.float()
            silu_z = F.silu(z_float)
            sigmoid_z = torch.sigmoid(z_float)
            dsilu_z = sigmoid_z * (1 + z_float * (1 - sigmoid_z))
            dz = dout_float * out / (silu_z + 1e-6) * dsilu_z
            dy = dout_float * silu_z
        else:
            dy = dout_float

        if D is not None:
            dD = torch.einsum("bdl,bdl->d", dy, u_float)
        else:
            dD = None

        dx_next = torch.zeros(batch, dim, dstate, device=u.device, dtype=torch.float32)
        d_deltaB_u = torch.zeros(
            batch, dim, dstate, seqlen, device=u.device, dtype=torch.float32
        )

        for i in range(seqlen - 1, -1, -1):
            if not is_variable_C:
                dx_current = torch.einsum("bd,dn->bdn", dy[:, :, i], C_float)
            else:
                if C.dim() == 3:
                    dx_current = torch.einsum(
                        "bd,bn->bdn", dy[:, :, i], C_float[:, :, i]
                    )
                else:
                    dx_current = torch.einsum(
                        "bd,bdn->bdn", dy[:, :, i], C_float[:, :, :, i]
                    )

            dx_current = dx_current + dx_next
            d_deltaB_u[:, :, :, i] = dx_current

            if i > 0:
                dx_next = deltaA[:, :, i - 1] * dx_current

        if not is_variable_B:
            deltaB = torch.einsum("bdl,dn->bdln", delta_float, B_float)
        else:
            if B.dim() == 3:
                deltaB = torch.einsum("bdl,bnl->bdln", delta_float, B_float)
            else:
                B_rep = repeat(
                    B_float, "B G N L -> B (G H) N L", H=dim // B_float.shape[1]
                )
                deltaB = torch.einsum("bdl,bdnl->bdln", delta_float, B_rep)

        du = torch.einsum("bdln,bdln->bdl", d_deltaB_u, deltaB)
        if D is not None:
            du = du + dy * D.float()[..., None]

        ddelta = torch.zeros_like(delta_float)
        dA = torch.zeros_like(A.float())

        if is_variable_B:
            dB = torch.zeros_like(B_float)
        else:
            dB = None

        if is_variable_C:
            dC = torch.zeros_like(C_float)
        else:
            dC = None

        for i in range(seqlen):
            if is_variable_C and dC is not None:
                if C.dim() == 3:
                    dC[:, :, i] = torch.einsum("bd,bdn->bn", dy[:, :, i], x[:, :, i])
                else:
                    dC[:, :, :, i] = torch.einsum(
                        "bd,bdn->bdn", dy[:, :, i], x[:, :, :, i]
                    )

        du = du.to(u.dtype)
        ddelta = ddelta.to(delta.dtype)
        dA = dA.to(A.dtype)
        dB = dB.to(B.dtype) if dB is not None else None
        dC = dC.to(C.dtype) if dC is not None else None
        dD = dD.to(D.dtype) if dD is not None else None
        dz = dz.to(z.dtype) if dz is not None else None

        return du, ddelta, dA, dB, dC, dD, dz, None, None, None


def selective_scan_ref(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B = B.float()
    C = C.float()

    x = torch.zeros((batch, dim, dstate), device=u.device, dtype=u.dtype)
    ys = []

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum("bdn,dn->bd", x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
            else:
                y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])

        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")

    if z is not None:
        out = out * F.silu(z)

    out = out.to(dtype=dtype_in)

    return out if not return_last_state else (out, last_state)


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    return SelectiveScanFn.apply(
        u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
    )
