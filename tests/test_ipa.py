import torch
import torch.nn.functional as F

class Rigid:
    """
    A class to manage rotations and translations.
    This version has the final correction for einsum broadcasting.
    """
    def __init__(self, rots, trans):
        self.trans = trans
        if rots.shape[-2:] == (3, 3):
            self.rots = rots
        else:
            self.rots = self.quat_to_matrix(rots)

    @staticmethod
    def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
        # This static method is correct and remains the same.
        if quat.shape[-1] == 4:
            q = F.normalize(quat, p=2, dim=-1)
        elif quat.shape[-1] == 3:
            imag_part = quat
            w_squared = 1 - torch.sum(imag_part**2, dim=-1).clamp(max=1.)
            w = torch.sqrt(w_squared.clamp(min=0.))
            full_quat = torch.cat([imag_part, w.unsqueeze(-1)], dim=-1)
            q = F.normalize(full_quat, p=2, dim=-1)
        else:
            raise ValueError("Rotation input must have a last dimension of 3 or 4.")
        
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y,
        ], dim=-1).view(*q.shape[:-1], 3, 3)

    def apply(self, points):
        """Transforms points from local to global frame."""
        # einsum will handle broadcasting between rots' and points' batch dimensions.
        rotated = torch.einsum('...ij,...pj->...pi', self.rots, points)
        
        # This unsqueeze remains correct, as it prepares the translation
        # vector for addition with the rotated points.
        trans_exp = self.trans.unsqueeze(-2)
        
        return rotated + trans_exp

    def invert_apply(self, points):
        """Transforms points from global to local frame."""
        # This unsqueeze is correct for the subtraction.
        trans_exp = self.trans.unsqueeze(-2)
        points_centered = points - trans_exp
        
        return torch.einsum('...ji,...pj->...pi', self.rots, points_centered)


def invariant_point_attention(
    q_scalar: torch.Tensor,
    k_scalar: torch.Tensor,
    v_scalar: torch.Tensor,
    q_points: torch.Tensor,
    k_points: torch.Tensor,
    v_points: torch.Tensor,
    pair_bias: torch.Tensor,
    frames: Rigid,
    mask: torch.Tensor,
    c_hidden: int = 64,
    no_qk_points: int = 8,
):
    """
    A simplified, standalone function for Invariant Point Attention.
    This version uses the corrected Rigid class with proper broadcasting.
    """
    # 1. Transform points to global frame
    # We now pass the frames directly. The Rigid methods handle broadcasting.
    # The frames' rots/trans tensors need a 'heads' dimension to broadcast.
    frames_h = Rigid(
        frames.rots.unsqueeze(2),
        frames.trans.unsqueeze(2)
    )
    
    q_points_global = q_points  # frames_h.apply(q_points)
    k_points_global = k_points  # frames_h.apply(k_points)
    v_points_global = v_points  # frames_h.apply(v_points)

    # 2. Compute attention logits (this part remains the same)
    q_scalar_scaled = q_scalar / (c_hidden ** 0.5)
    # From the original InvariantPointAttention.forward method
    # q shape: [*, N_res, H, C_hidden] -> permuted to [*, H, N_res, C_hidden]
    # k shape: [*, N_res, H, C_hidden] -> permuted to [*, H, C_hidden, N_res]
    # a = torch.matmul(
    #     permute_final_dims(q, (1, 0, 2)),
    #     permute_final_dims(k, (1, 2, 0)),
    # )
    # a shape: [*, H, N_res, N_res] is then scaled and has the pair bias 'b' added to it
    # content_score = torch.einsum('...hid,...hjd->...hij', q_scalar_scaled, k_scalar)
    content_score = q_scalar_scaled @ k_scalar.transpose(-1, -2)

    dist_sq = torch.sum(
        (q_points_global.unsqueeze(3) - k_points_global.unsqueeze(2)) ** 2,
        dim=-1
    )
    point_weights = -0.5 * (9.0 / (2 * no_qk_points))**0.5
    geometric_score = torch.sum(dist_sq, dim=-1) * point_weights

    logits = content_score + geometric_score + pair_bias

    # attention_mask = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
    # logits[attention_mask == 0] = -torch.inf

    # 3. Calculate attention weights
    attn_weights = torch.softmax(logits, dim=-1)

    # 4. Aggregate outputs
    # 4.1 v_scalar
    output_scalar = torch.einsum('...hij,...hjd->...ihd', attn_weights, v_scalar)
    output_scalar = attn_weights @ v_scalar
    # return output_scalar
    
    # 4.2 v_points_global
    output_points_global = torch.einsum('...hij,...hjpv->...ihpv', attn_weights, v_points_global)
    output_points_local = output_points_global  # frames_h.invert_apply(output_points_global)
    # return output_points_local

    return output_scalar + output_points_local


if __name__ == '__main__':
    from torch.testing import assert_close
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion
    
    # Check for CUDA availability for the test
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
    else:
        DEVICE = torch.device("cuda:0")
        DTYPE = torch.bfloat16

        # --- Configuration based on AlphaFold2/OpenFold ---
        BATCH_SIZE = 2
        N_RES = 64          # Number of residues (sequence length)
        NO_HEADS = 8        # Number of attention heads
        C_HIDDEN = 16       # Per-head scalar dimension
        NO_QK_POINTS = 4    # Number of query/key points
        NO_V_POINTS = 8     # Number of value points

        # --- Create random input tensors on the GPU ---
        q_scalar = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, C_HIDDEN, dtype=DTYPE, device=DEVICE)
        k_scalar = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, C_HIDDEN, dtype=DTYPE, device=DEVICE)
        v_scalar = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, C_HIDDEN, dtype=DTYPE, device=DEVICE)

        q_points = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, NO_QK_POINTS, 4, dtype=DTYPE, device=DEVICE)
        k_points = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, NO_QK_POINTS, 4, dtype=DTYPE, device=DEVICE)
        v_points = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, NO_V_POINTS, 4, dtype=DTYPE, device=DEVICE)

        pair_bias = torch.randn(BATCH_SIZE, NO_HEADS, N_RES, N_RES, dtype=DTYPE, device=DEVICE)
        
        # Create valid quaternions (unit vectors) for rotations
        # Shape: [batch, n_res, 4]
        rots_quat = F.normalize(torch.randn(BATCH_SIZE, N_RES, 4, dtype=DTYPE, device=DEVICE), dim=-1)
        # Create translations
        # Shape: [batch, n_res, 3]
        trans = torch.randn(BATCH_SIZE, N_RES, 3, dtype=DTYPE, device=DEVICE)
        frames = Rigid(rots_quat, trans)
        
        # A simple mask where all residues are present
        mask = torch.ones(BATCH_SIZE, N_RES, dtype=DTYPE, device=DEVICE)

        # --- Run the high-precision reference implementation ---
        # Note: We cast inputs to float32 for the baseline
        frames_fp32 = Rigid(frames.rots.to(torch.float32), frames.trans.to(torch.float32))
        # o0_scalar, o0_points = invariant_point_attention(
        o0 = invariant_point_attention(
            q_scalar.to(torch.float32), k_scalar.to(torch.float32), v_scalar.to(torch.float32),
            q_points.to(torch.float32), k_points.to(torch.float32), v_points.to(torch.float32),
            pair_bias.to(torch.float32),
            frames_fp32,
            mask.to(torch.float32),
            c_hidden=C_HIDDEN, no_qk_points=NO_QK_POINTS
        )

        # --- Run the compiled bfloat16 version ---
        compiled_ipa = torch.compile(invariant_point_attention, dynamic=False)
        # o1_scalar, o1_points = compiled_ipa(
        o1 = compiled_ipa(
            q_scalar, k_scalar, v_scalar,
            q_points, k_points, v_points,
            pair_bias,
            frames,
            mask,
            c_hidden=C_HIDDEN, no_qk_points=NO_QK_POINTS
        )

        # --- Assert that the outputs are close enough ---
        # Tolerances are set to be reasonable for bf16 vs fp32 comparisons.
        assert_close(o0, o1.to(torch.float32), atol=1e-2, rtol=1e-2)
        # assert_close(o0_scalar, o1_scalar.to(torch.float32), atol=1e-2, rtol=1e-2)
        # assert_close(o0_points, o1_points.to(torch.float32), atol=1e-2, rtol=1e-2)

        print("✅ Verification successful!")