"""
mace_onegpu: exact, memory-lean evaluation of two-layer MACE models (e.g. MACE-MPA-0) for millions of atoms on one GPU.

Modules (import them explicitly; nothing is imported here, so that importing the package needs neither a GPU nor
Triton, and so that the MACE monkeypatch in graphed_three_pass is only applied when it is actually used):

    mliap_threepass      ThreePassMLIAP: LAMMPS ML-IAP unified wrapper ("pair_style mliap unified <file.pt> 0")
    receiver_chunk_mace  receiver-chunked three-pass algorithm (reference implementation, model loading)
    fast_layers          exact node-level speed-ups (species-block skip_tp, receiver-only node ops)
    spline_w             tabulated radial tensor-product weights (Triton)
    conv_spline          fused tabulated-radial convolution (Triton, generated kernels)
    graphed_three_pass   CUDA-graph version + make_capture_safe()
    fused_conv           pickle-safe fused cuEquivariance convolution (needed to unpickle the .pt inside LAMMPS)
    fused_edge_conv      path tables from cuEquivariance (+ fully fused edge convolution, experimental)
    spline_radial        fp64 sampling of the radial networks (used by conv_spline.build_tables)
    fused_radial, radial_3xtf32, ace_layer1, ase_gradient_probe   optional variants / validation helpers

The LAMMPS model file (.pt) pickles classes by module path (mace_onegpu.mliap_threepass.ThreePassMLIAP,
mace_onegpu.fused_conv.FusedConvTP), so the repository root must be on PYTHONPATH of the Python interpreter embedded
in LAMMPS.
"""

__version__ = "0.1.0"
