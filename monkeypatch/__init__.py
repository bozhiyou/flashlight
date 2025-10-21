def disable_flashattention_replacement():
    """A monkey-patch that disables pre-defined sfdp pattern matching for the sake of fusion.
    """
    import torch._inductor.fx_passes.joint_graph
    from torch._inductor.pattern_matcher import init_once_fakemode
    @init_once_fakemode
    def lazy_init_no_sfdp():
        # from .fuse_attention import _sfdp_init
        from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
        from torch._inductor.fx_passes.pad_mm import _pad_mm_init

        _pad_mm_init()
        # _sfdp_init()
        _misc_patterns_init()
    torch._inductor.fx_passes.joint_graph.lazy_init = lazy_init_no_sfdp
