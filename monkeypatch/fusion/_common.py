# Max possible block size for reduction.
# A reduction dimension of size no more than this can be iterated within one block, which
# eliminates a level of loop and enables more fusion opportunity.
TRITON_MAX_RBLOCK = 256