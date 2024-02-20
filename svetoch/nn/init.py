import svetoch.tensor as ten

# Strictly speaking, this is not an in-place operation, but, unlike
# pytorch, there's no in-place normal() in numpy/cupy. Since init is a
# one-time thing, this will hopefully not be a big deal.
def kaiming_normal_(t):
    assert t.ndim >= 2
    if t.ndim == 2:
        fan_in = t.shape[0]
    else:
        fan_in = 1
        for n in t.shape[1:]:
            fan_in *= n

    std = (2. / fan_in)**.5
    t.array = ten.normal(0., std, t.shape, dtype=t.dtype, device=t.device).array
    return t
