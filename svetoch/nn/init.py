import svetoch

def kaiming_normal_(t):
    assert t.ndim >= 2
    if t.ndim == 2:
        fan_in = t.shape[0]
    else:
        fan_in = 1
        for n in t.shape[1:]:
            fan_in *= n

    std = (2. / fan_in)**.5
    t.v = svetoch.normal(0., std, t.shape, dtype=t.dtype, device=t.device).v
    return t
