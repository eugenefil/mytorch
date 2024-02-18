from collections import namedtuple


devices = {}
Ops = None


class Device:
    def __init__(self, type):
        dev = devices.get(type)
        if dev is None:
            raise ValueError(f"unknown device type '{type}'")
        self.type = type
        self.backend = dev["backend"]
        self.ops = dev["ops"]

    def __repr__(self):
        return self.type

    __str__ = __repr__

    def __eq__(self, other):
        return self.type == other.type


def from_data(data):
    for type, dev in devices.items():
        probe = dev["probe_data"]
        if probe and probe(data):
            return Device(type)
    return Device("cpu")


def from_device(device):
    if device is None:
        return Device("cpu")
    elif isinstance(device, str):
        return Device(device)
    assert isinstance(device, Device)
    return device


def register_device(type, backend, ops, probe_data=None):
    if type == "cpu":
        # cpu ops provide the full reference list of ops and also default
        # values for those ops, so cpu device must registered first
        global Ops
        # namedtuple provides dot access to its fields and default values
        Ops = namedtuple("Ops", ops.keys(), defaults=ops.values())
    ops = Ops(**ops) # missing slots are filled with defaults (i.e. cpu ops)

    devices[type] = {
        "backend": backend,
        "ops": ops,
        "probe_data": probe_data
    }
