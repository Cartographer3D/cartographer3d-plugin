from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast, final

from typing_extensions import override

from cartographer.adapters.klipper_like.axis_twist_compensation import KlipperLikeAxisTwistCompensationAdapter
from cartographer.interfaces.printer import AxisTwistCompensation

if TYPE_CHECKING:
    from configfile import ConfigWrapper
    from klippy import Printer


class _ProbeRegistry(Protocol):
    def get_probe(self, name: str) -> object | None: ...


class _ProbeList(Protocol):
    def get_list(self, printer: Printer) -> _ProbeRegistry: ...


class _MeshCompensation(Protocol):
    def get_z_compensation_value(self, pos: list[float], *, probe: object) -> float: ...


class KalicoAxisTwistCompensationAdapter(KlipperLikeAxisTwistCompensationAdapter):
    @override
    def get_z_compensation_value(self, *, x: float, y: float) -> float:
        return self.compensation.get_z_compensation_value([x, y, 0])


@final
class KalicoMeshAxisTwistCompensationAdapter(AxisTwistCompensation):
    def __init__(self, config: ConfigWrapper) -> None:
        printer = config.get_printer()
        probe_list = cast("_ProbeList", import_module("extras.probe").ProbeList)
        # The native manager requires the registry, but scan registration happens later.
        self.registry = probe_list.get_list(printer)
        manager = cast("object", printer.load_object(config, "axis_twist_compensation"))
        self.compensation = cast("_MeshCompensation", manager)

    @override
    def get_z_compensation_value(self, *, x: float, y: float, probe_name: str = "cartographer") -> float:
        probe = self.registry.get_probe(probe_name)
        if probe is None:
            msg = f"Probe '{probe_name}' is not registered in Kalico's probe_list"
            raise RuntimeError(msg)
        # Native lookup applies probe XY offsets itself; pass nozzle coordinates unchanged.
        return self.compensation.get_z_compensation_value([x, y, 0], probe=probe)


def create_axis_twist_compensation(
    config: ConfigWrapper,
) -> KalicoAxisTwistCompensationAdapter | KalicoMeshAxisTwistCompensationAdapter:
    native_class = import_module("extras.axis_twist_compensation").AxisTwistCompensation
    if callable(getattr(native_class, "get_compensation", None)):
        return KalicoMeshAxisTwistCompensationAdapter(config)
    return KalicoAxisTwistCompensationAdapter(config)
