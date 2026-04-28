from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

# Stub Klipper modules not present in the test environment.
for _name in ("mcu", "reactor", "configfile", "klippy", "toolhead", "stepper", "greenlet"):
    if _name not in sys.modules:
        sys.modules[_name] = Mock()

# These imports must come after the sys.modules stubs above.
from cartographer.adapters.kalico.mcu_platform import KalicoMcuPlatform  # noqa: E402
from cartographer.adapters.klipper.mcu_platform import KlipperMcuPlatform  # noqa: E402

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_klipper_platform(mock_mcu: Mock, mocker: MockerFixture) -> KlipperMcuPlatform:
    """Construct a KlipperMcuPlatform wired to *mock_mcu*."""
    mock_config = mocker.MagicMock()
    sys.modules["mcu"].get_printer_mcu.return_value = mock_mcu  # type: ignore[attr-defined]
    return KlipperMcuPlatform(mock_config, "cartographer")


def _make_kalico_platform(mock_mcu: Mock, mocker: MockerFixture) -> KalicoMcuPlatform:
    """Construct a KalicoMcuPlatform wired to *mock_mcu*."""
    mock_config = mocker.MagicMock()
    sys.modules["mcu"].get_printer_mcu.return_value = mock_mcu  # type: ignore[attr-defined]
    return KalicoMcuPlatform(mock_config, "cartographer")


# ===========================================================================
# KlipperMcuPlatform tests
# ===========================================================================


class TestKlipperMcuPlatformRegisterDataResponse:
    def test_register_data_response_uses_serial_when_available(self, mocker: MockerFixture) -> None:
        """When register_serial_response exists, it is called with (handler, fmt)."""
        mock_mcu = mocker.MagicMock()
        # MagicMock has register_serial_response by default, so hasattr returns True.
        platform = _make_klipper_platform(mock_mcu, mocker)

        handler = mocker.Mock()
        platform.register_data_response(handler, "cartographer_data clock=%u", "cartographer_data")

        mock_mcu.register_serial_response.assert_called_once_with(handler, "cartographer_data clock=%u")
        mock_mcu.register_response.assert_not_called()

    def test_register_data_response_falls_back_to_register_response(self, mocker: MockerFixture) -> None:
        """When register_serial_response is absent, register_response is called with (handler, name)."""
        # Use spec=[] so hasattr() returns False for register_serial_response.
        mock_mcu = mocker.MagicMock(spec=[])
        # Manually add register_response so it is accessible.
        mock_mcu.register_response = mocker.Mock()
        platform = _make_klipper_platform(mock_mcu, mocker)

        handler = mocker.Mock()
        platform.register_data_response(handler, "cartographer_data clock=%u", "cartographer_data")

        mock_mcu.register_response.assert_called_once_with(handler, "cartographer_data")


class TestKlipperMcuPlatformIsDisconnected:
    def test_is_disconnected_always_false(self, mocker: MockerFixture) -> None:
        """Stock Klipper never reports a non-critical disconnection."""
        mock_mcu = mocker.MagicMock()
        platform = _make_klipper_platform(mock_mcu, mocker)

        assert platform.is_disconnected() is False


class TestKlipperMcuPlatformLifecycle:
    def test_register_lifecycle_handlers_registers_standard_events(self, mocker: MockerFixture) -> None:
        """identify/connect/shutdown are always registered."""
        mock_mcu = mocker.MagicMock()
        platform = _make_klipper_platform(mock_mcu, mocker)

        on_identify = mocker.Mock()
        on_connect = mocker.Mock()
        on_shutdown = mocker.Mock()

        platform.register_lifecycle_handlers(
            on_identify=on_identify,
            on_connect=on_connect,
            on_shutdown=on_shutdown,
        )

        mock_printer: MagicMock = platform._printer  # pyright: ignore[reportAssignmentType, reportPrivateUsage]
        mock_printer.register_event_handler.assert_any_call("klippy:mcu_identify", on_identify)
        mock_printer.register_event_handler.assert_any_call("klippy:connect", on_connect)
        mock_printer.register_event_handler.assert_any_call("klippy:shutdown", on_shutdown)

    def test_register_lifecycle_handlers_ignores_reconnect(self, mocker: MockerFixture) -> None:
        """on_reconnect and on_disconnect callbacks are silently ignored."""
        mock_mcu = mocker.MagicMock()
        platform = _make_klipper_platform(mock_mcu, mocker)

        on_reconnect = mocker.Mock()
        on_disconnect = mocker.Mock()

        platform.register_lifecycle_handlers(
            on_identify=mocker.Mock(),
            on_connect=mocker.Mock(),
            on_shutdown=mocker.Mock(),
            on_reconnect=on_reconnect,
            on_disconnect=on_disconnect,
        )

        # Neither reconnect nor disconnect should be passed to register_event_handler.
        mock_printer: MagicMock = platform._printer  # pyright: ignore[reportAssignmentType, reportPrivateUsage]
        all_registered_callbacks = [call.args[1] for call in mock_printer.register_event_handler.call_args_list]
        assert on_reconnect not in all_registered_callbacks
        assert on_disconnect not in all_registered_callbacks


class TestKlipperMcuPlatformTriggerDispatch:
    def test_create_trigger_dispatch(self, mocker: MockerFixture) -> None:
        """create_trigger_dispatch delegates to the Klipper TriggerDispatch constructor."""
        mock_mcu = mocker.MagicMock()
        platform = _make_klipper_platform(mock_mcu, mocker)

        td = platform.create_trigger_dispatch()

        # The TriggerDispatch mock from the stubbed mcu module should have been
        # called with the underlying MCU object.
        sys.modules["mcu"].TriggerDispatch.assert_called_with(mock_mcu)  # type: ignore[attr-defined]
        assert td is sys.modules["mcu"].TriggerDispatch.return_value  # type: ignore[attr-defined]


# ===========================================================================
# KalicoMcuPlatform tests
# ===========================================================================


class TestKalicoMcuPlatformRegisterDataResponse:
    def test_register_data_response_uses_register_response(self, mocker: MockerFixture) -> None:
        """Kalico uses register_response with (handler, name)."""
        mock_mcu = mocker.MagicMock()
        platform = _make_kalico_platform(mock_mcu, mocker)

        handler = mocker.Mock()
        platform.register_data_response(handler, "cartographer_data clock=%u", "cartographer_data")

        mock_mcu.register_response.assert_called_once_with(handler, "cartographer_data")


class TestKalicoMcuPlatformIsDisconnected:
    def test_is_disconnected_delegates_to_mcu(self, mocker: MockerFixture) -> None:
        """is_disconnected returns the MCU's non_critical_disconnected attribute."""
        mock_mcu = mocker.MagicMock()
        platform = _make_kalico_platform(mock_mcu, mocker)

        mock_mcu.non_critical_disconnected = True
        assert platform.is_disconnected() is True

        mock_mcu.non_critical_disconnected = False
        assert platform.is_disconnected() is False


class TestKalicoMcuPlatformLifecycle:
    def test_register_lifecycle_handlers_registers_reconnect_events(self, mocker: MockerFixture) -> None:
        """When reconnect/disconnect callbacks are provided, Kalico event names are used."""
        mock_mcu = mocker.MagicMock()
        mock_mcu.get_non_critical_reconnect_event_name.return_value = "mcu_cartographer:reconnect"
        mock_mcu.get_non_critical_disconnect_event_name.return_value = "mcu_cartographer:disconnect"

        platform = _make_kalico_platform(mock_mcu, mocker)

        on_reconnect = mocker.Mock()
        on_disconnect = mocker.Mock()

        platform.register_lifecycle_handlers(
            on_identify=mocker.Mock(),
            on_connect=mocker.Mock(),
            on_shutdown=mocker.Mock(),
            on_reconnect=on_reconnect,
            on_disconnect=on_disconnect,
        )

        mock_printer: MagicMock = platform._printer  # pyright: ignore[reportAssignmentType, reportPrivateUsage]
        mock_printer.register_event_handler.assert_any_call("mcu_cartographer:reconnect", on_reconnect)
        mock_printer.register_event_handler.assert_any_call("mcu_cartographer:disconnect", on_disconnect)

    def test_register_lifecycle_handlers_skips_reconnect_when_none(self, mocker: MockerFixture) -> None:
        """When reconnect/disconnect callbacks are None, event names are not queried."""
        mock_mcu = mocker.MagicMock()
        platform = _make_kalico_platform(mock_mcu, mocker)

        platform.register_lifecycle_handlers(
            on_identify=mocker.Mock(),
            on_connect=mocker.Mock(),
            on_shutdown=mocker.Mock(),
        )

        mock_mcu.get_non_critical_reconnect_event_name.assert_not_called()
        mock_mcu.get_non_critical_disconnect_event_name.assert_not_called()

        # Only the three standard events should be registered.
        mock_printer: MagicMock = platform._printer  # pyright: ignore[reportAssignmentType, reportPrivateUsage]
        assert mock_printer.register_event_handler.call_count == 3
