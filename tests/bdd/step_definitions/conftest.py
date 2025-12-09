from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from pytest_bdd import given, parsers, then

from cartographer.interfaces.configuration import Configuration, ScanModelConfiguration, TouchModelConfiguration
from cartographer.interfaces.printer import Position, Sample
from tests.bdd.helpers.context import Context

if TYPE_CHECKING:
    from pytest import LogCaptureFixture
    from pytest_bdd.parser import Feature, Scenario

    from cartographer.probe.scan_mode import ScanMode
    from cartographer.probe.touch_mode import TouchMode
    from cartographer.stream import Session
    from tests.mocks.params import MockParams


def sample(frequency: float):
    return Sample(
        raw_count=0,
        frequency=1 / frequency,
        time=0,
        position=Position(0, 0, frequency),
        temperature=0,
    )


@pytest.fixture
def context() -> Context:
    return Context()


@then("it should throw an error")
def then_it_should_throw_error(context: Context) -> None:
    assert context.error is not None, "Expected an error, but none was thrown."


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_bdd_after_scenario(request: pytest.FixtureRequest, feature: Feature, scenario: Scenario):
    del feature  # Unused, but required by the hook signature
    yield
    context: Context = request.getfixturevalue("context")
    steps = [s.name for s in scenario.steps]
    if context.error is not None and "it should throw an error" not in steps:
        pytest.fail(f"Unhandled error: {context.error}")


@given("macro parameters:")
def given_parameters(datatable: list[list[str]], params: MockParams):
    params.params = {key: value for key, value in datatable}


@then(parsers.parse('it should log "{output}"'))
def then_log_result(caplog: LogCaptureFixture, output: str):
    assert output in caplog.text


@given("a probe")
def given_probe() -> None:
    pass


@given("the probe has scan calibrated")
def given_scan_calibrated(scan: ScanMode, config: Configuration, session: Session[Sample]):
    config.save_scan_model(
        ScanModelConfiguration(
            name="default",
            coefficients=[0.3] * 9,
            domain=(0.1, 5.5),
            z_offset=0.0,
            reference_temperature=30,
        )
    )
    scan.load_model("default")
    session.get_items = lambda: [sample(frequency=2) for _ in range(11)]


@given(parsers.parse("the probe has scan z-offset {offset:g}"))
def given_scan_offset(scan: ScanMode, config: Configuration, offset: float):
    config.save_scan_model(replace(config.scan.models["default"], z_offset=offset))
    scan.load_model("default")


@given("the probe has touch calibrated")
def given_touch_calibrated(touch: TouchMode, config: Configuration):
    config.touch.models["default"] = TouchModelConfiguration(
        name="default",
        threshold=1000,
        speed=3,
        z_offset=0.0,
    )
    touch.load_model("default")


@given(parsers.parse("the probe has touch z-offset {offset:g}"))
def given_touch_offset(touch: TouchMode, config: Configuration, offset: float):
    config.save_touch_model(replace(config.touch.models["default"], z_offset=offset))
    touch.load_model("default")
