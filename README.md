# Cartographer3D Plugin

![PyPI - License](https://img.shields.io/pypi/l/cartographer3d-plugin)
![PyPI - Version](https://img.shields.io/pypi/v/cartographer3d-plugin)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cartographer3d-plugin)
![GitHub known bugs](https://img.shields.io/github/issues-search/Cartographer3D/cartographer3d-plugin?query=is%3Aissue%20is%3Aopen%20type%3ABug&label=known%20bugs)

The official Cartographer3D plugin.

Documentation can be found at [https://docs.cartographer3d.com/](https://docs.cartographer3d.com/)

## Kalico multiple probes

Support is capability-based: Kalico with the central probe registry from
[Kalico PR #972](https://github.com/KalicoCrew/kalico/pull/972) always registers Cartographer as `cartographer`.
`register_as_probe: true` (default) makes it the default probe; `false` registers it as named-only.
Older Kalico and other firmware retain their existing behavior.

The supported standard mux commands are `PROBE`, `PROBE_ACCURACY`, `QUERY_PROBE`, and `Z_OFFSET_APPLY_PROBE`;
Cartographer extends rather than overrides other probes' handlers. Probe selectors are case-sensitive.

```gcode
PROBE PROBE=cartographer
BED_MESH_CALIBRATE PROBE=cartographer METHOD=scan
```

For bed mesh, omitting `PROBE` follows the default probe. Selected Cartographer uses optimized scanning by default;
other selected or default probes use native automatic meshing. `METHOD=scan` requires Cartographer;
`METHOD=manual` remains native.

Endstop chip names are unchanged: `probe:z_virtual_endstop` with `register_as_probe: true`,
`cartographer_probe:z_virtual_endstop` with `false`. There is no `cartographer:` pin-chip alias.

Native `CALIBRATE_Z` and `PROBE_Z_ACCURACY` cannot use Cartographer; use the existing
`CARTOGRAPHER_SCAN_CALIBRATE` / `CARTOGRAPHER_TOUCH_CALIBRATE` commands instead.
`[nozzle_cleanup]` selecting Cartographer is rejected at configuration time; if Cartographer is the default,
explicitly select another available probe. Cartographer does not provide nozzle-cleaning support.
