# Telemetry

Cartographer3D collects anonymous usage telemetry to help us understand how the
plugin is used, prioritise development, and catch compatibility issues early.

**Telemetry is opt-in and disabled by default.**

## Opting in

Add to your `printer.cfg`:

```ini
[cartographer]
telemetry: True
```

## What we collect

### Startup event (sent once per boot)

| Field | Example | Purpose |
|-------|---------|---------|
| `anonymous_id` | `a1b2c3d4-...` | Persistent random UUID — not linked to any account or identity |
| `plugin_version` | `1.2.0` | Track adoption of new releases |
| `environment` | `klipper` / `kalico` | Know which runtime to prioritise |
| `python_version` | `3.11.2` | Decide when to drop old Python support |
| `os_platform` | `Linux-6.1.0-aarch64` | Understand host hardware |
| `os_arch` | `aarch64` | Architecture distribution |
| `mcu_version` | `v0.12.0-xxx` | Firmware version spread |
| `config.general` | offsets, speeds, backlash | Whether defaults are appropriate |
| `config.scan` | samples, probe_speed | Feature usage |
| `config.touch` | samples, max_samples, retract_distance | Feature usage |
| `config.bed_mesh` | probe_count, mesh_min/max, zero_reference_position | Typical bed configurations |
| `config.coil` | `has_calibration: true/false` | Temperature compensation adoption |
| `bed_size` | `{"min": [35, 6], "max": [315, 315]}` | Bed size distribution |
| `calibration_summary` | model counts, has_temperature_calibration | Calibration feature usage |

### Calibration events (sent after successful calibration)

**Scan calibration:**

| Field | Example |
|-------|---------|
| `calibration_type` | `"scan"` |
| `method` | `"touch"` or `"manual"` |
| `mcu_version` | `v0.12.0-xxx` |

**Touch calibration:**

| Field | Example |
|-------|---------|
| `calibration_type` | `"touch"` |
| `threshold` | `2500` |
| `speed` | `2` |
| `median_range` | `0.003` |

**Temperature calibration:**

| Field | Example |
|-------|---------|
| `calibration_type` | `"temperature"` |
| `mcu_version` | `v0.12.0-xxx` |

## What we do NOT collect

- No IP addresses are stored server-side
- No file paths or hostnames
- No print data or gcode
- No personal information of any kind
- No raw calibration coefficients or mesh data

## How it works

- Events are sent via HTTPS to [BetterStack Logs](https://betterstack.com/logs)
- All network calls happen in background threads — telemetry never blocks printing
- If sending fails (network down, timeout), the event is silently dropped
- The anonymous ID is stored in `~/.cartographer3d/telemetry_id`

## Opting out

Remove or set to `False`:

```ini
[cartographer]
telemetry: False
```

Delete your anonymous ID:

```bash
rm ~/.cartographer3d/telemetry_id
```
