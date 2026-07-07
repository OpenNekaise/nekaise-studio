# Control card LB01 (AHU1) — Norrsken Demo Building (synthetic)

## Operation
AHU1 runs weekdays 06:00-18:00, off outside schedule. Night operation only on frost
protection or extended-operation override.

## Supply air temperature control
Setpoint for GT11 follows an outdoor-compensation curve on GT21:

| GT21 outdoor (degC) | GT11 supply setpoint (degC) |
|---------------------|------------------------------|
| -20                 | 21.0                         |
| 0                   | 19.0                         |
| +10                 | 17.5                         |
| +20                 | 16.0                         |

The controller modulates SV61 (heating valve) 0-100% in sequence before fan speed
reduction. Deadband 0.5 K.

## Fans
GP1 and GP2 are VSD fans, pressure-controlled; minimum speed 30%, maximum 100%.
GP2 tracks GP1 minus 5 percentage points.

## Frost protection
If GT11 falls below 7.0 degC while running, SV61 drives to 100%. If the frost guard
contact GX1 trips (2.0 degC at the coil), the unit stops, dampers close, and alarm
class A "FROST GUARD LB01" is raised. Manual reset required.
