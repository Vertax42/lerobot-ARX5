# Stations

One YAML per physical bench. A station says what hardware is bolted down — arm
identities, mounting geometry, home/start poses, camera serials — and nothing
about how a given run drives it.

```
stations/
  bi_flexiv_rizon4_rt/forward-05.yaml
  bi_elite_cs66_rt/diagonal-08.yaml
```

Pick one with `bi_mount_type`, in a recipe or on the CLI:

```yaml
robot:
  type: bi_flexiv_rizon4_rt
  bi_mount_type: forward-05
```

```bash
lerobot-teleoperate --robot.type=bi_flexiv_rizon4_rt --robot.bi_mount_type=forward-05
```

## Stations vs. recipes

They answer different questions and change on different schedules.

| | `stations/` | `recipes/` |
|---|---|---|
| Answers | what is bolted to this bench | how this run should behave |
| Holds | arm SNs/IPs, mount geometry, home/start poses, camera serials | control mode, servo gains, guards, gripper force, dataset fields |
| Changes when | hardware is swapped or re-mounted | you tune a task or start a new campaign |
| Reused by | every recipe targeting that bench | one task, one bench |

A tuning value in a station file is a bug: it silently applies to every task run
on that bench. A camera serial in a recipe is also a bug, though a louder one —
it now takes effect (see precedence below), so a stale copy overrides the real
station.

## Precedence

Lowest to highest:

```
dataclass default  <  station YAML  <  recipe YAML  <  CLI --robot.xxx=
```

Every station-supplied config field defaults to `None`, and the station only
fills the ones the caller left alone. So an explicit value always wins:

```bash
# run forward-05 with a spare left arm, without editing the station file
lerobot-teleoperate --robot.bi_mount_type=forward-05 --robot.left_robot_sn=Rizon4-000001
```

> Before stations existed, the Flexiv config overwrote these unconditionally and
> a `left_robot_sn` written into a recipe was silently discarded. It is honoured
> now — if you have an old recipe carrying a stale hardware value, it will start
> taking effect. Grep your recipes for `_sn`, `_ip`, `_start`, `_home`, `mount_`
> and `cameras` before upgrading.

Two fields need a note:

- **`cameras`** is all-or-nothing. Supply a non-empty dict and the station's
  cameras are ignored wholesale, rather than merged key by key.
- **`{left,right}_world_rotation`** (Elite) is three-state, because `None` now
  means "inherit". Pass `[]` to say "ignore the station's matrix, build the
  rotation from the `tilt`/`zrot`/`world_yaw` angles instead".

## Where files are looked up

First hit wins:

1. `bi_mount_type` used verbatim, if it contains `/` or ends in `.yaml`/`.yml` —
   for a bench not worth committing yet
2. `$LEROBOT_STATIONS_DIR/<robot_type>/<name>.yaml`
3. `$PWD/stations/<robot_type>/<name>.yaml`
4. `<repo>/stations/<robot_type>/<name>.yaml`, inferred from the installed
   package location

(4) only resolves for an editable install, which is what `setup_env.sh` produces
and what every bench in the fleet runs. Under a plain wheel the directory is not
present — set `LEROBOT_STATIONS_DIR` to a checkout's `stations/`. There is
deliberately no copy of these files inside the package; a second copy would only
drift from this one.

Because (2) and (3) come first, a bench can shadow a committed station with a
local file of the same name for bring-up, without touching the repo.

## Format

```yaml
name: forward-05                  # must equal the filename
robot_type: bi_flexiv_rizon4_rt   # must equal the parent directory

arms:
  left:
    serial_number: Rizon4-063786
    start_deg: [96.59, 65.45, -6.65, 78.76, 85.09, -5.48, -103.47]
    home_deg:  [96.59, 65.45, -6.65, 78.76, 85.09, -5.48, -103.47]
  right:
    ...

cameras:
  head:            {type: realsense,     serial: "346522071766"}
  left_wrist:      {type: opencv,        serial: XC000047}
  left_tactile_0:  {type: xense_tactile, serial: OG001359}
  ...
```

`name` and `robot_type` duplicate the file's location on purpose: the loader
cross-checks both, so a file copied into the wrong directory fails loudly
instead of quietly describing the wrong bench.

**Camera keys are observation keys.** `left_wrist` here is `left_wrist` in the
observation dict and in the recorded dataset, so renaming one renames the
dataset column. Camera `type` is one of `realsense`, `opencv`, `xense_tactile`.

Each camera may also carry `fps`, `width`, `height`, `warmup_s`, `use_depth`
(realsense only) or `camera_properties` (xense_tactile only). These are
**overrides**: omit them and the robot's own default applies. Set one only when
this bench genuinely differs — an override that merely restates the default is
noise that rots the day the default moves.

### Per-robot fields

**`bi_flexiv_rizon4_rt`** — `arms.{left,right}` take `serial_number`,
`start_deg`, `home_deg` (J1..J7, degrees).

**`bi_elite_cs66_rt`** — `arms.{left,right}` take `ip`, `local_ip`, `start_rad`,
`home_rad` (J1..J6, radians), and a `mount` block giving the world←base rotation:

```yaml
    mount:
      tilt_deg: 45.0          # α, about base-X   ┐ teach-pendant readings;
      zrot_deg: 90.0          # β, about Z        ┘ these fix gravity, not heading
      world_yaw_deg: 180.0    # γ, about world-Z; aligns headings into one frame
      world_rotation:         # optional; OVERRIDES the angle build above
        - [0.0, 1.0, 0.0]     # rows = world X/Y/Z axes expressed in base
        - [-0.70710678, 0.0, 0.70710678]
        - [0.70710678, 0.0, 0.70710678]
```

The full derivation for the existing diagonal benches — including why the left
arm's real tilt is about base-Y and not the base-X the pendant implies — is in
[`bi_elite_cs66_rt/diagonal-07.yaml`](bi_elite_cs66_rt/diagonal-07.yaml).

### What is deliberately absent

**Grippers.** Both backends resolve left/right at connect — serial by board-SN
parity (odd → left, even → right), taccap by the firmware-burned SN — so there is
no per-bench gripper identity to pin.

**Wrist and tactile cameras, when auto-discovery is on.** Those move with the
gripper, so the robot can sniff them at connect instead of reading them from the
station — `taccap_auto_discover_cameras` (on by default) for TacCap grippers, and
`serial_auto_discover_cameras` (**off** by default) for serial parallel-jaw ones.
Keep the entries in the file either way: they are the source of truth whenever
discovery is off, and the same station serves both backends.

## Camera auto-discovery

Each arm's gripper, wrist camera and two tactile sensors sit behind one per-side
USB hub. The gripper already knows its own side — by firmware SN for TacCap, by
board-SN parity for serial — so its hub identifies the arm, and every other
device on that hub belongs to the same arm. No serial number required.

The payoff is not the six saved lines; it is that swapping a tactile pad or a
wrist camera stops being a config edit.

```bash
# What discovery would resolve, without connecting a robot:
python -c "
from lerobot.robots.grippers.serial_discovery import discover_serial_gripper_cameras
for s, d in sorted(discover_serial_gripper_cameras().items()): print(s, d)"
```

For the serial backend this is **off by default**, so the station file stays the
verified source of truth. Before turning it on for a bench, run the command above
and check the six serials against that bench's station file. Then set it in the
recipe:

```yaml
robot:
  serial_auto_discover_cameras: true
```

Discovery **refuses rather than guesses**: if a hub has anything other than
exactly one non-tactile video device, or fewer than two tactile sensors, it
raises and names what it found. A plausible-but-wrong camera would mislabel a
dataset in a way nobody notices until the recording is useless.

Tactile order follows USB port, not enumeration order, so `*_tactile_0` is the
same physical pad on every run.

> Why topology rather than a naming rule: the wrist cameras do follow
> odd→left/even→right across all seven benches, but the tactile serials follow
> nothing — `diagonal-02` has left `OG000867, OG000865` and right `OG000142,
> OG000866`, so any rule based on ordering or parity is already broken by the
> hardware on the floor.

## Adding a bench

1. Copy the nearest existing station and rename both the file and its `name`.
2. Fill in arm identities, measured start/home poses, and camera serials.
3. For Elite, work out the `mount` block and **verify it on-station** with the
   axis test before trusting it.
4. Check it loads and resolves the way you expect, without touching hardware:

   ```bash
   python -c "
   from lerobot.robots.bi_flexiv_rizon4_rt.config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig as C
   c = C(bi_mount_type='forward-07')
   print(c.left_robot_sn, c.right_robot_sn, sorted(c.cameras))"
   ```

5. `pytest tests/robots/test_stations.py` — the per-file tests pick up new
   stations automatically.
6. Add a `recipes/teleop/<robot_type>/<name>.yaml` pointing at it.

Adding a station touches no Python. If you find yourself editing a config
dataclass to bring up a bench, that is the signal something bench-specific has
leaked into the robot — fix that instead of working around it.
