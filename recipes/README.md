# CLI recipes

YAML configs for `lerobot-teleoperate` and `lerobot-record`. Instead of
copy-pasting a long command, store the parameters in a file here and pass its
path — both CLIs accept `--config_path` (they share the same draccus parser).

```bash
# Teleoperate
lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal.yaml

# Record
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test.yaml
```

CLI flags still work and **override** the YAML, so you don't edit the file for
one-off tweaks:

```bash
lerobot-record --config_path=recipes/record/bi_elite_cs66_rt/test.yaml \
    --dataset.num_episodes=1 --resume=true

lerobot-teleoperate --config_path=recipes/teleop/bi_elite_cs66_rt/diagonal.yaml --dryrun=true
```

## Layout

Split by CLI, then by robot type:

```
recipes/
  teleop/<robot_type>/<variant>.yaml    -> lerobot-teleoperate --config_path=…
  record/<robot_type>/<task>.yaml       -> lerobot-record      --config_path=…
```

```
recipes/
  teleop/
    bi_elite_cs66_rt/diagonal.yaml
    bi_flexiv_rizon4_rt/forward.yaml
  record/
    bi_elite_cs66_rt/test.yaml
    bi_flexiv_rizon4_rt/assemble_box.yaml
```

## What goes in a recipe

Every **honored per-recipe knob** is written out explicitly with its value —
control mode, servo/timeout params, gripper motion limits, trace thresholds,
dataset fields, teleop sensitivities, and the top-level loop flags — so an
operator can see and change any parameter without reading the code.

**Station hardware is deliberately NOT listed**: cameras, gripper/robot SNs,
mount geometry (`*_mount_*_deg`, `world_rotation`), `local_ip`, and home/start
poses. Those come from the `bi_mount_type` preset, which **overwrites them at
load** (`__post_init__`) — writing them into a recipe would be silently ignored.
Choose the physical station via `bi_mount_type`.

- Elite: `left_robot_ip` / `right_robot_ip` are explicit-wins, so they *are* in
  the recipe; everything else hardware-ish comes from the preset.
- Flexiv: robot SNs are overwritten unconditionally by the preset, so they are
  **not** in the recipe — switch stations with `bi_mount_type` only.
- Flexiv `force_control_frame` is a C++ enum with no YAML decoder, so it is set
  in code (defaults to `WORLD`) and omitted from the recipe.

## YAML format

The YAML mirrors the CLI flags exactly: a dotted `--robot.servoj_gain=300`
becomes nested `robot: { servoj_gain: 300 }`, and `type:` is the discriminator
that selects the robot / teleop class. Enums use their string value
(`control_mode: cartesian_servo`).

## Naming convention

- `teleop/<robot_type>/<variant>.yaml` — variant = mount/station, e.g.
  `diagonal.yaml`, `forward.yaml`.
- `record/<robot_type>/<task>.yaml` — one per dataset/campaign, e.g.
  `assemble_box.yaml`. Keep a `test.yaml` smoke-test per robot (2 short
  episodes, `push_to_hub: false`).

## Why this over copy-pasting from a markdown file

- **Single source of truth** — the recipe *is* the runnable artifact.
- **Provenance for free** — recipes are committed, so `git log recipes/` shows
  exactly which parameters produced each dataset and when.
- **No silent mistakes** — no forgetting `--resume=false` or pasting a wrong SN.
- **Diff-friendly** — bumping `num_episodes` is a one-line, reviewable change.

The full flag reference for every robot still lives in
[`../src/lerobot/scripts/client_commands.md`](../src/lerobot/scripts/client_commands.md).
