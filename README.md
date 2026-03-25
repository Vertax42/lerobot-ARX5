# 🎯 Project Overview

🤗 This repository is a fork of lerobot by XenseRobotics, used for Xense's multimodal tactile data acquisition system. It implements numerous new features compared to the original repository.

## 🔧 Installation

This repository is tested on Ubuntu 22.04, NVIDIA Unix Driver Archive >= 570.144. We strongly recommend using [`Mamba`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) to manage your conda environments. You can also use conda, but it will take significantly longer. Due to the xensesdk dependencies, lerobot-xense works only with python 3.10 and pytorch 2.8.0+ with cuda-12.8 right now.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### 📦 Environment Setup

**Step 1:** 📂 Clone the repository with all submodules:

```bash
git clone \
  --recurse-submodules \
  https://github.com/Vertax42/lerobot-xense.git
cd lerobot-xense
```

> If you already cloned without submodules, initialize them manually:

> ```bash
> git submodule update --init --recursive --progress
> ```

This repository uses `third_party/` git submodules to manage hardware SDK dependencies:

| Submodule | Installed package |
|-----------|-------------------|
| `third_party/ARX5_SDK` | `pyarx` |
| `third_party/libpyflexiv` | `flexiv_rt` |
| `third_party/XenseVR-PC-Service` | `xensevr_pc_service_sdk` |
| `third_party/xensesdk` | `xensesdk` |
| `third_party/XGripper` | `xensegripper` |
| `third_party/xense_franka` | `xense_franka` |

**Step 2:** 🐍 Create and activate the conda/mamba environment:

```bash
bash ./setup_env.sh --mamba <optional_env_name>
mamba activate <optional_env_name> # or conda activate <optional_env_name>
```

**Step 3:** 📦 Install LeRobot-Xense and all hardware SDK bindings:

```bash
bash ./setup_env.sh --install
```

This step will:

- Update the conda environment from `conda_environment.yaml`
- Install the main package from `pyproject.toml`
- Build and install all `third_party` SDK packages: `pyarx`, `flexiv_rt`, `xensevr_pc_service_sdk`, `xensesdk`, `xensegripper`, `xense_franka`
- Configure SpaceMouse udev rules and HID permissions automatically

> You will be prompted for `sudo` password during installation (for ARX5 real-time capability and udev rules).

**Step 4:** ✅ Verify the installation:

```bash
python -c 'import pyarx; print("pyarx OK ->", pyarx.__file__)'
python -c 'import flexiv_rt; print("flexiv_rt OK ->", flexiv_rt.__file__)'
python -c 'import xensevr_pc_service_sdk; print("xensevr_pc_service_sdk OK ->", xensevr_pc_service_sdk.__file__)'
python -c 'import xensesdk; print("xensesdk OK ->", xensesdk.__file__)'
python -c 'import xensegripper; print("xensegripper OK ->", xensegripper.__file__)'
```

**Step 5:** 📌 Check if FFmpeg 7.X is installed in your environment and `libsvtav1` encoder is supported:

```bash
mamba list | grep ffmpeg
ffmpeg -encoders | grep libsvtav1
```

### ARX5 Real-time Thread Permissions

The ARX5 SDK requires `CAP_SYS_NICE` on the Python interpreter for real-time CAN thread scheduling. This is handled by `setup_env.sh --install`, but can be set manually:

```bash
PY_EXE=$(python -c 'import sys, os; p = sys.executable; print(os.path.realpath(p))')
sudo setcap cap_sys_nice+ep "$PY_EXE"
getcap "$PY_EXE"  # should show: cap_sys_nice+ep
```

## 🐭 SpaceMouse Teleoperation System

This project includes advanced SpaceMouse support with both single and dual-device modes for precise robotic control.

### Dependencies

**System Requirements:**

- Ubuntu 22.04 (tested) or other Linux distributions
- Python 3.10+
- libhidapi (installed via apt)

**Python Packages:**

- `pyspacemouse` - Modern cross-platform SpaceMouse library
- `hidapi` - Python wrapper for HID API
- `easyhid` - Easy-to-use HID library (dependency of pyspacemouse)

All Python dependencies are automatically installed by `setup_env.sh --install`.

### Permissions Setup

SpaceMouse requires proper udev rules to allow non-root access. See **Step 2** in the Installation section above for complete setup instructions.

### Testing Your SpaceMouse

After installation and permissions setup, test your SpaceMouse:

```bash
# Basic functionality test (prints real-time 6-DoF values)
python test_pyspacemouse_basic.py

# Test with lerobot integration
python test_spacemouse.py
```

The test script will display real-time position (x, y, z) and orientation (roll, pitch, yaw) values as you move the SpaceMouse.

> 📝 **Note:** If you're using a 3Dconnexion Universal Receiver (wireless), you may see multiple devices listed (e.g., 14 "UniversalReceiver" entries). This is normal - the receiver exposes multiple HID interfaces for different functions. PySpaceMouse will automatically select the correct interface for 6-DoF input.

### Features

- ✅ **Modern PySpaceMouse Integration**: Uses PySpaceMouse library for cross-platform SpaceMouse support
- ✅ **No System Services Required**: Direct HID communication, no need for spacenavd daemon  
- ✅ **Single Device Mode**: Traditional 6-DoF control with one SpaceMouse
- ✅ **Dual Device Mode**: Advanced left/right hand coordination for complex manipulation
- ✅ **Flexible Axis Assignment**: Configure which device controls position vs orientation
- ✅ **Independent Sensitivity**: Per-device sensitivity settings for optimal control

### Single Device Configuration

```python
from lerobot.teleoperators.spacemouse import SpacemouseConfig, SpacemouseTeleop

# Standard single SpaceMouse setup (default)
config = SpacemouseConfig(
    pos_sensitivity=0.8,     # Position control sensitivity
    ori_sensitivity=1.5,     # Orientation control sensitivity
    deadzone=0.1,           # Deadzone threshold
    frequency=200,          # Polling frequency (Hz)
)

teleop = SpacemouseTeleop(config)
```

### Dual Device Configuration

Perfect for complex robotic tasks requiring precise position and orientation control:

```python
from lerobot.teleoperators.spacemouse import SpacemouseConfig, DeviceConfig

# Left hand controls position, right hand controls orientation
config = SpacemouseConfig(
    multi_device_mode=True,
    left_device=DeviceConfig(
        device_index=0,
        enabled_axes=(True, True, True, False, False, False),  # X, Y, Z position only
        pos_sensitivity=0.8,
        ori_sensitivity=0.0,  # Disabled
    ),
    right_device=DeviceConfig(
        device_index=1, 
        enabled_axes=(False, False, False, True, True, True),  # Roll, pitch, yaw only
        pos_sensitivity=0.0,  # Disabled
        ori_sensitivity=1.5,
    )
)

teleop = SpacemouseTeleop(config)
```

### Example Configurations

See [examples/spacemouse_dual_config_example.py](examples/spacemouse_dual_config_example.py) for complete configuration examples including:
- Position/Orientation split control
- Dual-arm robot control  
- Fine/Coarse movement control

### Use Cases

- 🤖 **Dual-Arm Robots**: Independent control of two robotic arms
- 🎯 **Precision Manipulation**: Decouple position and orientation control for fine tasks
- 🔄 **Complex Assembly**: Left hand positions, right hand orients components
- 🏭 **Industrial Applications**: Enhanced ergonomics and control precision

### Hardware Requirements

- **Single Mode**: Any 3Dconnexion SpaceMouse device
- **Dual Mode**: Two identical SpaceMouse devices (e.g., two SpaceNavigators)

### Supported Devices

All 3Dconnexion devices supported by PySpaceMouse:
- SpaceNavigator
- SpaceMouse Pro
- SpaceMouse Wireless
- SpaceMouse Compact
- And more...

## 🤖 Flexiv Rizon4 Robot with Flare Gripper Policy Implementation

Lerobot record XenseFlare dataset can be directly used for FlexivRizon4 policy training.  🎉

### Bimanual Flexiv Rizon4 RT + BiPico4 Record Controls

For `lerobot-record` with `--robot.type=bi_flexiv_rizon4_rt --teleop.type=bi_pico4`, the controller buttons are mapped as follows:

- Right controller `A`: reset both arms to the start pose using the non-blocking RT reset path
- Left controller `X`: re-record the current episode
- Left controller `Y`: finish the current episode early and continue to the next step
- Right controller `B`: stop the recording session

During RT reset, the record loop keeps running: observations are still sampled, teleop actions are still read, and the teleop pose is re-synced to the robot once the reset trajectory finishes.

## Record Loop Implementation (`flexiv_rizon4_rt_record_loop`)

This section documents the dataset construction logic in `flexiv_rizon4_rt_record_loop`, which handles both normal teleoperation recording and the RT reset trajectory recording for `bi_flexiv_rizon4_rt + bi_pico4`.

### State Variables

| Variable | Role |
|---|---|
| `reset_triggered` | Per-frame flag. Set `True` the frame reset is triggered. Skips `send_action` and dataset write for that frame only. Automatically `False` at the start of every iteration. |
| `prev_rt_moving` | Edge-detection flag. Set `True` while `robot.rt_moving` is `True`. Cleared to `False` when movement stops, at which point `_sync_rt_teleop_to_robot_pose()` is called once. |
| `prev_observation_frame` | Holds the previous frame's observation dict. Used by the shifted-frame logic to pair `obs[t-1]` with the robot's actual position at `obs[t]` as the action. |

### Three Frame Modes

#### 1. Normal teleoperation (`robot_is_moving=False`, `reset_triggered=False`)

```
robot.send_action(teleop_action) → sent_action
dataset: { obs[t],  action = sent_action[t] }   # direct frame
prev_observation_frame = obs[t]
```

Action is what was actually commanded to and accepted by the robot this frame.

#### 2. Reset trigger frame (`reset_triggered=True`)

```
robot.reset_to_initial_position()   # C++ RT thread takes over arm control
send_action  → skipped
dataset      → skipped
prev_observation_frame = obs[T]     # saved as anchor for next iteration
```

`obs[T]` is intentionally not written to the dataset here. It will be used as `prev_observation_frame` for the first shifted frame in the next iteration, so it appears exactly once in the dataset (as `obs[t-1]` of a shifted pair). Without this skip, `obs[T]` would appear twice — once as a direct frame with `teleop_action[T]`, and again as the prev of the first shifted frame.

#### 3. RT reset in progress (`robot_is_moving=True`)

```
send_action  → skipped (C++ RT thread drives the arm)
current_as_action = { key: obs[t][key] for key in robot.action_features }
dataset: { obs[t-1],  action = current_as_action }   # shifted frame
prev_observation_frame = obs[t]
```

The action is extracted directly from the current observation using the same keys as `robot.action_features` (TCP pose + gripper). This records where the robot actually moved to, not what the teleop commanded. This is the same shifted-frame convention used by `bi_arx5_record_loop` for gravity-compensation demonstrations.

### Complete Frame Sequence Around a Reset

```
frame T-2  normal teleop  →  dataset: { obs[T-2], action[T-2] }
frame T-1  normal teleop  →  dataset: { obs[T-1], action[T-1] },  prev=obs[T-1]
frame T    reset trigger  →  dataset: skipped,                     prev=obs[T]
frame T+1  rt_moving      →  dataset: { obs[T],   obs[T+1]_pos },  prev=obs[T+1]
frame T+2  rt_moving      →  dataset: { obs[T+1], obs[T+2]_pos },  prev=obs[T+2]
  ...
frame N    rt_moving      →  dataset: { obs[N-1], obs[N]_pos },    prev=obs[N]
frame N+1  reset done     →  _sync_rt_teleop_to_robot_pose()
           normal teleop  →  dataset: { obs[N+1], action[N+1] }
```

### Post-Reset Teleop Sync

When `prev_rt_moving` transitions from `True` to `False` (frame N+1), `_sync_rt_teleop_to_robot_pose()` is called. This reads the robot's current TCP pose (now at start position) and calls `teleop.reset_to_pose()`, updating the Pico4's internal `_start_pos` reference. Without this sync, the teleop would compute position deltas from the pre-reset pose, causing the arm to jump on the first grip after reset.

## 🔑 The `LeRobotDataset` format

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}` one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of an episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ meta: a LeRobotDatasetMetadata object containing:
  │  ├ info: a dictionary of metadata on the dataset
  │  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  │  ├ fps (int): frame per second the dataset is recorded/synchronized to
  │  │  ├ features (dict): all features contained in the dataset with their shapes and types
  │  │  ├ total_episodes (int): total number of episodes in the dataset
  │  │  ├ total_frames (int): total number of frames in the dataset
  │  │  ├ robot_type (str): robot type used for recording
  │  │  ├ data_path (str): formattable string for the parquet files
  │  │  └ video_path (str): formattable string for the video files (if using videos)
  │  ├ episodes: a DataFrame containing episode metadata with columns:
  │  │  ├ episode_index (int): index of the episode
  │  │  ├ tasks (list): list of tasks for this episode
  │  │  ├ length (int): number of frames in this episode
  │  │  ├ dataset_from_index (int): start index of this episode in the dataset
  │  │  └ dataset_to_index (int): end index of this episode in the dataset
  │  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  │  ├ observation.images.front_cam: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  │  └ ...
  │  └ tasks: a DataFrame containing task information with task names as index and task_index as values
  ├ root (Path): local directory where the dataset is stored
  ├ image_transforms (Callable): optional image transformations to apply to visual modalities
  └ delta_timestamps (dict): optional delta timestamps for temporal queries
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:

- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

## 📝 Recent Updates

### SpaceMouse System Upgrade (2025-01-23)

🎉 **Major SpaceMouse System Overhaul:**

- **Modern Library Migration**: Migrated from legacy `spnav` to modern `PySpaceMouse` library
- **Cross-Platform Support**: Now supports Linux, macOS, and Windows
- **No System Dependencies**: Removed requirement for `spacenavd` system service
- **Dual-Device Support**: Revolutionary dual SpaceMouse mode for advanced manipulation
- **Flexible Configuration**: Per-device sensitivity and axis assignment
- **Hardware Independence**: Direct HID communication for better reliability

**Breaking Changes:**
- `spacenavd` service is no longer required
- Configuration options have been expanded with new dual-device parameters
- Old single-device configurations remain fully compatible

**Migration Benefits:**
- ✅ Easier setup (no system services to configure)
- ✅ Better cross-platform compatibility  
- ✅ More responsive input handling
- ✅ Advanced dual-hand control capabilities
- ✅ Future-proof with active library maintenance

## Citation

If you use this codebase, please cite the original LeRobot project:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
