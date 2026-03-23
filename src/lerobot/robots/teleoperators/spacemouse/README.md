# SpaceMouse teleoperator

Uses [`pyspacemouse`](https://pypi.org/project/pyspacemouse/) (HID via `easyhid`).

## Linux: `hidraw` permissions

If you see **`Failed to open device`** / **`HIDException`**, your user usually cannot read `/dev/hidraw*`.

1. **udev rule** (grants group `plugdev` read/write on hidraw nodes):

   ```bash
   echo 'KERNEL=="hidraw*", SUBSYSTEM=="hidraw", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-hidraw-permissions.rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

2. **Add your user to `plugdev`**:

   ```bash
   sudo usermod -aG plugdev "$USER"
   ```

3. **Apply the new group** in the current session (or log out and back in):

   ```bash
   newgrp plugdev
   ```

4. **Unplug and replug** the SpaceMouse USB (or receiver), then retry.

This rule affects **all** `hidraw` devices. For a stricter setup, use a rule matched on 3Dconnexion / SpaceMouse USB vendor/product IDs only.

Also ensure no other program holds the device (3DxWare, `spacenavd`, Blender, another teleop process).
