```bash
sudo v4l2-ctl -d /dev/video14   --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl -d /dev/video14 --get-fmt-video
``
```bash
sudo dmesg -w | grep -E "usb|uvc|bandwidth|altsetting"
```

```bash
sudo dkms remove -m uvcvideo-xense -v 1.0 --all
sudo dkms add -m uvcvideo-xense -v 1.0
sudo dkms build -m uvcvideo-xense -v 1.0
sudo dkms install -m uvcvideo-xense -v 1.0

sudo modprobe -r uvcvideo
sudo modprobe uvcvideo

sudo dmesg | tail -30 | grep -i "altsetting\|3938\|10bb"
```
## Xense Multisensor Robot lerobot-teleoperate command

```python
lerobot-teleoperate \
    --robot.type=xense_multisensor \
    --teleop.type=mock_teleop \
    --fps=30 \
    --debug_timing=false \
    --display_data=true
```