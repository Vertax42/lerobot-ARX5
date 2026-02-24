#!/usr/bin/env python3
"""
Spacemouse sender - runs on the machine with the spacemouse connected.
Outputs spacemouse state as JSON lines to stdout.

Usage (on remote machine with spacemouse):
    python spacemouse_sender.py

To pipe over SSH to robot:
    ssh user@robot-pc "cd /path/to/xense_franka && python examples/spacemouse_receiver.py" < <(python spacemouse_sender.py)

Or from robot side:
    ssh user@spacemouse-pc "python spacemouse_sender.py" | python examples/spacemouse_receiver.py
"""

import pyspacemouse
import json
import sys
import time

def main():
    success = pyspacemouse.open()
    if not success:
        print("ERROR: Failed to connect to spacemouse", file=sys.stderr)
        sys.exit(1)
    
    print("Spacemouse connected, streaming data...", file=sys.stderr)
    
    try:
        while True:
            event = pyspacemouse.read()
            
            # Output as JSON line
            data = {
                "x": event.x,
                "y": event.y,
                "z": event.z,
                "roll": event.roll,
                "pitch": event.pitch,
                "yaw": event.yaw,
                "buttons": event.buttons,
                "t": time.time()
            }
            
            print(json.dumps(data), flush=True)
            
            # ~100Hz update rate
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopped", file=sys.stderr)

if __name__ == "__main__":
    main()
