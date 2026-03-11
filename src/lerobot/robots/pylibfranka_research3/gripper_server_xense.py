from fastapi import FastAPI
import threading
from xensegripper import XenseGripper
import uvicorn,time,argparse
from pydantic import BaseModel

class MoveRequest(BaseModel):
    pos: float
    vmax: float = 100
    fmax: float = 30
    
class GripperXense:
    def __init__(self,id="7ec0c7f50ea6"):
        self.gripper = XenseGripper.create(id)
        self.running = True
        self.status = {}
        self.start()

    def start(self):
        self.thread = threading.Thread(target=self.get_event, daemon=True)
        self.thread.start()

    def get_event(self):
        while self.running:
            self.status = self.gripper.get_gripper_status()
            time.sleep(0.05)

    def move(self, pos, vmax=100, fmax=30):
        self.gripper.set_position(pos, vmax, fmax)

    def get_pos(self):
        print(self.status)
        return self.status.get("position", 0.0)

    def stop(self):
        self.thread.join()

app = FastAPI()

@app.get("/get_pos")
def get_pos():
    return {"position": gripper.get_pos()}

@app.post("/move")
def move(req: MoveRequest):
    print("Received move request:", req)

    gripper.move(req.pos, req.vmax, req.fmax)

    return {
        "status": "ok",
        "target": req.pos,
        "vmax": req.vmax,
        "fmax": req.fmax,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--port", type=int, default=7001)
    args = parser.parse_args()
    
    gripper = GripperXense(id=args.id)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
