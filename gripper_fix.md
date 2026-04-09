# 夹爪状态读取修复记录

## 根本原因

1. **Host 端串口竞争**：两个夹爪并行 `connect()` 时，SN=000002 的端口扫描会打开已被 SN=000001 持有的 `/dev/ttyUSB0`，随后关闭触发 `HUPCL`，导致 SN=000001 的 `_run_loop` 收到 `b''`（EOF），抛出 `SerialException` 后永久退出，之后所有状态查询返回 `None`（`position=nan`）。

2. **固件不主动推送状态**：MCU 只在收到 `0x9C` 查询时才发状态帧，`_run_loop` 一旦崩溃，Host 就永远收不到状态。

---

## Fix 1 — Host：防止并发扫描污染已连接端口

**文件**：`third_party/XGripper/xensegripper/serial_device.py`

```python
# 修改前
self.ser = serial.Serial(
    port=port,
    baudrate=baudrate,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=timeout
)

# 修改后（加 exclusive=True）
self.ser = serial.Serial(
    port=port,
    baudrate=baudrate,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=timeout,
    exclusive=True,
)
```

`exclusive=True` 使 OS 层面阻止第二次 `open()`，并发扫描到已连接端口时得到 `EBUSY`，被 `find_port_by_sn` 的 `except Exception: pass` 跳过，不影响已连接夹爪的 `_run_loop`。

---

## Fix 2 — 固件：启用状态自动推送

**文件**：`gripper_/Application/Src/gripper_ctl.c`，`StartGripperTask` 函数，约第 240 行

```c
// 修改前（自动推送代码被注释，MCU 从不主动发状态）
if(gripper.type == GRIPPER_STATE)
{
    gripperState.pos   = gripper.Data.state.pos;
    gripperState.force = gripper.Data.state.force;
    gripperState.speed = gripper.Data.state.speed;

//  resp.Head             = GRIPPER_FRAME_HEAD;
//  resp.Id               = 1;
//  resp.Length           = 8;
//  resp.Data.State.Cmd   = GRIPPER_GET_STATE;
//  resp.Data.State.Pos   = gripperState.pos;
//  resp.Data.State.Force = gripperState.force;
//  resp.Data.State.Speed = gripperState.speed;
//
//  uint16_t crc = crc16((uint8_t *)&resp, sizeof(GripperFrame)-2);
//  resp.Crc16 = crc;
}

// 修改后（取消注释 + 补充 gripper_com_send 调用）
if(gripper.type == GRIPPER_STATE)
{
    gripperState.pos   = gripper.Data.state.pos;
    gripperState.force = gripper.Data.state.force;
    gripperState.speed = gripper.Data.state.speed;

    resp.Head              = GRIPPER_FRAME_HEAD;
    resp.Id                = 1;
    resp.Length            = 8;
    resp.Data.State.Cmd    = GRIPPER_GET_STATE;
    resp.Data.State.Pos    = gripperState.pos;
    resp.Data.State.Force  = gripperState.force;
    resp.Data.State.Speed  = gripperState.speed;

    uint16_t crc = crc16((uint8_t *)&resp, sizeof(GripperFrame)-2);
    resp.Crc16 = crc;
    gripper_com_send((uint8_t *)&resp, sizeof(GripperFrame), 40);  // 原来缺的这行
}
```

每次内部状态更新（~50 Hz），MCU 自动向 UART1 推一帧状态，Host `_run_loop` 无需轮询即可持续收到数据，`_gripper_status` 缓存始终保持新鲜。

---

## 两个 Fix 的协同效果

| | Fix 1 (`exclusive=True`) | Fix 2 (固件自动推送) |
|---|---|---|
| 防止 `_run_loop` 崩溃 | ✅ | — |
| `_run_loop` 崩溃后仍可恢复 | — | ✅（持续推送，Host 只需重连） |
| 初始化 sync move 不再 timeout | ✅（不崩溃） | ✅（不依赖纯 poll） |
| 降低 Host 轮询压力 | — | ✅ |
