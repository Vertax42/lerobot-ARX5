from xense_franka import SyncFrankaController

controller = SyncFrankaController("192.168.99.111")
controller.start()
controller.move([0, 0, 0, -1.57, 0, 1.57, 0.78])
controller.switch("osc")

state = controller.get_state()
print(state)
ee = controller.get_ee_pose()
print("Current EE pose:\n", ee) #16d
e44 = ee.reshape(4, 4).T
print("Current EE pose (4x4):\n", e44) #16d
joint_positions = controller.get_joint_positions()
print("Current joint positions:\n", joint_positions) #7d
# for i in range(100):
#     ee = controller.get_ee_pose()
#     ee[:3, 3] += [0.001, 0, 0]
#     controller.set_ee_pose(ee)

controller.stop()