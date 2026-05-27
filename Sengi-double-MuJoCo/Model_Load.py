import mujoco_py
import numpy as np
import time

# 加载模型
model = mujoco_py.load_model_from_path(r"D:\Code\Model\Sengi\Sengi.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)  

# 定义关节名称列表
joint_names = [
    "back_joint", "hindleg_joint_1", "hindleg_joint_2", "hindleg_joint_3",
    "front_joint", "foreleg_joint_1", "foreleg_joint_2", "foreleg_joint_3"
]

# 获取关节在 qpos 中的索引
joint_indices = []
for name in joint_names:
    joint_id = sim.model.joint_name2id(name) 
    qpos_addr = sim.model.jnt_qposadr[joint_id]
    joint_indices.append(qpos_addr)
print(joint_indices)
initial_joint_pos = [0, 0, 0, 0, 0, 0, 0, 0]

# PD控制器参数
Kp = 2
Kd = 0.1

# PD控制器
def PDcontrol(target_pos):
    # 获取当前关节位置
    current_pos = sim.data.qpos[joint_indices].copy()
    
    # 获取关节速度
    current_vel = np.zeros(len(joint_indices))
    for i, name in enumerate(joint_names):
        try:
            joint_id = sim.model.joint_name2id(name)
            # 获取关节在 qvel 中的地址
            dof_addr = sim.model.jnt_dofadr[joint_id]
            if dof_addr >= 0 and dof_addr < len(sim.data.qvel):
                current_vel[i] = sim.data.qvel[dof_addr]
        except Exception:
            current_vel[i] = 0.0
    
    pos_error = target_pos - current_pos
    vel_error = 0 - current_vel 
    torque = Kp * pos_error + Kd * vel_error
    torque = np.clip(torque, -0.3, 0.3)
    return torque

# 主循环
dt = sim.model.opt.timestep
current_time = 0.0

while True:
    target_pos = initial_joint_pos.copy()
    
    target_pos[6] = -1
    torque = PDcontrol(target_pos)
    
    for i in range(sim.model.nu):
        sim.data.ctrl[i] = torque[i]
    
    sim.step()
    viewer.render()
    current_time += dt