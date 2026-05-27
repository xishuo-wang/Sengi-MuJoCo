简化双关节象鼩-MuJoCo

===============================================================================
1、260527_1
XML配置
friction="1 0.01 0.01" solref="0.005 1"
<body name="base_link" pos="0 0 0.10" quat="0 0 0.707107 0.707107">

参数搜索
TOR = 0.5
TOR_SPINE = TOR*1.5

仿真
OPTIMAL_A_LEGH_HIP = 0.9
OPTIMAL_A_LEGH_KNEE = 0.5
OPTIMAL_PHASE_LAG = -np.pi / 20 * 1

冲量分析
冲量计算区间: 0.10s - 0.20s
Ix (水平): 0.150466 N·s
Iz (垂直): 0.375141 N·s
|I| (矢量和): 0.404191 N·s
Fx区间最大力: 3.3273 N
Fz区间最大力: 7.6413 N
合力最大力: 8.2266 N