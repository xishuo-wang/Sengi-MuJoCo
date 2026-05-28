简化单关节象鼩-MuJoCo

===============================================================================
260527_1
XML配置
friction="0.45 0.01 0.01" solimp="0.9 0.95 0.01" solref="0.01 0.5"
<body name="base_link" pos="0 0 0.10" quat="0 0 0.707107 0.707107">

参数搜索
F = 3
TOR = 0.5
TOR_SPINE = TOR * 1.5
A_SPINE = -0.7

仿真
OPTIMAL_A_LEGH_HIP = 1.4
OPTIMAL_A_LEGH_KNEE = 1
OPTIMAL_PHASE_LAG = np.pi / 20 * 2

冲量分析
平均速度: 0.2655 m/s
冲量计算区间: 0.08s - 0.20s
Ix (水平): 0.096089 N·s
Iz (垂直): 0.599450 N·s
|I| (矢量和): 0.607103 N·s
Fx区间最大力: 4.8002 N
Fz区间最大力: 24.0381 N
合力最大力: 24.3172 N


===============================================================================
260528_1
XML配置
friction="0.45 0.01 0.01" solimp="0.9 0.95 0.01" solref="0.01 0.5"
<body name="base_link" pos="0 0 0.10" quat="0 0 0.707107 0.707107">

参数搜索
F = 2
TOR = 0.5
TOR_SPINE = TOR * 1.5
A_SPINE = -0.7

仿真
OPTIMAL_A_LEGH_HIP = 1.3
OPTIMAL_A_LEGH_KNEE = 0
OPTIMAL_PHASE_LAG = np.pi / 20 * 7

冲量分析
平均速度: 0.2150 m/s
冲量计算区间: 0.07s - 0.20s
Ix (水平): 0.071724 N·s
Iz (垂直): 0.461739 N·s
|I| (矢量和): 0.467277 N·s
Fx区间最大力: 4.4124 N
Fz区间最大力: 17.6620 N
合力最大力: 17.8407 N


===============================================================================
260528_1
XML配置
friction="0.45 0.01 0.01" solimp="0.9 0.95 0.01" solref="0.01 0.5"
<body name="base_link" pos="0 0 0.10" quat="0 0 0.707107 0.707107">

参数搜索
F = 2
TOR = 0.5
TOR_SPINE = TOR * 1.5
A_SPINE = -0.7

仿真
OPTIMAL_A_LEGH_HIP = 1.3
OPTIMAL_A_LEGH_KNEE = 0
OPTIMAL_PHASE_LAG = np.pi / 20 * 7

冲量分析
平均速度: 0.2150 m/s
冲量计算区间: 0.07s - 0.20s
Ix (水平): 0.071724 N·s
Iz (垂直): 0.461739 N·s
|I| (矢量和): 0.467277 N·s
Fx区间最大力: 4.4124 N
Fz区间最大力: 17.6620 N
合力最大力: 17.8407 N