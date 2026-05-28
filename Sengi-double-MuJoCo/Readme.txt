简化双关节象鼩-MuJoCo

===============================================================================
1、260527_1
XML配置
friction="0.45 0.01 0.01" solimp="0.9 0.95 0.01" solref="0.01 0.5"
<body name="base_link" pos="0 0 0.10">

参数搜索
TOR = 0.5
TOR_SPINE = TOR*1.5

仿真
OPTIMAL_A_LEGH_HIP = 1.4
OPTIMAL_A_LEGH_KNEE = 0.8
OPTIMAL_PHASE_LAG = -np.pi / 20 * 0

冲量分析
平均速度: 0.4764 m/s
冲量计算区间: 0.10s - 0.20s
Ix (水平): 0.149534 N·s
Iz (垂直): 0.671689 N·s
|I| (矢量和): 0.688132 N·s
Fx区间最大力: 5.7078 N
Fz区间最大力: 34.4184 N
合力最大力: 34.4738 N