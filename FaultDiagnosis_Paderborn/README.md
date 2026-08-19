## 数据集说明与来源 (Dataset Attribution)

# 本项目使用了**帕德博恩大学轴承数据集 (Paderborn University Bearing Dataset)**，
# 这是一个在滚动轴承状态监测领域广泛使用的公开基准数据集。
# 下载地址：https://groups.uni-paderborn.de/kat/BearingDataCenter/

### 数据集简介

# 该数据集由帕德博恩大学（Paderborn University）的Lessmeier等人于2016年发布。数据集旨在
# 为数据驱动的滚动轴承故障诊断算法提供开发和验证平台。

# 数据集包含了来自32个轴承的**电机相电流信号**和**振动信号**。这些轴承分为三类：
# *   **健康轴承 (Healthy)**: 6个
# *   **内圈故障轴承 (Inner Race Fault)**: 11个
# *   **外圈故障轴承 (Outer Race Fault)**: 12个

# 其中，故障轴承又分为**人工损伤**（通过钻孔、电火花加工等方式制造）和**自然损伤**（通过加速寿命
# 试验产生）两类。数据在四种不同的工况下采集，转速范围为900至1500 rpm，负载扭矩为0.1至0.7 Nm，
# 径向力为400至1000 N。每个信号时长4秒，采样频率为64 kHz。

### 引用信息 (Citation)

# 如果你在研究中使用了该数据集，请引用以下论文(关于数据集的详细内容也请参考下面的论文)：
# "Lessmeier, C., Kimotho, J. K., Zimmer, D., & Sextro, W. (2016). Condition Monitoring 
# of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of 
# Electric Motors: A Benchmark Data Set for Data-Driven Classification. *PHM Society European 
# Conference*, 3(1). https://doi.org/10.36001/phme.2016.v3i1.1577"