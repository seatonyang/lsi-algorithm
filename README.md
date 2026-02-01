# Fringe Zernike 多项式自动生成与可视化工具

# Fringe Zernike Polynomial Generator

基于Fringe索引的Zernike多项式自动生成、数学表达式打印与论文风格可视化工具，专为光学检测、光刻等工程领域打造（区别于通用的Noll/Standard排序方案）。

## 📚 项目介绍

Zernike多项式是光学系统像差表征、成像质量评估的核心工具，目前公开实现多基于Noll/Standard排序，而光学检测、光刻等领域广泛使用的Fringe索引相关工具极少。本项目基于论文《Straightforward path to Zernike polynomials》实现了Fringe索引Zernike多项式的全自动生成与专业可视化，无需手动编写任何阶数的多项式表达式，同时支持论文规范的阶梯图绘制与数学表达式打印。

## ✨ 核心功能

1. 自动生成任意阶数（自定义`max_order`）的Fringe Zernike多项式，无需手动编写各阶表达式

2. 支持打印单个/所有阶数的Zernike多项式数学表达式（径向部分+角向部分+完整形式，系数已化简为具体数值）

3. 绘制论文规范的阶梯图（按`s=m+k`分组、右对齐，默认jet色彩映射，红色边框标记m=0项）

4. 支持单个多项式单独绘制，可自定义网格分辨率、色彩映射等参数

## 🎯 核心特性

- 严格遵循Fringe索引规则，适配光学检测、光刻等工程领域需求（区别于Noll/Standard排序）

- 多项式定义完全匹配论文《Straightforward path to Zernike polynomials》

- 完善的输入验证与错误处理，支持高分辨率网格生成，适配学术与工程仿真场景

## 🛠️ 环境依赖与安装

### 依赖库

- `numpy`：数值计算

- `matplotlib`：可视化绘图

### 安装命令

```Bash

pip install numpy matplotlib
```

## 🚀 快速开始

### 1. 克隆代码仓

```Bash

git clone <你的代码仓地址>
cd <代码仓目录>
```

### 2. 运行主脚本

直接运行Python文件，按提示输入需要生成的阶数（如64）即可自动完成表达式打印与可视化：

```Bash

python fringe_zernike_auto_generate_visualization.py
```

## 📖 详细使用示例

### 1. 初始化生成器

```Python

# 生成64阶Fringe Zernike多项式，网格分辨率128x128
from fringe_zernike_auto_generate_visualization import FringeZernike

# 初始化（max_order=阶数，resolution=网格分辨率）
zernike_gen = FringeZernike(max_order=64, resolution=128)
```

### 2. 打印多项式数学表达式

```Python

# 打印所有阶数的表达式（系数已化简）
zernike_gen.print_zernike_expression()

# 仅打印第4阶（Focus）的表达式
zernike_gen.print_zernike_expression(index=4)
```

### 3. 绘制单个多项式

```Python

# 绘制第9阶（球差），使用jet色彩映射
zernike_gen.plot_single(index=9, cmap="jet")
```

### 4. 绘制论文风格阶梯图

```Python

# 自动调整图大小，生成右对齐阶梯图（默认jet色彩）
zernike_gen.plot_all_stepwise(cmap="jet")

# 高阶数（如64阶）可手动指定图大小，保证显示效果
zernike_gen.plot_all_stepwise(figsize=(45, 35), cmap="jet")
```

## ⚙️ 关键参数说明

|参数名|作用|默认值|可选值|注意事项|
|---|---|---|---|---|
|`max_order`|生成的最大Fringe索引（阶数）|-|任意正整数（如36、64、100）|阶数越高，计算/绘图耗时越长|
|`resolution`|网格分辨率（影响多项式精度与绘图清晰度）|128|64、128、256、512|分辨率过高（如512）可能占用较多内存|
|`cmap`|色彩映射方案|jet|viridis、plasma、RdBu_r等|推荐使用jet（符合论文/工程常用风格）|
|`figsize`|图像尺寸（宽, 高）|自动计算|元组（如(45, 35)）|高阶数建议手动指定，避免子图挤压|
## ❓ 常见问题

### Q1: 运行报错 `TypeError: GridSpec.__init__() got an unexpected keyword argument 'rows'`

A1: 代码已修复该问题（GridSpec参数应为`nrows/ncols`而非`rows/cols`），直接使用最新代码即可。

### Q2: 阶数过高（如>100）导致内存不足/绘图卡顿

A2: 降低网格分辨率（如从256改为128），或分批次生成/绘制多项式（先打印表达式，再单独绘制关键阶数）。

### Q3: 表达式中系数显示异常

A3: 代码已自动化简阶乘为具体数值，若仍有异常，检查Python版本（推荐3.8+），或确认输入的阶数/索引在有效范围。

## 📄 许可证

本项目采用MIT许可证开源，可自由用于学术研究与工程开发，如需商用请注明出处。

## 🙏 致谢

本项目基于论文《Straightforward path to Zernike polynomials》实现，感谢原作者的研究成果；同时感谢numpy、matplotlib社区提供的优秀工具库。
