# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 14:58
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : interferogram_simulation.py
@Software    : PyCharm
@Description : 
"""

import numpy as np
from grid_utils import GridGenerator
from fringe_zernike_generator import FringeZernike


class InterferogramSimulator:
    """
    重构后：
    1. 依赖统一网格工具，不再手动生成网格/掩码
    2. 极径处理完全交给Zernike多项式，无需手动修改
    3. 逻辑更简洁，仅专注于干涉图生成
    """

    def __init__(self,
                 grid: GridGenerator,
                 max_order: int,
                 true_coeffs: np.ndarray,
                 phase_shifts: list,
                 I0: float = 1.0,
                 gamma: float = 0.8,
                 noise_std: float = 0.03):
        """
        :param grid: 统一网格工具实例
        :param max_order: Fringe最大索引
        :param true_coeffs: Fringe索引对应的真实系数（1~max_order）
        :param phase_shifts: 相移量列表
        :param I0: 光强直流分量
        :param gamma: 调制深度
        :param noise_std: 噪声标准差
        """
        self.grid = grid
        self.max_order = max_order
        self.true_coeffs = true_coeffs
        self.phase_shifts = phase_shifts
        self.I0 = I0
        self.gamma = gamma
        self.noise_std = noise_std

        # 输出变量
        self.interferograms = None
        self.true_phase = None
        self.zernike_generator = None

    def generate(self) -> tuple:
        """生成相移干涉图（核心方法）"""
        # 1. 初始化Zernike生成器（复用统一网格）
        self.zernike_generator = FringeZernike(
            max_order=self.max_order,
            grid=self.grid
        )

        # 2. 生成真实相位（仅单位圆内有效）
        self.true_phase = np.zeros((self.grid.size, self.grid.size))
        for idx in range(1, self.max_order + 1):
            z_poly = self.zernike_generator.generate(idx)
            self.true_phase += self.true_coeffs[idx - 1] * z_poly
        # 标准化相位：圆外NaN
        self.true_phase = self.grid.get_valid_phase(self.true_phase)

        # 3. 生成相移干涉图
        M = len(self.phase_shifts)
        self.interferograms = np.full((M, self.grid.size, self.grid.size), np.nan)

        for i, delta in enumerate(self.phase_shifts):
            # 仅单位圆内计算光强
            intensity = np.zeros((self.grid.size, self.grid.size))
            valid_phase = self.true_phase[self.grid.circle_mask]
            intensity[self.grid.circle_mask] = self.I0 * (1 + self.gamma * np.cos(valid_phase + delta))

            # 添加噪声（仅单位圆内）
            noise = np.random.normal(0, self.noise_std, intensity.shape)
            intensity[self.grid.circle_mask] += noise[self.grid.circle_mask]

            # 裁剪光强范围
            intensity = np.clip(intensity, 0, 2 * self.I0)

            # 仅单位圆内赋值，圆外NaN
            self.interferograms[i][self.grid.circle_mask] = intensity[self.grid.circle_mask]

        # 调试输出
        valid_pixels = np.sum(self.grid.circle_mask)
        print(f"✅ 圆形掩码验证：有效像素数 = {valid_pixels}, 总像素数 = {self.grid.size ** 2}")
        print(f"✅ 真实相位NaN像素数 = {np.sum(np.isnan(self.true_phase))}")
        print(f"✅ 干涉图1 NaN像素数 = {np.sum(np.isnan(self.interferograms[0]))}")

        return self.interferograms, self.true_phase


# ------------------------------
# 自验证main函数
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("InterferogramSimulator 自验证开始")
    print("=" * 80)

    # 1. 生成统一网格
    grid = GridGenerator(size=256)
    print(f"✅ 统一网格生成成功（size={grid.size}）")

    # 2. 配置参数
    max_order = 8
    true_coeffs = np.zeros(max_order)
    true_coeffs[1] = 0.6  # Tilt x
    true_coeffs[3] = 1.2  # Focus
    phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # 3. 初始化仿真器
    try:
        simulator = InterferogramSimulator(
            grid=grid,
            max_order=max_order,
            true_coeffs=true_coeffs,
            phase_shifts=phase_shifts,
            noise_std=0.01
        )
        print(f"✅ 仿真器初始化成功")
    except Exception as e:
        print(f"❌ 仿真器初始化失败：{e}")
        exit(1)

    # 4. 生成干涉图
    try:
        interferograms, true_phase = simulator.generate()
        print(f"✅ 干涉图生成成功，形状：{interferograms.shape}")
        print(f"✅ 真实相位形状：{true_phase.shape}")
    except Exception as e:
        print(f"❌ 干涉图生成失败：{e}")
        exit(1)

    # 5. 验证关键指标
    print(f"\n📊 掩码验证：")
    print(f"   有效像素数：{np.sum(grid.circle_mask)}")
    print(f"   掩码覆盖率：{np.sum(grid.circle_mask) / (grid.size ** 2) * 100:.2f}%")

    print(f"\n📊 干涉图验证：")
    for i in range(len(phase_shifts)):
        non_nan = np.sum(~np.isnan(interferograms[i]))
        print(f"   干涉图{i + 1} 非NaN像素数：{non_nan}（应等于有效像素数）")
        print(f"   干涉图{i + 1} 光强范围：{np.nanmin(interferograms[i]):.4f} ~ {np.nanmax(interferograms[i]):.4f}")

    # 6. 可视化验证
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['Arial']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(grid.circle_mask, cmap='gray')
        ax1.set_title('Circle Mask', fontsize=12)
        ax1.axis('off')

        ax2.imshow(interferograms[0], cmap='jet', vmin=0, vmax=2)
        ax2.set_title('Interferogram 1 (Shift=0π)', fontsize=12)
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        print("✅ 可视化验证成功")
    except Exception as e:
        print(f"❌ 可视化验证失败：{e}")

    print("=" * 80)
    print("InterferogramSimulator 自验证完成")
    print("=" * 80)