# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 15:04
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : zernike_fitting.py
@Software    : PyCharm
@Description : 
"""

import numpy as np
from scipy.linalg import lstsq
from grid_utils import GridGenerator
from fringe_zernike_generator import FringeZernike


class ZernikeFitter:
    """
    重构后：
    1. 依赖统一网格工具，极径处理完全交给Zernike多项式
    2. 移除所有手动修改极径的代码，鲁棒性提升
    3. 拟合逻辑更简洁，仅专注于最小二乘拟合
    """

    def __init__(self,
                 grid: GridGenerator,
                 unwrapped_phase: np.ndarray,
                 max_order: int):
        """
        :param grid: 统一网格工具实例
        :param unwrapped_phase: 解包裹相位（圆外NaN）
        :param max_order: Fringe最大索引
        """
        self.grid = grid
        self.unwrapped_phase = unwrapped_phase
        self.max_order = max_order

        # 拟合结果
        self.fitted_coeffs = None
        self.fitted_phase = None
        self.zernike_generator = None

    def fit(self) -> tuple:
        """最小二乘拟合Zernike系数"""
        # 1. 筛选有效像素：单位圆内 + 非NaN
        valid_mask = self.grid.circle_mask & (~np.isnan(self.unwrapped_phase))
        if np.sum(valid_mask) == 0:
            raise ValueError("无有效像素用于拟合！")
        K = np.sum(valid_mask)
        print(f"✅ 拟合有效像素数 = {K}")

        # 2. 初始化Zernike生成器（复用统一网格）
        self.zernike_generator = FringeZernike(
            max_order=self.max_order,
            grid=self.grid
        )

        # 3. 构建基底矩阵A（K×max_order）
        A = np.zeros((K, self.max_order), dtype=np.float64)
        for idx in range(1, self.max_order + 1):
            z_poly = self.zernike_generator.generate(idx)
            A[:, idx - 1] = z_poly[valid_mask]

        # 4. 最小二乘拟合
        y = self.unwrapped_phase[valid_mask]
        self.fitted_coeffs, _, _, _ = lstsq(A, y, cond=None)

        # 5. 生成拟合相位
        self.fitted_phase = np.zeros((self.grid.size, self.grid.size))
        for idx in range(1, self.max_order + 1):
            z_poly = self.zernike_generator.generate(idx)
            self.fitted_phase += self.fitted_coeffs[idx - 1] * z_poly
        # 标准化相位：圆外NaN
        self.fitted_phase = self.grid.get_valid_phase(self.fitted_phase)

        return self.fitted_coeffs, self.fitted_phase


# ------------------------------
# 自验证main函数
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("ZernikeFitter 自验证开始")
    print("=" * 80)

    # 1. 生成统一网格
    grid = GridGenerator(size=512)
    print(f"✅ 统一网格生成成功（size={grid.size}）")

    # 2. 生成测试数据（模拟解包裹相位）
    max_order = 64
    true_coeffs = np.zeros(max_order)
    true_coeffs[1] = 0.6  # Tilt x
    true_coeffs[3] = 1.2  # Focus
    true_coeffs[4] = 0.9  # Astigmatism x

    # 生成真实相位（用Zernike生成器）
    zernike_gen = FringeZernike(max_order=max_order, grid=grid)
    true_phase = np.zeros((grid.size, grid.size))
    for idx in range(1, max_order + 1):
        z_poly = zernike_gen.generate(idx)
        true_phase += true_coeffs[idx - 1] * z_poly
    true_phase = grid.get_valid_phase(true_phase)
    # 添加少量噪声
    true_phase[grid.circle_mask] += np.random.normal(0, 0.01, np.sum(grid.circle_mask))

    print(f"✅ 测试数据生成成功：")
    print(f"   真实相位形状：{true_phase.shape}")
    print(f"   真实系数：{true_coeffs}")

    # 3. 初始化拟合器
    try:
        fitter = ZernikeFitter(
            grid=grid,
            unwrapped_phase=true_phase,
            max_order=max_order
        )
        print("✅ 拟合器初始化成功")
    except Exception as e:
        print(f"❌ 拟合器初始化失败：{e}")
        exit(1)

    # 4. 执行拟合
    try:
        fitted_coeffs, fitted_phase = fitter.fit()
        print(f"✅ 拟合成功，拟合系数：{fitted_coeffs}")

        # 计算拟合误差
        coeff_rmse = np.sqrt(np.mean((true_coeffs - fitted_coeffs) ** 2))
        phase_rmse = np.sqrt(np.nanmean((true_phase - fitted_phase) ** 2))
        print(f"✅ 系数拟合RMSE：{coeff_rmse:.6f}（应<0.05）")
        print(f"✅ 相位拟合RMSE：{phase_rmse:.6f}（应<0.02）")
    except Exception as e:
        print(f"❌ 拟合失败：{e}")
        exit(1)

    # 5. 可视化验证
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['Arial']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        # 真实相位
        ax1.imshow(true_phase, cmap='jet')
        ax1.set_title('True Phase', fontsize=12)
        ax1.axis('off')
        # 拟合相位
        ax2.imshow(fitted_phase, cmap='jet')
        ax2.set_title('Fitted Phase', fontsize=12)
        ax2.axis('off')
        # 系数对比
        coeff_idx = np.arange(1, max_order + 1)
        ax3.bar(coeff_idx - 0.2, true_coeffs, 0.4, label='True', alpha=0.8)
        ax3.bar(coeff_idx + 0.2, fitted_coeffs, 0.4, label='Fitted', alpha=0.8)
        ax3.set_xlabel('Fringe Index')
        ax3.set_ylabel('Coefficient')
        ax3.set_title('Coefficient Comparison')
        ax3.legend()
        # 拟合误差
        error = np.abs(true_phase - fitted_phase)
        ax4.imshow(error, cmap='jet')
        ax4.set_title(f'Fitting Error (RMSE={phase_rmse:.4f})')
        ax4.axis('off')

        plt.tight_layout()
        plt.show()
        print("✅ 可视化验证成功")
    except Exception as e:
        print(f"❌ 可视化验证失败：{e}")

    print("=" * 80)
    print("ZernikeFitter 自验证完成")
    print("=" * 80)