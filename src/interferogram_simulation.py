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
import math
from fringe_zernike_generator import FringeZernike  # 导入用户的Zernike生成类


class InterferogramSimulator:
    """干涉图仿真类：生成带圆掩码的相移干涉图（适配用户Fringe索引Zernike）"""

    def __init__(self,
                 img_size: tuple = (512, 512),
                 max_order: int = 16,  # Fringe最大索引
                 true_coeffs: np.ndarray = None,
                 phase_shifts: list = None,
                 I0: float = 1.0,
                 gamma: float = 0.8,
                 noise_std: float = 0.03):
        """
        初始化仿真参数（适配Fringe索引）
        :param img_size: 干涉图尺寸
        :param max_order: Fringe索引最大阶数（对应true_coeffs长度）
        :param true_coeffs: Fringe索引对应的真实系数（索引1~max_order）
        :param phase_shifts: 相移量列表
        :param I0: 光强直流分量
        :param gamma: 调制深度
        :param noise_std: 噪声标准差
        """
        self.img_size = img_size
        self.max_order = max_order
        self.true_coeffs = true_coeffs  # 长度=max_order，对应Fringe索引1~max_order
        self.phase_shifts = phase_shifts
        self.I0 = I0
        self.gamma = gamma
        self.noise_std = noise_std

        # 内部变量
        self.interferograms = None
        self.true_phase = None
        self.rho = None
        self.theta = None
        self.circle_mask = None
        self.zernike_generator = None  # 用户的FringeZernike实例

    def generate(self) -> tuple:
        """生成相移干涉图（核心方法：使用用户的Zernike生成真实相位）"""
        H, W = self.img_size
        min_size = min(H, W)

        # ========== 修复1：重新生成笛卡尔坐标网格（用于圆形掩码） ==========
        # 生成笛卡尔坐标网格（-1~1）
        x = np.linspace(-1, 1, min_size)
        y = np.linspace(-1, 1, min_size)
        xx, yy = np.meshgrid(x, y)
        # 计算极坐标（rho: 0~√2 → 归一化到0~1）
        self.rho = np.sqrt(xx ** 2 + yy ** 2)  # 极径（0~√2）
        self.theta = np.arctan2(yy, xx)  # 极角（-π~π）
        # 生成严格的圆形掩码（rho<=1为有效区域）
        self.circle_mask = (self.rho <= 1.0)
        # 归一化极径到0~1（仅圆形区域内有效）
        self.rho_normalized = np.where(self.circle_mask, self.rho, 0)

        # ========== 修复2：初始化用户的FringeZernike生成器（适配当前网格） ==========
        self.zernike_generator = FringeZernike(
            max_order=self.max_order,
            resolution=min_size
        )
        # 替换用户生成器的网格为当前笛卡尔/极坐标网格（关键修复）
        self.zernike_generator.x = xx
        self.zernike_generator.y = yy
        self.zernike_generator.rr = self.rho_normalized  # 归一化极径
        self.zernike_generator.tt = self.theta

        # ========== 修复3：生成真实相位（仅圆形区域有效） ==========
        self.true_phase = np.zeros((min_size, min_size), dtype=np.float64)
        for idx in range(1, self.max_order + 1):
            # 用用户的generate方法生成对应Fringe索引的Zernike多项式
            zernike_poly = self.zernike_generator.generate(idx)
            # 仅圆形区域累加相位
            self.true_phase += self.true_coeffs[idx - 1] * zernike_poly * self.circle_mask

        # 圆外相位设为NaN（强制空值）
        self.true_phase = np.where(self.circle_mask, self.true_phase, np.nan)

        # ========== 修复4：生成相移干涉图（仅圆形区域有信号） ==========
        M = len(self.phase_shifts)
        self.interferograms = np.full((M, min_size, min_size), np.nan, dtype=np.float64)
        for i, delta in enumerate(self.phase_shifts):
            # 仅圆形区域计算光强
            intensity = np.zeros((min_size, min_size), dtype=np.float64)
            intensity[self.circle_mask] = self.I0 * (1 + self.gamma * np.cos(self.true_phase[self.circle_mask] + delta))
            # 添加噪声（仅圆形区域）
            noise = np.random.normal(0, self.noise_std, (min_size, min_size))
            intensity_noisy = intensity + noise * self.circle_mask
            # 裁剪光强范围（0~2I0）
            intensity_noisy = np.clip(intensity_noisy, 0, 2 * self.I0)
            # 仅圆形区域赋值，圆外保持NaN
            self.interferograms[i][self.circle_mask] = intensity_noisy[self.circle_mask]

        # 调试输出：验证圆形掩码
        print(f"✅ 圆形掩码验证：有效像素数 = {np.sum(self.circle_mask)}, 总像素数 = {min_size * min_size}")
        print(f"✅ 真实相位NaN像素数 = {np.sum(np.isnan(self.true_phase))}")
        print(f"✅ 干涉图1 NaN像素数 = {np.sum(np.isnan(self.interferograms[0]))}")

        return self.interferograms, self.true_phase, self.rho, self.theta, self.circle_mask
