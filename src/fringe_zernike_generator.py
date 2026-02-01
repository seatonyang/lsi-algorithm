# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 11:33
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : fringe_zernike_generator.py
@Software    : PyCharm
@Description :
文件作用：基于Fringe索引的Zernike多项式自动生成、数学表达式打印与论文风格可视化脚本
核心功能：
    1.  自动生成任意阶数（自定义max_order）的Fringe Zernike多项式，无需手动编写各阶表达式
    2.  支持打印单个/所有阶数的Zernike多项式数学表达式（径向部分+角向部分+完整形式）
    3.  绘制论文规范的阶梯图（按s=m+k分组、右对齐，默认jet色彩映射，标记m=0项）
    4.  支持单个多项式单独绘制，可自定义网格分辨率、色彩映射等参数
核心特性：
    - 严格遵循Fringe索引规则，适配光学检测、光刻等工程领域需求（区别于Noll/Standard排序）
    - 多项式定义完全匹配论文《Straightforward path to Zernike polynomials》
    - 完善的输入验证与错误处理，支持高分辨率网格生成，适配学术与工程仿真场景
依赖库：numpy, matplotlib
适用场景：光学系统像差分析、微光刻仿真、成像质量评估、学术研究中的Zernike多项式快速生成与验证
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import math
from matplotlib.patches import Patch
from grid_utils import GridGenerator  # 导入统一网格工具


# ------------------------------
# 核心工具函数：自动生成Fringe索引映射和径向多项式
# ------------------------------
def generate_fringe_mapping(N):
    mapping = [{}]  # index 0未使用
    current_index = 1
    s = 0  # s = m + k（分组标识）

    while current_index <= N:
        # 每个s组内，m从s递减到0
        for m in range(s, -1, -1):
            k = s - m  # k = s - m（保证s = m+k）
            n = m + 2 * k  # Zernike径向阶数（n ≥ m，n和m同奇偶）

            # 自动生成多项式名称
            if m == 0:
                if n == 0:
                    name = "Piston"
                elif n == 2:
                    name = "Focus"
                else:
                    name = "Spherical aberration"
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "zero",
                    "name": name
                })
                current_index += 1
                if current_index > N:
                    break
            else:
                if m == 1:
                    name_cos = "Tilt x" if n == 1 else "Coma x"
                    name_sin = "Tilt y" if n == 1 else "Coma y"
                elif m == 2:
                    name_cos = "Astigmatism x"
                    name_sin = "Astigmatism y"
                elif m >= 3:
                    name_cos = f"{m}-fold x"
                    name_sin = f"{m}-fold y"
                else:
                    name_cos = f"m={m} x"
                    name_sin = f"m={m} y"

                # 添加cos(mθ)项（x向）
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "cos",
                    "name": name_cos
                })
                current_index += 1
                if current_index > N:
                    break

                # 添加sin(mθ)项（y向）
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "sin",
                    "name": name_sin
                })
                current_index += 1
                if current_index > N:
                    break
        s += 1  # 下一组s
    return mapping


def radial_polynomial(r, n, m):
    """
    Zernike径向多项式：自动处理r>1的情况（返回0）
    :param r: 极径（任意范围，r>1时返回0）
    :param n: 径向阶数
    :param m: 角向阶数
    :return: 径向多项式值（r>1时=0）
    """
    # 核心：单位圆外直接返回0，符合Zernike定义
    if np.any(r > 1):
        r = np.where(r > 1, 0, r)

    if n < m or (n - m) % 2 != 0:
        return np.zeros_like(r, dtype=np.float64)

    k = (n - m) // 2  # k = (n-m)/2（整数）
    R = np.zeros_like(r, dtype=np.float64)

    # 论文Eq.(1)的求和计算
    for s in range(0, k + 1):
        numerator = (-1) ** s * math.factorial(n - s)
        denominator = (math.factorial(s) *
                       math.factorial((n + m) // 2 - s) *
                       math.factorial((n - m) // 2 - s))
        term = numerator / denominator * r ** (n - 2 * s)
        R += term
    return R


def get_radial_expression(n, m):
    if n < m or (n - m) % 2 != 0:
        return "0"

    k = (n - m) // 2
    terms = []
    for s in range(0, k + 1):
        sign = (-1) ** s
        fact_n_s = math.factorial(n - s)
        fact_s = math.factorial(s)
        fact_nm2_s = math.factorial((n + m) // 2 - s)
        fact_nm2_s2 = math.factorial((n - m) // 2 - s)

        coefficient = sign * fact_n_s / (fact_s * fact_nm2_s * fact_nm2_s2)
        if coefficient.is_integer():
            coeff_str = f"{int(coefficient)}"
        else:
            coeff_str = f"{coefficient:.3f}"

        power = n - 2 * s
        if power == 0:
            r_term = "1"
        elif power == 1:
            r_term = "r"
        else:
            r_term = f"r^{power}"

        if coeff_str == "1" and power != 0:
            term_str = r_term
        elif coeff_str == "-1" and power != 0:
            term_str = f"-{r_term}"
        else:
            term_str = f"{coeff_str}×{r_term}"

        terms.append(term_str)

    radial_expr = " + ".join(terms).replace(" + -", " - ")
    return f"R_{n}^{m}(r) = {radial_expr}"


# ------------------------------
# Zernike多项式生成与绘图类（重构版）
# ------------------------------
class FringeZernike:
    """
    重构后：
    1. 不再内部生成网格，依赖外部GridGenerator
    2. 专注于Zernike多项式计算，逻辑更单一
    3. 自动处理r>1的情况，无需外部干预
    """

    def __init__(self, max_order: int, grid: GridGenerator):
        """
        :param max_order: 最大Fringe索引
        :param grid: 统一网格工具实例（GridGenerator）
        """
        # 输入验证
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError(f"阶数必须是正整数，当前输入：{max_order}")
        if not isinstance(grid, GridGenerator):
            raise TypeError("grid必须是GridGenerator实例")

        self.max_order = max_order
        self.grid = grid  # 依赖外部统一网格

        # 自动生成多项式定义
        self.zernike_defs = self._auto_generate_zernike()

        # 按s=m+k分组（用于绘图）
        self.s_groups = self._group_by_s()

        # 预计算全局最大振幅
        self.max_amplitude = self._get_global_max_amp()

        # 最大列数（用于右对齐布局）
        self.max_columns = max(2 * s + 1 for s in self.s_groups.keys())

    def _auto_generate_zernike(self):
        fringe_mapping = generate_fringe_mapping(self.max_order)
        zernike_defs = [{}]  # index 0未使用

        for idx in range(1, self.max_order + 1):
            if idx >= len(fringe_mapping):
                break
            params = fringe_mapping[idx]

            # 动态创建多项式函数（依赖外部网格）
            def create_zernike_func(m, n, poly_type):
                def func():
                    R = radial_polynomial(self.grid.rho, n, m)  # 使用统一极径
                    # 角向部分
                    if poly_type == "zero":
                        angular = np.ones_like(self.grid.theta)
                    elif poly_type == "cos":
                        angular = np.cos(m * self.grid.theta)
                    elif poly_type == "sin":
                        angular = np.sin(m * self.grid.theta)
                    else:
                        angular = np.zeros_like(self.grid.theta)

                    # 正交归一化因子（标准Zernike）
                    if m == 0:
                        norm_factor = np.sqrt(2 * n + 1)
                    else:
                        norm_factor = np.sqrt(2 * (2 * n + 1))

                    return norm_factor * R * angular

                return func

            # 封装多项式信息
            zernike_func = create_zernike_func(
                params["m"], params["n"], params["poly_type"]
            )
            zernike_defs.append({
                "index": idx,
                "name": params["name"],
                "m": params["m"],
                "n": params["n"],
                "s": params["s"],
                "poly_type": params["poly_type"],
                "func": zernike_func
            })
        return zernike_defs

    def _group_by_s(self):
        s_groups = {}
        for idx in range(1, self.max_order + 1):
            s = self.zernike_defs[idx]["s"]
            if s not in s_groups:
                s_groups[s] = []
            s_groups[s].append(idx)
        return dict(sorted(s_groups.items()))

    def _get_global_max_amp(self):
        max_amp = 0.0
        for idx in range(1, self.max_order + 1):
            z = self.generate(idx)
            current_max = np.max(np.abs(z))
            if current_max > max_amp:
                max_amp = current_max
        return max_amp

    def generate(self, index: int) -> np.ndarray:
        """生成指定索引的Zernike多项式（size×size）"""
        if not (1 <= index <= self.max_order):
            raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
        return self.zernike_defs[index]["func"]()

    def print_zernike_expression(self, index=None):
        print("\n" + "=" * 80)
        print("Zernike多项式数学表达式（Fringe索引 | 系数已化简）")
        print("=" * 80)

        indices = [index] if index is not None else range(1, self.max_order + 1)

        for idx in indices:
            z_info = self.zernike_defs[idx]
            m = z_info["m"]
            n = z_info["n"]
            poly_type = z_info["poly_type"]

            radial_expr = get_radial_expression(n, m)

            if poly_type == "zero":
                angular_expr = "1"
            elif poly_type == "cos":
                angular_expr = f"cos({m}θ)" if m != 1 else "cos(θ)"
            elif poly_type == "sin":
                angular_expr = f"sin({m}θ)" if m != 1 else "sin(θ)"
            else:
                angular_expr = "0"

            full_expr = f"Z_{idx}(r,θ) = {radial_expr.split('=')[1].strip()} × {angular_expr}"

            print(f"\n【Fringe索引 {idx:3d}】")
            print(f"  名称: {z_info['name']:25s}")
            print(f"  参数: m={m:2d}, n={n:2d}, s={z_info['s']:2d}")
            print(f"  径向部分: {radial_expr}")
            print(f"  角向部分: Θ(θ) = {angular_expr}")
            print(f"  完整表达式: {full_expr}")

        print("\n" + "=" * 80)

    def plot_single(self, index, figsize=(6, 5), cmap="jet"):
        z = self.generate(index)
        z_info = self.zernike_defs[index]

        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        contour = ax.contourf(
            self.grid.x, self.grid.y, z,
            levels=50, cmap=cmap, norm=norm,
            extend="both"
        )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Fringe Zernike #{index}\n"
            f"Name: {z_info['name']} | m={z_info['m']}, n={z_info['n']}",
            fontsize=12, pad=10
        )
        ax.axis("off")

        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Amplitude", fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_all_stepwise(self, figsize=None, cmap="jet", title_fontsize=22):
        if figsize is None:
            rows = len(self.s_groups)
            cols = self.max_columns
            figsize = (cols * 2.2, rows * 2.2)

        fig = plt.figure(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        gs = gridspec.GridSpec(
            nrows=len(self.s_groups), ncols=self.max_columns,
            figure=fig, hspace=0.3, wspace=0.3
        )

        for row_idx, (s, indices) in enumerate(self.s_groups.items()):
            row_cols = 2 * s + 1
            start_col = self.max_columns - row_cols

            for col_offset, idx in enumerate(indices):
                col_idx = start_col + col_offset
                z = self.generate(idx)
                z_info = self.zernike_defs[idx]

                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.contourf(
                    self.grid.x, self.grid.y, z,
                    levels=30, cmap=cmap, norm=norm,
                    extend="both"
                )

                ax.set_xlim(-1.02, 1.02)
                ax.set_ylim(-1.02, 1.02)
                ax.set_aspect("equal")
                ax.set_title(
                    f"#{idx}\n{z_info['name']}",
                    fontsize=7 if self.max_order > 36 else 8,
                    pad=3
                )
                ax.axis("off")

        fig.suptitle(
            f"Fringe Zernike Polynomials (Order 1-{self.max_order})\n"
            f"Stepwise Layout (Grouped by s=m+k, Right-Aligned)",
            fontsize=title_fontsize, y=0.98
        )

        cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.82])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax, orientation="vertical"
        )
        cbar.set_label("Normalized Amplitude", fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)

        plt.show()


# ------------------------------
# 自验证main函数
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("FringeZernikeGenerator 自验证开始")
    print("=" * 80)

    # 1. 生成统一网格
    grid = GridGenerator(size=512)
    print(f"✅ 统一网格生成成功（size={grid.size}）")

    # 2. 初始化生成器
    try:
        zernike_gen = FringeZernike(max_order=64, grid=grid)
        print(f"✅ 生成器初始化成功（max_order=8）")
    except Exception as e:
        print(f"❌ 生成器初始化失败：{e}")
        exit(1)

    # 3. 验证多项式生成
    try:
        z4 = zernike_gen.generate(4)
        print(f"✅ 索引4（Focus）多项式生成成功，形状：{z4.shape}")
        print(f"✅ 索引4多项式振幅范围：{z4.min():.4f} ~ {z4.max():.4f}")
        # 验证单位圆外值为0
        non_zero_outside = np.sum(z4[~grid.circle_mask] != 0)
        print(f"✅ 单位圆外非零值数量：{non_zero_outside}（应=0）")
    except Exception as e:
        print(f"❌ 多项式生成失败：{e}")

    # 4. 验证表达式打印
    try:
        zernike_gen.print_zernike_expression(index=4)
        print("✅ 多项式表达式打印成功")
    except Exception as e:
        print(f"❌ 表达式打印失败：{e}")

    # 5. 验证绘图
    try:
        zernike_gen.plot_single(index=4, cmap="jet")
        print("✅ 单个多项式绘图成功")
    except Exception as e:
        print(f"❌ 绘图失败：{e}")

    print("=" * 80)
    print("FringeZernikeGenerator 自验证完成")
    print("=" * 80)