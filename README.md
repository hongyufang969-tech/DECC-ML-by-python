
# DECC-ML – 高频次随机分组协同进化优化器

**Cooperative Co-evolution for Large Scale Optimization Through More Frequent Random Grouping**  
**高频次随机分组大规模全局优化算法 Python 复现**

---

## 算法简介 / Overview

DECC-ML 是一种用于求解大规模全局优化问题（LSGO）的协同进化算法。它是对传统 MLCC 和 DECC-G 框架的深度优化与精简，专门针对高维复杂交互变量问题。  
DECC-ML is a cooperative co-evolution algorithm designed for large-scale global optimization (LSGO) problems. It simplifies and improves upon the traditional MLCC and DECC-G frameworks, specifically targeting high-dimensional problems with complex interacting variables.

算法核心思想 / Core ideas:

- **高频次随机分组 (More Frequent Random Grouping)**：通过仅让子优化器运行单次迭代，极大化随机分组频率，显著提升捕获多交互变量的概率[cite: 1]。  
  Maximizes random grouping frequency by running the sub-optimizer for only a single iteration, significantly increasing the probability of capturing multiple interacting variables[cite: 1].
- **均匀自适应子空间 (Uniform Self-Adaptation)**：使用均匀随机策略从预设集合 $\{5, 10, 25, 50, 100\}$ 中动态选择分组大小，机制简单且高效[cite: 1]。  
  Dynamically selects subcomponent sizes from a predefined set $\{5, 10, 25, 50, 100\}$ using a uniform random strategy, which is simple and efficient[cite: 1].
- **移除低效权重 (Removal of Adaptive Weighting)**：彻底移除了原 DECC-G 中低效的自适应权重机制，将节省的大量算力用于更有效的子空间探索[cite: 1]。  
  Completely removes the ineffective adaptive weighting mechanism from the original DECC-G, reallocating saved computational power to more effective subcomponent exploration[cite: 1].
- **SaNSDE 子优化器 (SaNSDE Sub-optimizer)**：采用自适应差分进化算法（SaNSDE）独立优化各子组件，并通过上下文向量（Context Vector）协调全局适应度[cite: 3]。  
  Employs Self-adaptive Differential Evolution (SaNSDE) to independently optimize subcomponents, coordinated globally via a Context Vector[cite: 3].

> 参考论文 / Reference: Omidvar et al., *Cooperative Co-evolution for Large Scale Optimization Through More Frequent Random Grouping*, CEC 2010[cite: 1].

---

## 文件结构 / File Structure

```text
DECC-ML_Project/
├── optimizer.py              # DECC-ML 主算法引擎实现 / Core DECC-ML engine implementation
├── main_test.py              # 运行示例脚本与基准测试控制台 / Main runner & benchmark console
└── README.md                 # 本文档 / This document
```

---

## 环境依赖 / Requirements

```text
Python >= 3.8
numpy
opfunu        # 必须，用于提供 CEC'2008 测试函数集
```

---

## 快速开始 / Quick Start

所有命令在项目根目录下运行。  
Run all commands from the project root directory.

**安装依赖并运行全量测试**  
**Install dependencies and run full benchmark:**

```bash
pip install numpy opfunu
python main_test.py
```

**测试单个函数或缩减算力（冒烟测试）**  
**Test a single function or reduce evaluations (Smoke Test):**

直接修改 `main_test.py` 顶部的配置常量：
Modify the configuration constants at the top of `main_test.py`:

```python
# 修改 main_test.py / Modify main_test.py
RUNS_PER_FUNCTION = 1           # 只跑 1 次 / Run only 1 time
MAX_EVALUATIONS = 50000         # 将评估次数降至 5w / Reduce FEs to 50k
TARGET_FUNCTIONS = [F12008]     # 仅测试 F1 函数 / Test only F1 function
```

然后再执行：

```bash
python main_test.py
```

---

## 接口说明 / Interface

DECC-ML 的接口设计极其简洁，严格遵循“优化器（Optimizer）”的设计模式，可无缝集成到任何自定义的适应度评估框架中。  
The DECC-ML interface is designed to be highly concise, strictly following the "Optimizer" design pattern, and can be seamlessly integrated into any custom fitness evaluation framework.

```python
import numpy as np
from optimizer import DECCMLOptimizer

# 1. 定义您的目标函数 / Define your objective function
def custom_objective_function(x):
    return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)) # Rastrigin

# 2. 初始化优化器 / Initialize the optimizer
optimizer = DECCMLOptimizer(
    objective=custom_objective_function,
    dimension=1000,
    lower_bound=np.full(1000, -5.12),
    upper_bound=np.full(1000, 5.12),
    max_evaluations=5_000_000
)

# 3. 启动求解 / Start solving
best_fitness, best_solution = optimizer.solve(log_interval=500_000)

print(f"Global Best Fitness: {best_fitness:.6e}")
```

---

## 算法参数说明 / Algorithm Parameters

`DECCMLOptimizer` 的 `__init__` 方法中可配置以下超参数（核心超参数已内置为类常量）：  
The following hyperparameters can be configured in the `__init__` method of `DECCMLOptimizer` (core hyperparameters are built-in as class constants):

| 参数 / Parameter | 类型 / Type | 说明 / Description |
|---|---|---|
| `dimension` | int | 问题维度 / Problem dimension |
| `max_evaluations` | int | 最大函数评估次数 / Maximum function evaluations |
| `population_size` | int | **内置默认: 50**。全局种群个体数 / Built-in default: 50. Population size |
| `s_pool` | tuple | **内置默认: (5, 10, 25, 50, 100)**。分组维度备选池 / Built-in default. Subcomponent size pool |
| `learning_period` | int | **内置默认: 50**。SaNSDE 自适应策略的学习周期 / Built-in default: 50. Learning period for SaNSDE |

---

## 输出说明 / Output

`optimizer.solve()` 返回一个元组：`(最优适应度值, 最优解向量)`。  
`optimizer.solve()` returns a tuple: `(Best fitness value, Best solution vector)`.

若使用配套的 `main_test.py` 运行，系统将自动生成极具学术规范的纯文本统计报告（包含 Mean Error, Std Dev, Variance）：  
If running with the provided `main_test.py`, the system will automatically generate a highly academic plain-text statistical report:

```text
=========================================================================================================
DECC-ML (SaNSDE) 最终实验数据报告 / Final Experimental Data Report
总耗时 / Total Time: 0.12 h | 评估次数 / FEs: 5000000 | 运行 / Runs: 5
=========================================================================================================
Function                                 | Mean Error      | Std Dev         | Variance       
---------------------------------------------------------------------------------------------------------
F1: Shifted Sphere Function              | 0.0000e+00      | 0.0000e+00      | 0.0000e+00     
=========================================================================================================
```

---

## 扩展与自定义 / Customization

- **修改子空间池 `s_pool`**：可在 `optimizer.py` 的类常量区修改 `DEFAULT_S_POOL`，以适应具有特定结构重叠特征的问题维度。  
  Modify `DEFAULT_S_POOL` in `optimizer.py` to adapt to problem dimensions with specific structural overlapping features.
- **替换底层的 SaNSDE 机制**：定位到 `_mutate_and_crossover()` 和 `_update_adaptation_stats()`，您可以将其替换为 JADE、SHADE 等更现代的自适应差分进化变体，而无需改动外层的高频分组控制流。  
  Locate `_mutate_and_crossover()` and `_update_adaptation_stats()` to replace them with more modern adaptive DE variants like JADE or SHADE, without changing the outer high-frequency grouping control flow.
```
