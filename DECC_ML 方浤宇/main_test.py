"""
CEC 2008 基准测试主程序 / CEC 2008 Benchmark Main Program.

用于运行并评估 DECC-ML 算法在 CEC 2008 测试集上的性能。
Used to run and evaluate DECC-ML algorithm performance on the CEC 2008 benchmark suite.
"""

import time
import logging
from typing import Dict, Any
import numpy as np

from opfunu.cec_based.cec2008 import F12008, F22008, F32008, F42008, F52008, F62008
from optimizer import DECCMLOptimizer

# ==========================================
# 配置常量 / Configuration Constants
# ==========================================
DIMENSION = 1000
RUNS_PER_FUNCTION = 5
MAX_EVALUATIONS = 5000000
LOG_INTERVAL = 500000

TARGET_FUNCTIONS = [F12008, F22008, F32008, F42008, F52008, F62008]

BIAS_DICT: Dict[str, float] = {
    "F12008": -450.0, "F22008": -450.0, "F32008": 390.0,
    "F42008": -330.0, "F52008": -180.0, "F62008": -140.0
}

# 初始化日志系统 / Initialize Logging System
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 保持纯净输出以便于查看表格 / Keep pure output for clear tables
)
logger = logging.getLogger(__name__)


def run_benchmarks() -> None:
    """
    执行完整的基准测试流程 / Execute the complete benchmark process.
    """
    logger.info("=" * 105)
    logger.info("CEC 2008 LSGO 基准测试自动评测脚本已启动 / Auto-evaluation Script Started")
    logger.info(f"独立运行 / Independent Runs: {RUNS_PER_FUNCTION} | 评估次数 / Max FEs: {MAX_EVALUATIONS}")
    logger.info("=" * 105)

    total_start_time = time.time()
    final_academic_report: Dict[str, Dict[str, float]] = {}

    for func_class in TARGET_FUNCTIONS:
        current_func = func_class(ndim=DIMENSION)
        func_name = current_func.name
        bias = BIAS_DICT.get(func_class.__name__, 0.0)

        logger.info(f"\n开始评测函数 / Starting evaluation for: {func_name} (Bias: {bias})")
        error_list = []

        for run in range(1, RUNS_PER_FUNCTION + 1):
            run_start = time.time()
            logger.info(f"  -> 运行 / Run {run}/{RUNS_PER_FUNCTION} 初始化 / Initializing...")

            optimizer = DECCMLOptimizer(
                objective=current_func.evaluate,
                dimension=DIMENSION,
                lower_bound=current_func.lb,
                upper_bound=current_func.ub,
                max_evaluations=MAX_EVALUATIONS
            )

            best_fitness, _ = optimizer.solve(log_interval=LOG_INTERVAL)

            # 计算绝对误差 / Calculate absolute error
            run_error = float(abs(best_fitness - bias))
            error_list.append(run_error)
            
            elapsed_time = time.time() - run_start
            logger.info(f"   完成运行 / Run completed {run}/{RUNS_PER_FUNCTION} | "
                        f"耗时 / Time: {elapsed_time:.2f}s | "
                        f"最优解 / Best: {best_fitness:.6e} | "
                        f"Error: {run_error:.4e}\n")

        # 统计分析 / Statistical analysis
        mean_err = float(np.mean(error_list))
        std_err = float(np.std(error_list))
        var_err = float(np.var(error_list))
        final_academic_report[func_name] = {'Mean': mean_err, 'Std': std_err, 'Var': var_err}
        
        logger.info(f"[{func_name}] 统计 / Stats -> Mean: {mean_err:.4e} | Std: {std_err:.4e} | Var: {var_err:.4e}")
        logger.info("-" * 70)

    # 打印最终报告 / Print final report
    _print_final_report(final_academic_report, time.time() - total_start_time)


def _print_final_report(report: Dict[str, Dict[str, float]], total_time: float) -> None:
    """
    格式化输出最终实验报告 / Format and output the final experimental report.

    Args:
        report: 实验统计结果字典 / Experimental statistics dictionary
        total_time: 总耗时(秒) / Total elapsed time in seconds
    """
    logger.info("\n" + "=" * 105)
    logger.info("DECC-ML (SaNSDE) 最终实验数据报告 / Final Experimental Data Report")
    logger.info(f"总耗时 / Total Time: {total_time/3600:.2f} h | 评估次数 / FEs: {MAX_EVALUATIONS} | 运行 / Runs: {RUNS_PER_FUNCTION}")
    logger.info("=" * 105)

    header = f"{'Function':<40} | {'Mean Error':<15} | {'Std Dev':<15} | {'Variance':<15}"
    logger.info(header)
    logger.info("-" * 105)

    for name, stats in report.items():
        row = f"{name:<40} | {stats['Mean']:<15.4e} | {stats['Std']:<15.4e} | {stats['Var']:<15.4e}"
        logger.info(row)

    logger.info("=" * 105)
    logger.info("实验全部结束 / Experiment completely finished.")


if __name__ == "__main__":
    run_benchmarks()