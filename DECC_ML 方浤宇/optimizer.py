"""
DECC-ML 优化器模块 / DECC-ML Optimizer Module.

提供基于自适应差分进化的大规模全局优化求解器。
Provides a large-scale global optimization solver based on self-adaptive differential evolution.
"""

import logging
from typing import Callable, Tuple, List, Optional
import numpy as np
import numpy.typing as npt

# 配置日志 / Configure logging
logger = logging.getLogger(__name__)


class DECCMLOptimizer:
    """
    DECC-ML (SaNSDE) 优化器类 / DECC-ML (SaNSDE) Optimizer Class.

    用于在大规模连续空间中寻找复杂目标函数的最优解。
    Used for finding optima of complex objective functions in large-scale continuous space.

    Attributes:
        objective (Callable): 目标函数 / Objective function
        dimension (int): 决策变量维度 / Decision variable dimension
        max_evaluations (int): 最大适应度评估次数 / Maximum fitness evaluations
        population_size (int): 种群大小 / Population size
    """

    # 常量配置 / Constant Configuration
    DEFAULT_POPULATION_SIZE = 50
    DEFAULT_S_POOL = (5, 10, 25, 50, 100)
    DEFAULT_LEARNING_PERIOD = 50

    def __init__(
        self,
        objective: Callable[[npt.NDArray[np.float64]], float],
        dimension: int,
        lower_bound: npt.NDArray[np.float64],
        upper_bound: npt.NDArray[np.float64],
        max_evaluations: int
    ) -> None:
        """
        初始化 DECC-ML 优化器 / Initialize DECC-ML Optimizer.

        Args:
            objective: 目标函数 / Objective function to minimize
            dimension: 问题维度 / Problem dimension
            lower_bound: 搜索空间下界 / Lower bounds of search space
            upper_bound: 搜索空间上界 / Upper bounds of search space
            max_evaluations: 最大评估次数 / Maximum number of evaluations

        Raises:
            ValueError: 当维度或评估次数无效时 / When dimension or evaluations are invalid
        """
        if dimension < 1:
            raise ValueError(f"维度必须 >= 1 / Dimension must be >= 1, got {dimension}")
        if max_evaluations <= 0:
            raise ValueError(f"评估次数必须 > 0 / Evaluations must be > 0, got {max_evaluations}")

        self.objective = objective
        self.dimension = dimension
        self.lower_bound = np.asarray(lower_bound, dtype=np.float64)
        self.upper_bound = np.asarray(upper_bound, dtype=np.float64)
        self.max_evaluations = max_evaluations

        self.population_size = self.DEFAULT_POPULATION_SIZE
        self.s_pool = self.DEFAULT_S_POOL
        self.learning_period = self.DEFAULT_LEARNING_PERIOD

        self._rng = np.random.default_rng()
        self._current_evaluations = 0

        # 初始化 SaNSDE 状态参数 / Initialize SaNSDE state parameters
        self._reset_sansde_parameters()

    def _reset_sansde_parameters(self) -> None:
        """重置内部自适应参数与统计信息 / Reset internal adaptive parameters and statistics."""
        self._p_strategy = 0.5
        self._mu_cr = 0.5
        self._mu_f = 0.5
        self._lp_counter = 0
        self._success_cr: List[float] = []
        self._success_f: List[float] = []
        self._ns1 = self._nf1 = self._ns2 = self._nf2 = 0

    def _mutate_and_crossover(
        self,
        sub_pop: npt.NDArray[np.float64],
        local_best: npt.NDArray[np.float64],
        sub_indices: npt.NDArray[np.int64],
        current_s: int
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """
        执行变异与交叉操作 / Execute mutation and crossover operations.

        Args:
            sub_pop: 当前子种群 / Current sub-population
            local_best: 局部最优解 / Local best solution
            sub_indices: 子代维度索引 / Sub-dimension indices
            current_s: 当前分组大小 / Current group size

        Returns:
            包含试验种群、CR记录、F记录、策略记录的元组 / Tuple of trial pop, CR records, F records, strategy records
        """
        trials_sub = np.zeros_like(sub_pop)
        cr_records = np.zeros(self.population_size)
        f_records = np.zeros(self.population_size)
        strat_records = np.zeros(self.population_size, dtype=int)

        lb_sub = self.lower_bound[sub_indices]
        ub_sub = self.upper_bound[sub_indices]

        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            r1, r2, r3 = self._rng.choice(idxs, 3, replace=False)

            # 生成自适应参数 / Generate adaptive parameters
            cr_i = np.clip(self._rng.normal(self._mu_cr, 0.1), 0.0, 1.0)
            f_i = self._rng.normal(self._mu_f, 0.3)
            while f_i <= 0:
                f_i = self._rng.normal(self._mu_f, 0.3)
            f_i = min(f_i, 1.0)

            cr_records[i] = cr_i
            f_records[i] = f_i

            # 变异策略选择 / Mutation strategy selection
            if self._rng.random() < self._p_strategy:
                v = sub_pop[r1] + f_i * (sub_pop[r2] - sub_pop[r3])
                strat_records[i] = 1
            else:
                v = sub_pop[i] + f_i * (local_best - sub_pop[i]) + f_i * (sub_pop[r1] - sub_pop[r2])
                strat_records[i] = 2

            v = np.clip(v, lb_sub, ub_sub)
            cross_points = self._rng.random(current_s) < cr_i
            trials_sub[i] = np.where(cross_points, v, sub_pop[i])

        return trials_sub, cr_records, f_records, strat_records

    def _evaluate_coevolution(
        self,
        best_context: npt.NDArray[np.float64],
        sub_indices: npt.NDArray[np.int64],
        sub_pop: npt.NDArray[np.float64],
        trials_sub: npt.NDArray[np.float64]
    ) -> Tuple[Optional[npt.NDArray[np.bool_]], Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
        """
        执行协作共进化评估 / Execute cooperative co-evolution evaluation.

        Args:
            best_context: 全局最优上下文 / Global best context
            sub_indices: 子代维度索引 / Sub-dimension indices
            sub_pop: 目标子种群 / Target sub-population
            trials_sub: 试验子种群 / Trial sub-population

        Returns:
            若超过评估次数返回 None。否则返回更好索引掩码、试验适应度、目标适应度 /
            Returns None if evaluations exceeded. Else returns better indices mask, trial fitness, target fitness
        """
        context_pop = np.tile(best_context, (self.population_size, 1))

        # 评估目标解 / Evaluate target solutions
        target_context = context_pop.copy()
        target_context[:, sub_indices] = sub_pop
        target_fitness = np.array([self.objective(ind) for ind in target_context])
        self._current_evaluations += self.population_size

        if self._current_evaluations >= self.max_evaluations:
            return None, None, None

        # 评估试验解 / Evaluate trial solutions
        trial_context = context_pop.copy()
        trial_context[:, sub_indices] = trials_sub
        trial_fitness = np.array([self.objective(ind) for ind in trial_context])
        self._current_evaluations += self.population_size

        better_idxs = trial_fitness < target_fitness
        return better_idxs, trial_fitness, target_fitness

    def _update_adaptation_stats(
        self,
        better_idxs: npt.NDArray[np.bool_],
        cr_records: npt.NDArray[np.float64],
        f_records: npt.NDArray[np.float64],
        strat_records: npt.NDArray[np.int64]
    ) -> None:
        """
        更新 SaNSDE 自适应统计信息 / Update SaNSDE adaptive statistics.

        Args:
            better_idxs: 适应度改进掩码 / Fitness improvement mask
            cr_records: CR 记录数组 / CR records array
            f_records: F 记录数组 / F records array
            strat_records: 策略记录数组 / Strategy records array
        """
        for i in range(self.population_size):
            if better_idxs[i]:
                self._success_cr.append(cr_records[i])
                self._success_f.append(f_records[i])
                if strat_records[i] == 1:
                    self._ns1 += 1
                else:
                    self._ns2 += 1
            else:
                if strat_records[i] == 1:
                    self._nf1 += 1
                else:
                    self._nf2 += 1

        self._lp_counter += 1
        if self._lp_counter >= self.learning_period:
            denom1, denom2 = self._ns1 + self._nf1, self._ns2 + self._nf2
            if denom1 > 0 and denom2 > 0:
                sr1, sr2 = self._ns1 / denom1, self._ns2 / denom2
                if (sr1 + sr2) > 0:
                    self._p_strategy = sr1 / (sr1 + sr2)

            if len(self._success_cr) > 0:
                self._mu_cr = float(np.mean(self._success_cr))
            if len(self._success_f) > 0:
                self._mu_f = float(np.mean(self._success_f))

            self._success_cr.clear()
            self._success_f.clear()
            self._ns1 = self._nf1 = self._ns2 = self._nf2 = 0
            self._lp_counter = 0

    def solve(self, log_interval: int = 500000) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        执行全局优化求解过程 / Execute global optimization solving process.

        Args:
            log_interval: 日志打印的评估间隔 / Evaluation interval for logging

        Returns:
            Tuple[float, ndarray]: (最优适应度, 最优解向量) / (Best fitness, Best solution vector)
        """
        self._reset_sansde_parameters()
        self._current_evaluations = 0

        # 种群初始化 / Initialize population
        pop = self._rng.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        fitness = np.array([self.objective(ind) for ind in pop])
        self._current_evaluations += self.population_size

        best_idx = int(np.argmin(fitness))
        best_context = pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        old_best_fitness = float('inf')
        current_s = self._rng.choice(self.s_pool)
        last_print_evals = 0

        logger.debug("开始 DECC-ML 优化迭代 / Starting DECC-ML optimization iteration")

        while self._current_evaluations < self.max_evaluations:
            # 分组大小自适应机制 / Adaptive group size mechanism
            if best_fitness >= old_best_fitness:
                current_s = self._rng.choice(self.s_pool)
            old_best_fitness = best_fitness

            shuffled_indices = self._rng.permutation(self.dimension)
            groups = [shuffled_indices[i:i+current_s] for i in range(0, self.dimension, current_s)]

            for sub_indices in groups:
                if self._current_evaluations >= self.max_evaluations:
                    break

                sub_pop = pop[:, sub_indices]
                local_best = best_context[sub_indices]

                # 1. 变异与交叉 / Mutation and Crossover
                trials_sub, cr_recs, f_recs, strat_recs = self._mutate_and_crossover(
                    sub_pop, local_best, sub_indices, current_s
                )

                # 2. 协作评估 / Cooperative Evaluation
                better_idxs, trial_fitness, target_fitness = self._evaluate_coevolution(
                    best_context, sub_indices, sub_pop, trials_sub
                )
                if better_idxs is None:
                    break

                # 3. 更新统计参数 / Update Statistics
                self._update_adaptation_stats(better_idxs, cr_recs, f_recs, strat_recs)

                # 4. 种群与全局最优更新 / Population and Global Best Update
                temp_pop = pop.copy()
                temp_pop[:, sub_indices] = trials_sub
                pop = np.where(better_idxs[:, None], temp_pop, pop)

                all_fitness_in_round = np.where(better_idxs, trial_fitness, target_fitness)
                current_min_idx = int(np.argmin(all_fitness_in_round))

                if all_fitness_in_round[current_min_idx] < best_fitness:
                    best_fitness = float(all_fitness_in_round[current_min_idx])
                    best_context = pop[current_min_idx].copy()

            # 进度监控与日志 / Progress monitoring and logging
            if self._current_evaluations - last_print_evals >= log_interval:
                logger.info(f"[进度/Progress] FEs: {self._current_evaluations:7d} / {self.max_evaluations} "
                            f"| 全局最优解/Global Best: {best_fitness:.6e}")
                last_print_evals = self._current_evaluations

        return best_fitness, best_context