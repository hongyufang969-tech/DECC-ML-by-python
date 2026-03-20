import numpy as np
import time
from opfunu.cec_based.cec2008 import F12008, F22008, F32008, F42008, F52008, F62008

# ==========================================
# 1. 实验参数配置
# ==========================================
D = 1000                
RUNS = 5                
MAX_FEs = 5000000       

# 测试函数集
target_funcs = [F12008, F22008, F32008, F42008, F52008, F62008]

# 官方 Bias 设定
BIAS_DICT = {
    "F12008": -450.0, "F22008": -450.0, "F32008": 390.0,
    "F42008": -330.0, "F52008": -180.0, "F62008": -140.0
}

final_academic_report = {}

print("======================================================================")
print("CEC 2008 LSGO 基准测试自动评测脚本已启动")
print(f"评测函数: F1 - F6 | 独立运行: {RUNS} 次/函数 | FEs: {MAX_FEs}")
print("======================================================================")

total_start_time = time.time()

# ==========================================
# 2. 遍历测试函数
# ==========================================
for func_class in target_funcs:
    current_func = func_class(ndim=D)
    func_name = current_func.name
    
    # 使用类的内置名称直接匹配 Bias 字典
    bias = BIAS_DICT.get(func_class.__name__, 0.0)
    
    print(f"\n开始评测函数: {func_name} (Bias: {bias})")
    error_list = []
    
    # ==========================================
    # 3. 独立运行评测
    # ==========================================
    for run in range(1, RUNS + 1):
        run_start = time.time()
        print(f"  -> 运行 {run}/{RUNS} 初始化...")
        
        # 重置 SaNSDE 参数与记忆库
        NP = 50
        
        S_POOL = [5, 10, 25, 50, 100]
        
        p_strategy = 0.5
        mu_cr = 0.5
        mu_f = 0.5
        LP = 50
        lp_counter = 0
        success_cr, success_f = [], []
        ns1, nf1, ns2, nf2 = 0, 0, 0, 0
        last_print_FEs = 0  

        def fitness_func(population):
            return np.array([current_func.evaluate(ind) for ind in population])

        # 种群初始化
        pop = np.random.uniform(current_func.lb, current_func.ub, (NP, D))
        fitness = fitness_func(pop)
        current_FEs = NP  
        best_idx = np.argmin(fitness)
        best_context = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        # 在进入 while 循环前，初始化抽取分组大小 s
        s = np.random.choice(S_POOL) 

        # --- DECC-ML 优化主循环 ---
        while current_FEs < MAX_FEs:
            # 记录当前周期开始前的最优适应度
            old_best_fitness = best_fitness 
            
            shuffled_indices = np.random.permutation(D)
            # 依据当前的 s 对全维度进行随机分组
            groups = [shuffled_indices[i:i+s] for i in range(0, D, s)]
            
            for sub_indices in groups:
                if current_FEs >= MAX_FEs: break
                sub_pop = pop[:, sub_indices]
                trials_sub = np.zeros_like(sub_pop)
                local_best = best_context[sub_indices]
                
                F_records, CR_records, strat_records = np.zeros(NP), np.zeros(NP), np.zeros(NP)
                
                # 变异与交叉操作
                for i in range(NP):
                    idxs = [idx for idx in range(NP) if idx != i]
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    
                    CR_i = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
                    F_i = np.random.normal(mu_f, 0.3)
                    while F_i <= 0: F_i = np.random.normal(mu_f, 0.3)
                    F_i = min(F_i, 1.0)
                    
                    CR_records[i], F_records[i] = CR_i, F_i
                    
                    if np.random.rand() < p_strategy:
                        v = sub_pop[r1] + F_i * (sub_pop[r2] - sub_pop[r3])
                        strat_records[i] = 1
                    else:
                        v = sub_pop[i] + F_i * (local_best - sub_pop[i]) + F_i * (sub_pop[r1] - sub_pop[r2])
                        strat_records[i] = 2
                        
                    v = np.clip(v, current_func.lb[sub_indices], current_func.ub[sub_indices])
                    cross_points = np.random.rand(s) < CR_i
                    trials_sub[i] = np.where(cross_points, v, sub_pop[i])
                    
                context_pop = np.tile(best_context, (NP, 1))
                
                # 评估目标解
                target_context = context_pop.copy()
                target_context[:, sub_indices] = sub_pop
                target_fitness = fitness_func(target_context)
                current_FEs += NP
                if current_FEs >= MAX_FEs: break
                
                # 评估试验解
                trial_context = context_pop.copy()
                trial_context[:, sub_indices] = trials_sub
                trial_fitness = fitness_func(trial_context)
                current_FEs += NP
                
                better_idxs = trial_fitness < target_fitness
                
                # 更新 SaNSDE 统计信息
                for i in range(NP):
                    if better_idxs[i]:
                        success_cr.append(CR_records[i])
                        success_f.append(F_records[i])
                        if strat_records[i] == 1: ns1 += 1
                        else: ns2 += 1
                    else:
                        if strat_records[i] == 1: nf1 += 1
                        else: nf2 += 1
                        
                lp_counter += 1
                if lp_counter >= LP:
                    denom1, denom2 = ns1 + nf1, ns2 + nf2
                    if denom1 > 0 and denom2 > 0:
                        sr1, sr2 = ns1 / denom1, ns2 / denom2
                        if (sr1 + sr2) > 0: p_strategy = sr1 / (sr1 + sr2)
                    if len(success_cr) > 0: mu_cr = np.mean(success_cr)
                    if len(success_f) > 0: mu_f = np.mean(success_f)
                    success_cr, success_f = [], []
                    ns1, nf1, ns2, nf2, lp_counter = 0, 0, 0, 0, 0

                # 种群更新
                temp_pop = pop.copy()
                temp_pop[:, sub_indices] = trials_sub
                pop = np.where(better_idxs[:, None], temp_pop, pop)
                
                all_fitness_in_this_round = np.where(better_idxs, trial_fitness, target_fitness)
                current_min_idx = np.argmin(all_fitness_in_this_round)
                
                if all_fitness_in_this_round[current_min_idx] < best_fitness:
                    best_fitness = all_fitness_in_this_round[current_min_idx]
                    best_context = pop[current_min_idx].copy()

            # --- 分组大小自适应调整机制 ---
            # 若在一个完整周期的优化后，全局最优解未获提升，则重新随机分配 s
            if best_fitness >= old_best_fitness:
                s = np.random.choice(S_POOL)

            # 节点式进度监控
            if current_FEs - last_print_FEs >= 500000:
                print(f"      [进度] FEs: {current_FEs:7d} / {MAX_FEs} | 全局最优解: {best_fitness:.6e}")
                last_print_FEs = current_FEs

        # 计算绝对误差
        run_error = abs(best_fitness - bias)
        error_list.append(run_error)
        print(f"   完成运行 {run}/{RUNS} | 耗时: {time.time() - run_start:.2f}s | 全局最优解: {best_fitness:.6e} | Error: {run_error:.4e}\n")
    
    # 统计均值、标准差与方差
    mean_err = np.mean(error_list)
    std_err = np.std(error_list)
    var_err = np.var(error_list)
    final_academic_report[func_name] = {'Mean': mean_err, 'Std': std_err, 'Var': var_err}
    print(f"[{func_name}] 统计结果 -> Mean: {mean_err:.4e} | Std: {std_err:.4e} | Variance: {var_err:.4e}\n" + "-"*70)
# ==========================================
# 4. 输出最终实验报告
# ==========================================
total_cost = time.time() - total_start_time
print("\n" + "=" * 105)
print("DECC-ML (SaNSDE) 最终实验数据报告")
print(f"总耗时: {total_cost/3600:.2f} h | FEs: 5e6 | 独立运行: {RUNS}")
print("=" * 105)

print(f"{'Function':<40} | {'Mean Error':<15} | {'Std Dev':<15} | {'Variance':<15}")
print("-" * 105)
for name, stats in final_academic_report.items():
    print(f"{name:<40} | {stats['Mean']:<15.4e} | {stats['Std']:<15.4e} | {stats['Var']:<15.4e}")
print("=" * 105)
print("实验全部结束。")