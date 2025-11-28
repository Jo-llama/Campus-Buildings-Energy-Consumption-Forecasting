import shap
import numpy as np
import matplotlib.pyplot as plt


def shap_summary_global(model, X, max_display=30):
    """
    Globalne znaczenie cech dla modelu drzewiastego (np. XGBoost)
    przy użyciu SHAP TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)  # TreeExplainer jest zalecany dla XGBoost[web:104][web:113]
    shap_values = explainer(X)

    # beeswarm – rozkład wartości SHAP dla wszystkich obserwacji[web:103][web:106]
    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=X.columns,
        max_display=max_display
    )


def shap_summary_bar(model, X, max_display=30):
    """
    Globalne znaczenie cech – barplot z średnich |SHAP|[web:108][web:117].
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=X.columns,
        max_display=max_display,
        plot_type="bar"
    )


def shap_waterfall_single(model, X, index=0, max_display=15):
    """
    Lokalna interpretacja dla jednej obserwacji (waterfall plot)[web:112][web:113].
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    shap.plots.waterfall(
        shap_values[index],
        max_display=max_display
    )


def model_comparison_before_after(benchmark_results, optimized_results, title="Benchmark vs Optimized Models"):
    """
    Compare benchmark models (04) with optimized models (05).
    Shows improvement from feature selection + hyperparameter tuning.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create dataframes
    df_bench = pd.DataFrame(benchmark_results).T  # Lasso, Ridge, GB, RF
    df_optim = pd.DataFrame(optimized_results).T  # Lasso (tuned), XGBoost

    # Combine
    df_all = pd.concat([df_bench, df_optim], axis=0)
    df_all['Stage'] = ['Benchmark'] * len(df_bench) + ['Optimized'] * len(df_optim)

    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. RMSE Comparison
    sns.barplot(data=df_all, x=df_all.index, y='RMSE', hue='Stage', ax=axes[0], palette=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE (kWh)', fontsize=11)
    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # 2. R² Comparison
    sns.barplot(data=df_all, x=df_all.index, y='R2', hue='Stage', ax=axes[1], palette=['#FF6B6B', '#4ECDC4'])
    axes[1].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R² Score', fontsize=11)
    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    # 3. Improvement %
    best_bench_rmse = df_bench['RMSE'].min()
    best_optim_rmse = df_optim['RMSE'].min()
    improvement_rmse = ((best_bench_rmse - best_optim_rmse) / best_bench_rmse) * 100

    best_bench_r2 = df_bench['R2'].max()
    best_optim_r2 = df_optim['R2'].max()
    improvement_r2 = ((best_optim_r2 - best_bench_r2) / best_bench_r2) * 100

    improvements = [improvement_rmse, improvement_r2]
    metrics = ['RMSE\nImprovement\n(lower better)', 'R² Improvement\n(higher better)']
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]

    axes[2].bar(metrics, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Improvement (%)', fontsize=11)
    axes[2].set_title('Improvement: Benchmark → Optimized', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[2].grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, (met, imp) in enumerate(zip(metrics, improvements)):
        axes[2].text(i, imp + (1 if imp > 0 else -2), f'{imp:.1f}%',
                     ha='center', fontsize=11, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Print detailed comparison
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON: BENCHMARK vs OPTIMIZED")
    print("=" * 70)
    print("\n🔵 BENCHMARK MODELS (04_modeling.ipynb):")
    print(f"   All features ({48} features)")
    print(df_bench.to_string())
    print(f"\n   ✓ Best: {df_bench['RMSE'].idxmin()} (RMSE: {df_bench['RMSE'].min():.4f})")

    print("\n" + "-" * 70)
    print("\n🟢 OPTIMIZED MODELS (05-optymalizacja.ipynb):")
    print(f"   Feature-selected + Optuna tuned")
    print(df_optim.to_string())
    print(f"\n   ✓ Best: {df_optim['RMSE'].idxmin()} (RMSE: {df_optim['RMSE'].min():.4f})")

    print("\n" + "-" * 70)
    print("\n📈 IMPROVEMENTS:")
    print(f"   • RMSE: {best_bench_rmse:.4f} → {best_optim_rmse:.4f} ({improvement_rmse:+.1f}%)")
    print(f"   • R²:   {best_bench_r2:.4f} → {best_optim_r2:.4f} ({improvement_r2:+.1f}%)")
    print(
        f"\n   • Features: {48} → {len(selected_features)} (reduced by {100 * (1 - len(selected_features) / 48):.1f}%)")
    print("\n" + "=" * 70 + "\n")

    return df_all


def lasso_comparison_benchmark_vs_optuna(
        lasso_benchmark,
        lasso_optuna,
        X_test_benchmark,
        X_test_optuna,
        benchmark_metrics,
        optuna_metrics,
        title="Lasso: Benchmark vs Optuna (Optimized)"
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # 1. METRYKI
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_names = ['RMSE', 'MAE', 'R²']
    benchmark_vals = [benchmark_metrics['RMSE'], benchmark_metrics['MAE'], benchmark_metrics['R2']]
    optuna_vals = [optuna_metrics['RMSE'], optuna_metrics['MAE'], optuna_metrics['R2']]
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, benchmark_vals, width, label='Benchmark (All features)', color='#FF6B6B', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width / 2, optuna_vals, width, label='Optuna (Feature-selected)', color='#4ECDC4', alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. IMPROVEMENT %
    ax2 = fig.add_subplot(gs[0, 1])
    rmse_improvement = ((benchmark_metrics['RMSE'] - optuna_metrics['RMSE']) / benchmark_metrics['RMSE']) * 100
    mae_improvement = ((benchmark_metrics['MAE'] - optuna_metrics['MAE']) / benchmark_metrics['MAE']) * 100
    r2_improvement = ((optuna_metrics['R2'] - benchmark_metrics['R2']) / benchmark_metrics['R2']) * 100
    improvements = [rmse_improvement, mae_improvement, r2_improvement]
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    bars = ax2.barh(metrics_names, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Optimization Improvement', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax2.text(imp + (0.5 if imp > 0 else -0.5), i, f'{imp:+.1f}%', va='center', ha='left' if imp > 0 else 'right',
                 fontsize=10, fontweight='bold')

    # 3. Coefficients benchmark
    ax3 = fig.add_subplot(gs[1, 0])
    coef_bench = pd.Series(lasso_benchmark.coef_, index=X_test_benchmark.columns)
    coef_bench_sorted = coef_bench.abs().nlargest(15)
    coef_bench_sorted_signed = coef_bench.loc[coef_bench_sorted.index]
    colors_bench = ['green' if x > 0 else 'red' for x in coef_bench_sorted_signed.values]
    coef_bench_sorted_signed.plot(kind='barh', ax=ax3, color=colors_bench, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax3.set_title(f'Top 15 Features - Benchmark Lasso\n({len(X_test_benchmark.columns)} features)', fontsize=11,
                  fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # 4. Optimized coefficients
    ax4 = fig.add_subplot(gs[1, 1])
    coef_optuna = pd.Series(lasso_optuna.coef_, index=X_test_optuna.columns)
    coef_optuna_sorted = coef_optuna.abs().nlargest(15)
    coef_optuna_sorted_signed = coef_optuna.loc[coef_optuna_sorted.index]
    colors_optuna = ['green' if x > 0 else 'red' for x in coef_optuna_sorted_signed.values]
    coef_optuna_sorted_signed.plot(kind='barh', ax=ax4, color=colors_optuna, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
    ax4.set_title(f'Top 15 Features - Optuna Lasso\n({len(X_test_optuna.columns)} features)', fontsize=11,
                  fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # 5. Feature count
    ax5 = fig.add_subplot(gs[2, 0])
    non_zero_bench = (lasso_benchmark.coef_ != 0).sum()
    non_zero_optuna = (lasso_optuna.coef_ != 0).sum()
    total_bench = len(X_test_benchmark.columns)
    total_optuna = len(X_test_optuna.columns)
    df_features = pd.DataFrame({
        'Total Features': [total_bench, total_optuna],
        'Non-Zero Coef': [non_zero_bench, non_zero_optuna]
    }, index=['Benchmark', 'Optuna'])
    df_features.plot(kind='bar', ax=ax5, color=['#95E1D3', '#F38181'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('Feature Count Comparison', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
    ax5.legend(['Features'], loc='upper right', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)
    for container in ax5.containers:
        ax5.bar_label(container, fontsize=10, fontweight='bold')

    # 6. Summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    summary_text = f"""
📊 SUMMARY: LASSO BENCHMARK vs OPTUNA

🔵 BENCHMARK LASSO:
• Features: {total_bench} (all)
• Non-zero: {non_zero_bench}
• RMSE: {benchmark_metrics['RMSE']:.4f}
• MAE: {benchmark_metrics['MAE']:.4f}
• R²: {benchmark_metrics['R2']:.4f}

🟢 OPTUNA LASSO:
• Features: {total_optuna}
• Non-zero: {non_zero_optuna}
• RMSE: {optuna_metrics['RMSE']:.4f}
• MAE: {optuna_metrics['MAE']:.4f}
• R²: {optuna_metrics['R2']:.4f}

📈 IMPROVEMENTS:
• RMSE: {rmse_improvement:+.2f}%
• MAE: {mae_improvement:+.2f}%
• R²: {r2_improvement:+.2f}%
• Feature reduction: {100 * (1 - total_optuna / total_bench):.1f}%
    """
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10, verticalalignment='top')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.show()
