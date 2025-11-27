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
