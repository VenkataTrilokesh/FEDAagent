from __future__ import annotations

from math import ceil
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12


class Visualizer:
    """Reusable plotting utilities for generated notebooks."""

    @staticmethod
    def plot_missing_values(df: pd.DataFrame, top_n: int = 20) -> None:
        missing = df.isna().mean().sort_values(ascending=False).head(top_n)
        missing = missing[missing > 0]
        if missing.empty:
            print("No missing values to visualize.")
            return
        ax = missing.mul(100).sort_values().plot(kind="barh", color="#5B8FF9")
        ax.set_title("Missing Value Percentage by Column")
        ax.set_xlabel("Missing %")
        ax.set_ylabel("Features")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: Iterable[str], max_cols: int = 6) -> None:
        cols = list(numeric_cols)[:max_cols]
        if not cols:
            print("No numeric columns available.")
            return
        rows = ceil(len(cols) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2E8B57")
            ax.set_title(f"Distribution of {col}")
        for ax in axes[len(cols):]:
            ax.remove()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_boxplots(df: pd.DataFrame, numeric_cols: Iterable[str], max_cols: int = 6) -> None:
        cols = list(numeric_cols)[:max_cols]
        if not cols:
            print("No numeric columns available.")
            return
        rows = ceil(len(cols) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, cols):
            sns.boxplot(x=df[col], ax=ax, color="#FFB347")
            ax.set_title(f"Boxplot of {col}")
        for ax in axes[len(cols):]:
            ax.remove()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_categorical_counts(df: pd.DataFrame, categorical_cols: Iterable[str], max_cols: int = 6) -> None:
        cols = list(categorical_cols)[:max_cols]
        if not cols:
            print("No categorical columns available.")
            return
        rows = ceil(len(cols) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, cols):
            top_values = df[col].astype(str).value_counts().head(10)
            sns.barplot(x=top_values.values, y=top_values.index, ax=ax, palette="viridis")
            ax.set_title(f"Top Categories in {col}")
            ax.set_xlabel("Count")
            ax.set_ylabel(col)
        for ax in axes[len(cols):]:
            ax.remove()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame, max_features: int = 15) -> None:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            print("Not enough numeric features for a correlation heatmap.")
            return
        selected = numeric_df.var().sort_values(ascending=False).head(max_features).index
        corr = numeric_df[selected].corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pairplot(df: pd.DataFrame, numeric_cols: Iterable[str], target_col: Optional[str] = None, max_features: int = 5, sample_size: int = 400) -> None:
        cols = list(numeric_cols)[:max_features]
        if len(cols) < 2:
            print("Not enough numeric columns for pairplot.")
            return
        plot_df = df[cols + ([target_col] if target_col and target_col in df.columns else [])].dropna()
        if len(plot_df) > sample_size:
            plot_df = plot_df.sample(sample_size, random_state=42)
        sns.pairplot(plot_df, hue=target_col if target_col and target_col in plot_df.columns else None, corner=True)
        plt.show()

    @staticmethod
    def plot_target_relationships(df: pd.DataFrame, target_col: Optional[str], max_numeric: int = 4, max_cat: int = 4) -> None:
        if not target_col or target_col not in df.columns:
            print("No target column available for target-wise plots.")
            return

        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target_col][:max_numeric]
        categorical_cols = [c for c in df.columns if c not in numeric_cols + [target_col] and not pd.api.types.is_datetime64_any_dtype(df[c])][:max_cat]

        if pd.api.types.is_numeric_dtype(df[target_col]):
            for col in numeric_cols:
                plt.figure(figsize=(8, 5))
                sns.scatterplot(data=df, x=col, y=target_col, alpha=0.7)
                plt.title(f"{col} vs {target_col}")
                plt.tight_layout()
                plt.show()
        else:
            for col in numeric_cols:
                plt.figure(figsize=(8, 5))
                sns.boxplot(data=df, x=target_col, y=col)
                plt.title(f"{col} by {target_col}")
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.show()
            for col in categorical_cols:
                plt.figure(figsize=(10, 5))
                tmp = df[[col, target_col]].astype(str)
                order = tmp[col].value_counts().head(8).index
                sns.countplot(data=tmp[tmp[col].isin(order)], x=col, hue=target_col)
                plt.title(f"{col} vs {target_col}")
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.show()

    @staticmethod
    def plot_interactive_summary(df: pd.DataFrame, target_col: Optional[str] = None):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            print("No numeric columns available for interactive visualization.")
            return None
        x_col = numeric_cols[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        color = target_col if target_col in df.columns else None
        fig = px.scatter(df, x=x_col, y=y_col, color=color, title="Interactive Numeric Overview", opacity=0.7)
        fig.show()
        return fig
