import pandas as pd               
import numpy as np                
import matplotlib.pyplot as plt   
import seaborn as sns              
from scipy.stats import spearmanr 
from IPython.display import display


# ============================================================
# Statistical Utilities
# ============================================================

# P-Value Formatting
def format_pvalue(p):
    """
    Format p-values for statistical output.

    Values below 0.001 are reported as '<0.001';
    all others are displayed to three decimal places.
    """
    return '<0.001' if p < 0.001 else f'{p:.3f}'


# Gini Coefficient
def gini(array):
    """
    Calculate the Gini coefficient for a one-dimensional numeric array.

    A value of 0 represents perfect equality, while values approaching 1
    indicate increasing concentration or inequality.

    Parameters
    ----------
    array : array-like
        Non-negative numeric values.

    Returns
    -------
    float
        Gini coefficient.
    """

    sorted_array = np.sort(np.asarray(array))
    n = len(sorted_array)

    return (
        2 * np.sum(np.arange(1, n + 1) * sorted_array)
        / (n * sorted_array.sum())
    ) - (n + 1) / n


# Spearman Correlation Summary
def corr_summary(activity_counts, stats_df):
    """
    Calculate Spearman correlations between transaction activity
    and area-level property size metrics.

    Parameters
    ----------
    activity_counts : pandas.Series
        Transaction counts used as the activity measure.

    stats_df : pandas.DataFrame
        Area-level statistics containing:
        - median
        - IQR
        - relative_IQR

    Returns
    -------
    dict
        Spearman correlation coefficients and p-values
        for each size metric.
    """
    metrics = {
        'median': 'Median Size',
        'IQR': 'Absolute IQR',
        'relative_IQR': 'Relative IQR'
    }

    results = {}

    for column, label in metrics.items():
        rho, p_value = spearmanr(
            activity_counts,
            stats_df[column]
        )

        results[f'{label} rho'] = rho
        results[f'{label} p-value'] = p_value

    return results


# ============================================================
# Feature Transformation
# ============================================================

# Bedroom Category Simplification
def simplify_rooms(room):
    """
    Consolidate bedroom-count categories into a simplified room grouping.

    Studio, 1 B/R, 2 B/R, and 3 B/R categories are retained unchanged.
    Properties with four or more bedrooms are grouped into a single
    '4+ B/R' category.

    Parameters
    ----------
    room : str
        Original room-category label.

    Returns
    -------
    str
        Simplified room-category label.
    """
    if pd.isna(room):
        return room
        
    if room == 'Studio':
        return room

    if room.endswith(' B/R'):
        bedroom_count = int(room.split()[0])

        if bedroom_count >= 4:
            return '4+ B/R'

    return room


# ============================================================
# Area-Level Size Statistics
# ============================================================


def create_type_size_stats(df_type, area_size_stats):
    """
    Create area-level property size statistics for a single property type.

    Calculates:
    - type-specific transaction count
    - median and mean property size
    - first and third quartiles
    - absolute IQR
    - relative IQR

    Total area transaction counts are also merged in to allow comparison
    against both type-specific and overall market activity.

    Parameters
    ----------
    df_type : pandas.DataFrame
        Transactions belonging to one property type.

    area_size_stats : pandas.DataFrame
        Area-level statistics containing total transaction counts.

    Returns
    -------
    pandas.DataFrame
        Area-level size statistics for the selected property type.
    """
    stats = (
        df_type.groupby('area')['size']
        .agg(
            transaction_count='count',
            median='median',
            mean='mean',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )

    stats['IQR'] = (
        stats['q3'] - stats['q1']
    )

    stats['relative_IQR'] = (
        stats['IQR'] / stats['median']
    )

    stats = stats.merge(
        area_size_stats[['area', 'transaction_count']],
        on='area',
        how='left',
        suffixes=('', '_global')
    )

    stats = (
        stats
        .sort_values('transaction_count', ascending=False)
        .reset_index(drop=True)
    )

    return stats


# ============================================================
# Visualization
# ============================================================

# Grouped Share Analysis
def plot_grouped_share(
    df, 
    category_col, 
    category_order, 
    area_summary, 
    title, 
    ylabel="Average % per Category",
    palette="crest",
    show_correlation=False
):
    """
    Plots grouped bar chart showing average category composition (e.g., property type or room type) 
    across transaction-activity tertiles, returns the summary DataFrame and optionally displays Spearman correlations
    between each category's share and transaction activity.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'area' and the categorical column to analyze.
    category_col : str
        Categorical feature to analyse (e.g. 'property_type', 'rooms_simplified').
    category_order : list
        Ordered list of category labels to plot.
    area_summary : pd.DataFrame
        Must contain ['area', 'transaction_count'] for grouping activity levels.
    title : str
        Plot title.
    ylabel : str, optional
        Label for y-axis (default: 'Average % per Category').
    palette : str or list, optional
        Seaborn-compatible color palette.
    show_correlation : bool, optional
        If True, prints Spearman correlation between each category's share and transaction activity.
    
    Returns
    -------
    area_share : pd.DataFrame
        Area-level DataFrame containing category shares and transaction activity.
    summary : pandas.DataFrame
        Mean category shares within each transaction-activity tertile.
    """

    # Compute category share per area
    mix = df.groupby(['area', category_col]).size().reset_index(name='count')
    total_per_area = mix.groupby('area')['count'].sum().reset_index(name='total')
    mix = mix.merge(total_per_area, on='area')
    mix['pct'] = (mix['count'] / mix['total']) * 100

    # Pivot wider so each category becomes a column
    area_share = mix.pivot(index='area', columns=category_col, values='pct').fillna(0)

    # Ensure all expected categories are present
    for col in category_order:
        if col not in area_share.columns:
            area_share[col] = 0

    # Keep category columns in the requested order
    area_share = area_share[category_order]


    # Merge area transaction counts
    area_share = area_share.merge(area_summary[['area', 'transaction_count']], on='area')

    # Divide areas into three transaction-activity tertiles
    area_share['activity_group'] = pd.qcut(
        area_share['transaction_count'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    # Compute mean % per activity level
    summary = (
        area_share.groupby('activity_group', observed=False)[category_order]
        .mean()
        .reset_index()
    )

    # Plot grouped bar chart
    x = np.arange(len(summary))
    bar_width = 0.15

    num_cats = len(category_order)
    offset_center = (num_cats - 1) / 2

    # Determine colors inside function
    if isinstance(palette, (list, tuple)):
        colors = palette
    else:
        colors = sns.color_palette(palette, n_colors=num_cats)

    plt.figure(figsize=(11,6))


    for i, cat in enumerate(category_order):
        plt.bar(
                x + (i - offset_center) * bar_width,
                summary[cat],
                width=bar_width,
                label=cat,
                color=colors[i])

    plt.xticks(x,summary['activity_group'])
    plt.title(title)
    plt.xlabel('Activity Level (Tertiles)')
    plt.ylabel(ylabel)
    plt.legend(
        title=category_col.replace("_", " ").title(),
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.grid(axis='y', alpha=0.3)

    # Add percentage labels above bars
    for idx, row in summary.iterrows():
        for i, cat in enumerate(category_order):
            x_pos = idx + (i - offset_center) * bar_width
            plt.text(
                x_pos,
                row[cat] + 0.5,
                f"{row[cat]:.1f}%",
                ha='center',
                va='bottom',
                fontsize=8
            )
            
    plt.tight_layout()
    plt.show()

    # Optional: compute and display Spearman correlations
    if show_correlation:
        correlation_results = []
        for cat in category_order:
            rho, p = spearmanr(area_share[cat], area_share['transaction_count'])
            correlation_results.append({'Category Type': cat, 'Spearman ρ': rho, 'p-value': p})

        correlation_df = pd.DataFrame(correlation_results)
        print(f"Spearman correlation between {category_col} share and transaction activity:")
        display(correlation_df.style.format({'Spearman ρ': '{:.3f}', 'p-value': format_pvalue}))

    return area_share, summary
