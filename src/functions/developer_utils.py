import pandas as pd # Data Manipulation
import numpy as np # Data Manipulation
import re
from rapidfuzz import process, fuzz


# ============================================================
# Text preprocessing
# ============================================================


# --- Normalize all text ---
def normalize_text(s):
    """
    Clean and standardize strings for text matching.
    - Convert to lowercase
    - Remove excessive whitespace
    - Strip leading/trailing spaces
    """
    s = str(s).lower().strip()
    s = re.sub(r'\s+', ' ', s)       # collapse multiple spaces
    return s


# ============================================================
# Exact developer matching
# ============================================================


#  --- Extract all developer matches ---
def extract_all_developers(text, pattern):
    """
    Extract all developer names matched by a compiled regex pattern.

    Parameters
    ----------
    text : str
        Text in which developer names are searched.

    pattern : re.Pattern
        Compiled regular-expression pattern containing the developer names.

    Returns
    -------
    list
        Unique developer matches found in the text, preserving match order.
    """
    matches = [m.group(1) for m in pattern.finditer(text)]
    
    # Preserve order while removing duplicates
    return list(dict.fromkeys(matches))


# --- Combine match lists ---
def combine_matches(row, match_columns):
    """
    Combine developer matches from multiple columns into one unique list.

    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing the match columns.

    match_columns : list
        Columns containing lists of developer matches.

    Returns
    -------
    list
        Unique developer candidates, preserving their original order.
    """
    matches = []

    for col in match_columns:
        matches.extend(row[col])
    
    return list(dict.fromkeys(matches))


# --- Inspect exact-match conflicts ---
def inspect_exact_conflict(df, dev1, dev2, n=20):
    """
    Summarize transactions containing a specific pair of
    conflicting developer candidates.
    """
    mask = df['developer_candidates'].apply(
        lambda x: set(x) == {dev1, dev2}
    )

    result = df.loc[
        mask,
        [
            'building',
            'project',
            'master_project',
            'building_exact',
            'project_exact',
            'developer_candidates'
        ]
    ].copy()

    # Convert list columns to tuples so pandas can group/hash them
    for col in [
        'building_exact',
        'project_exact',
        'developer_candidates'
    ]:
        result[col] = result[col].apply(tuple)

    result = (
        result
        .value_counts(
            subset=[
                'building',
                'project',
                'master_project',
                'building_exact',
                'project_exact',
                'developer_candidates'
            ]
        )
        .reset_index(name='transactions')
        .sort_values('transactions', ascending=False)
        .reset_index(drop=True)
    )

    return result.head(n)


# --- Assign unambiguous developer ---
def assign_developer(candidates):
    """
    Assign unambiguous exact matches.
    Rows with zero or multiple candidates remain unresolved.
    """
    if len(candidates) == 1:
        return candidates[0]
    else:
        return np.nan


# ============================================================
# Similarity-based propagation
# ============================================================


# --- Find similarity-based matches ---
def find_similarity_matches(df, field, cutoff):
    """
    Find the closest known value for each unresolved value using
    RapidFuzz token-sort similarity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the text field and developer assignments.

    field : str
        Name of the text column to compare, such as 'project' or 'building'.

    cutoff : int or float
        Minimum similarity score required for a candidate match.

    Returns
    -------
    pandas.DataFrame
        Candidate matches with the unmatched value, matched known value,
        suggested developer and similarity score.
    """

    known = (
    df[
        df['developer'].notna()
        & df[field].notna()
        & df[field].str.strip().ne('')
    ]
    [[field, 'developer']]
    .drop_duplicates()
    )

    ambiguous_values = (
        known.groupby(field)['developer']
        .nunique()
    )

    ambiguous_values = ambiguous_values[
        ambiguous_values > 1
    ].index

    known = (
        known[
            ~known[field].isin(ambiguous_values)
        ]
        .drop_duplicates(subset=field)
    )   

    unknown = (
        df[df['developer'].isna()]
        [[field]]
        .dropna()
    )

    unknown = unknown[
        unknown[field].str.strip() != ''
    ].drop_duplicates(subset=field)

    matches = []

    for value in unknown[field]:
        match = process.extractOne(
            value,
            known[field],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=cutoff
        )

        if match:
            matched_value, score, _ = match

            matched_dev = known.loc[
                known[field] == matched_value,
                'developer'
            ].iloc[0]

            matches.append({
                f'unmatched_{field}': value,
                f'matched_{field}': matched_value,
                'developer_suggested': matched_dev,
                'similarity': score
            })

    columns = [
        f'unmatched_{field}',
        f'matched_{field}',
        'developer_suggested',
        'similarity'
    ]

    result = pd.DataFrame(matches, columns=columns)

    if not result.empty:
        result = (
            result
            .sort_values('similarity', ascending=False)
            .reset_index(drop=True)
        )

    return result


# --- Apply similarity-based matches ---
def apply_similarity_matches(
    df,
    matches,
    field,
    excluded_values=None
):
    """
    Propagate developer assignments from validated similarity matches.

    Only rows with a missing developer are updated. Candidate values
    explicitly identified as false or ambiguous matches can be excluded.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the text field and developer assignments.

    matches : pandas.DataFrame
        Candidate matches generated by find_similarity_matches().

    field : str
        Text field used for propagation, such as 'project' or 'building'.

    excluded_values : collection, optional
        Unmatched values that should not be propagated.

    Returns
    -------
    pandas.DataFrame
        DataFrame with validated developer assignments applied.
    """
    if excluded_values is None:
        excluded_values = set()

    unmatched_col = f'unmatched_{field}'

    propagate_map = (
        matches.loc[
            ~matches[unmatched_col].isin(excluded_values)
        ]
        .set_index(unmatched_col)['developer_suggested']
        .to_dict()
    )

    df['developer'] = df['developer'].fillna(
        df[field].map(propagate_map)
    )

    return df


# ============================================================
# Prefix-based similarity propagation
# ============================================================


# --- Split names into prefix and suffix ---
def split_name_prefix(name):
    """
    Split a normalized project or building name at the first
    non-letter/non-space character.

    Parameters
    ----------
    name : str
        Normalized project or building name.

    Returns
    -------
    pandas.Series
        Two values:
        - prefix: text before the first separator, number or symbol
        - suffix: remaining text, or None when no suffix exists
    """
    if pd.isna(name):
        return pd.Series([None, None])

    parts = re.split(r'[^A-Za-z\s]+', name, maxsplit=1)

    prefix = parts[0].strip()

    if len(parts) == 1:
        return pd.Series([prefix, None])

    return pd.Series([prefix, parts[1].strip()])


# --- Find prefix-based similarity matches ---
def find_prefix_similarity_matches(
    df,
    prefix_col,
    cutoff=90,
    min_length=3
):
    """
    Match unresolved prefixes against prefixes with an identified developer.

    - Ignores empty prefixes and prefixes shorter than the minimum length.
    - Excludes known prefixes associated with multiple developers.
    - Uses RapidFuzz token-sort similarity to identify candidate matches.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing prefix and developer columns.

    prefix_col : str
        Name of the prefix column to compare.

    cutoff : int or float, default=90
        Minimum similarity score required for a candidate match.

    min_length : int, default=3
        Minimum number of characters required for a prefix.

    Returns
    -------
    tuple
        matches_df:
            DataFrame containing candidate prefix matches.

        ambiguous_prefixes:
            Series containing known prefixes associated with more than
            one developer.
    """
    # Known prefix/developer combinations
    known = (
        df[
            df['developer'].notna()
            & df[prefix_col].notna()
            & df[prefix_col].str.strip().ne('')
            & df[prefix_col].str.len().ge(min_length)
        ]
        [[prefix_col, 'developer']]
        .drop_duplicates()
    )

    # Identify ambiguous known prefixes
    ambiguous_prefixes = (
        known.groupby(prefix_col)['developer']
        .nunique()
    )

    ambiguous_prefixes = ambiguous_prefixes[
        ambiguous_prefixes > 1
    ]

    # Keep only unambiguous known prefixes
    known_clean = (
        known[
            ~known[prefix_col].isin(ambiguous_prefixes.index)
        ]
        .drop_duplicates(subset=prefix_col)
    )

    # Currently unmatched prefixes
    unknown = (
        df[
            df['developer'].isna()
            & df[prefix_col].notna()
            & df[prefix_col].str.strip().ne('')
            & df[prefix_col].str.len().ge(min_length)
        ]
        [[prefix_col]]
        .drop_duplicates()
    )

    matches = []

    for value in unknown[prefix_col]:

        match = process.extractOne(
            value,
            known_clean[prefix_col],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=cutoff
        )

        if match:
            matched_value, score, _ = match

            matched_dev = known_clean.loc[
                known_clean[prefix_col] == matched_value,
                'developer'
            ].iloc[0]

            matches.append({
                'unmatched_prefix': value,
                'matched_prefix': matched_value,
                'developer_suggested': matched_dev,
                'similarity': score
            })

    columns = [
        'unmatched_prefix',
        'matched_prefix',
        'developer_suggested',
        'similarity'
    ]

    matches_df = pd.DataFrame(matches, columns=columns)

    if not matches_df.empty:
        matches_df = (
            matches_df
            .sort_values('similarity', ascending=False)
            .reset_index(drop=True)
        )

    return matches_df, ambiguous_prefixes


# --- Apply prefix-based similarity matches ---
def apply_prefix_matches(
    df,
    matches_df,
    prefix_col,
    excluded_prefixes=None
):
    """
    Propagate developers from validated prefix-similarity matches.

    Only currently unresolved transactions are updated. Prefixes
    explicitly identified as false or ambiguous matches are excluded.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing prefix and developer columns.

    matches_df : pandas.DataFrame
        Candidate matches generated by find_prefix_similarity_matches().

    prefix_col : str
        Prefix column used for propagation.

    excluded_prefixes : collection, optional
        Prefixes that should not receive propagated assignments.

    Returns
    -------
    pandas.DataFrame
        DataFrame with validated prefix-based developer assignments applied.
    """

    if excluded_prefixes is None:
        excluded_prefixes = []

    excluded_prefixes = {
        value.lower()
        for value in excluded_prefixes
    }

    valid_matches = matches_df[
        ~matches_df['unmatched_prefix']
        .str.lower()
        .isin(excluded_prefixes)
    ].copy()

    mapping_dict = (
        valid_matches
        .set_index('unmatched_prefix')['developer_suggested']
        .to_dict()
    )

    mask = (
        df['developer'].isna()
        & df[prefix_col].isin(mapping_dict)
    )

    df.loc[mask, 'developer'] = (
        df.loc[mask, prefix_col]
        .map(mapping_dict)
    )

    print(
        f"{mask.sum():,} transactions assigned through "
        f"{prefix_col} propagation."
    )

    return df