from . import *

def fill_nulls(df:pd.DataFrame):
    """
    Fills missing values based on their corresponding categoric values.

    Parameters
    ----------
    df: pd.DataFrame
    """
    # Grabbing categoric columns having no null value
    no_nulls = [col for col in df.select_dtypes(exclude="number").columns if df[col].isna().sum() == 0]
    no_nulls = list(df[no_nulls].nunique().sort_values().index)

    # Grabbing columns having null value
    nulls_num = [col for col in df.select_dtypes("number").columns if df[col].isna().sum() != 0]
    nulls_obj = [col for col in df.select_dtypes(exclude="number") if col not in no_nulls]

    # Filling null values iteratively by gradually decreasing no_nulls; filled by median on numerics, by mode on categorics
    for i in range(len(no_nulls), 1, -1):
        no_nulls_subset = no_nulls[:i]
        for col in nulls_num:
            df[col].fillna(df.groupby(no_nulls_subset)[col].transform("median"), inplace=True)

        for col in nulls_obj:
            df[col].fillna(df.groupby(no_nulls_subset)[col].transform(lambda x: x.mode()[0] \
                if not x.mode().empty else np.nan), inplace=True)

def outlier_fences(df:pd.DataFrame,col:str):
    """
    Calculates outlier fences to determine outliers using interquartile range (iqr).
    I prefer q1=0.05 and q3=0.95 instead of q1=0.25 and q3=0.75 since LightGBM is robust to outliers (generally).

    Parameters
    ----------
    df: pd.DataFrame
    col: str
       Numeric column whose outlier fences are to be determined.

    Returns
    -------
    low: float
        Lower fence.
    up: float
        Upper fence.
    """
    q1 = df[col].quantile(0.05)
    q3 = df[col].quantile(0.95)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return low, up

def check_outliers(df:pd.DataFrame,col:str):
    """
    Returns true if there are any outliers for variable col.

    Parameters
    ----------
    df: pd.DataFrame
    col: str
        Numeric column is to be checked.

    Returns
    -------
    bool:
        True if there are any outliers, False otherwise.
    """
    low, up = outlier_fences(df,col)
    return df.query(f"{col} > {up} | {col} < {low}").shape[0] > 0

def replace_outliers(df:pd.DataFrame,col:str):
    """
    Replaces outliers assigning lower and upper fences.

    Parameters
    ----------
    df: pd.DataFrame
    col: str
        Numeric column whose outliers are to be replaced.
    """
    low, up = outlier_fences(df,col)
    df.loc[df[col] < low, col] = low
    df.loc[df[col] > up, col] = up

def get_outliers(df:pd.DataFrame,col_name:str,get_index=False):
    """
    Grabs either outliers or their indexes, can be used for showing number of outliers for numeric column.

    Parameters
    ----------
    df: pd.DataFrame
    col_name: str
        Numeric column whose outliers are to be showed.
    get_index: bool, default=False
        Set False if you need to observe outliers and their values. Otherwise, returns only their indexes.

    Returns
    -------
        pd.DataFrame:
            Outliers if get_index==False.
        pd.Index:
            Indexes of outliers if get_index==True.
    """
    low, up = outlier_fences(df,col_name)
    outliers = df.query(f"({col_name} < {low}) | ({col_name} > {up})")
    if get_index:
        return outliers.index
    else:
        return outliers