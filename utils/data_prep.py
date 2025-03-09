from . import pd,np

def fill_nulls(df: pd.DataFrame):
    no_nulls = [col for col in df.select_dtypes(exclude="number").columns if df[col].isna().sum() == 0]
    no_nulls = list(df[no_nulls].nunique().sort_values().index)

    nulls_num = [col for col in df.select_dtypes("number").columns if df[col].isna().sum() != 0]
    nulls_obj = [col for col in df.select_dtypes(exclude="number") if col not in no_nulls]

    for i in range(len(no_nulls), 1, -1):
        no_nulls_subset = no_nulls[:i]
        for col in nulls_num:
            df[col].fillna(df.groupby(no_nulls_subset)[col].transform("median"), inplace=True)

        for col in nulls_obj:
            df[col].fillna(df.groupby(no_nulls_subset)[col].transform(lambda x: x.mode()[0] \
                if not x.mode().empty else np.nan), inplace=True)

def outlier_fences(df:pd.DataFrame,col):
    q1 = df[col].quantile(0.05)
    q3 = df[col].quantile(0.95)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return low, up

def check_outliers(df:pd.DataFrame,col):
    low, up = outlier_fences(df,col)
    return df.query(f"{col} > {up} | {col} < {low}").shape[0] > 0

def replace_outliers(df:pd.DataFrame,col):
    low, up = outlier_fences(df,col)
    df.loc[df[col] < low, col] = low
    df.loc[df[col] > up, col] = up

def get_outliers(df:pd.DataFrame,col_name:str,get_index=False,**kwargs):
    low, up = outlier_fences(df,col_name,**kwargs)
    outliers = df.query(f"({col_name} < {low}) | ({col_name} > {up})")
    if get_index:
        return outliers.index
    else:
        return outliers