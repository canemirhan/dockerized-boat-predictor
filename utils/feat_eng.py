from . import *

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def feat_extract(df:pd.DataFrame):
    """
    Feature extraction pipeline.

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    df: pd.DataFrame
        New dataframe with added new features.
    """
    # Extracting class name from boat_class column, after dropping boat_class column
    # Example: "power-mega" on boat_class column => "mega" on class column
    df["class"] = df["boat_class"].apply(lambda x: re.findall("-(.+)", x)[0])
    df.drop("boat_class", axis=1, inplace=True)

    df["length_beam"] = df["length_ft"] * df["beam_ft"]

    df["hp_per_lb"] = df["total_hp"] / df["length_beam"]

    # In division, pandas assigns n/0 to NaN, so filling nulls after division
    df["hp_per_engine"] = df["total_hp"] / df["num_engines"]
    df["hp_per_engine"].fillna(0, inplace=True)

    df["boat_age"] = 2025 - df["year"]
    df["boat_age"] = df["boat_age"].astype("int")
    df.drop("year", axis=1, inplace=True)

    return df

def rare_encoding(df:pd.DataFrame, col:str, th:float, rare:str):
    """
    Applies rare encoding for a given categoric column.
    Categories with a relative frequency below the given threshold are replaced with a label.

    Parameters
    ----------
    df: pd.DataFrame
    col: str
        Categoric variable is to be applied rare encoding.
    th: float
        Rare ratio threshold.
    rare: str
        Label is to be replaced to rare categoric values.
    """
    ratios = df[col].value_counts(normalize=True)
    rares = ratios[ratios <= th].index
    df[col] = np.where(df[col].isin(rares), rare, df[col])

def feat_select(x: pd.DataFrame, y:pd.DataFrame, min:int, max:int):
    """
    Feature selection function. I used SelectKBest method which is one of filter methods.
    After FS, plots the result based on rmse for k features within [min,max] interval.

    Parameters
    ----------
    x: pd.DataFrame
        Feature/Independent dataframe
    y: pd.DataFrame
        Target/Dependent dataframe
    min: int
        Min number of features are to be evaluated.
    max: int
        Max number of features are to be evaluated.
    """
    # Initalize model object and errors list to store rmse scores
    lgb = LGBMRegressor(verbose=0)
    errors = []

    # Feature selection loop started from min features to max features
    for i in range(min, max + 1):
        feat_selector = SelectKBest(mutual_info_regression, k=i)
        feat_selector.fit(x, y)

        x_selected = feat_selector.transform(x)
        cv = cross_validate(lgb, x_selected, y, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error")
        errors.append(np.mean(-cv["test_score"]))

    # Plotting results
    ax = sns.barplot(x=np.arange(min, max + 1), y=errors)
    plt.xlabel("Best K Feature")
    plt.ylabel("RMSE")
    for _, container in enumerate(ax.containers):
        ax.bar_label(container, rotation=45)

    plt.show()


