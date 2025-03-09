from . import *

from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def feat_extract(df:pd.DataFrame):
    df["class"] = df["boat_class"].apply(lambda x: re.findall("-(.+)", x)[0])
    df.drop("boat_class", axis=1, inplace=True)

    df["lenght_beam"] = df["length_ft"] * df["beam_ft"]

    df["hp_per_lb"] = df["total_hp"] / df["lenght_beam"]

    df["hp_per_engine"] = df["total_hp"] / df["num_engines"]
    df["hp_per_engine"].fillna(0, inplace=True)

    df["boat_age"] = 2025 - df["year"]
    df["boat_age"] = df["boat_age"].astype("int")
    df.drop("year", axis=1, inplace=True)

    return df

def rare_encoding(df:pd.DataFrame,col:str,th:float, rare:str):
    ratios = df[col].value_counts(normalize=True)
    rares = ratios[ratios <= th].index
    df[col] = np.where(df[col].isin(rares), rare, df[col])

def feat_select(x, y, min, max):
    lgb = LGBMRegressor(verbose=0)
    errors = []
    for i in range(min, max + 1):
        feat_selector = SelectKBest(mutual_info_regression, k=i)
        feat_selector.fit(x, y)

        x_selected = feat_selector.transform(x)
        cv = cross_validate(lgb, x_selected, y, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error")
        errors.append(np.mean(-cv["test_score"]))

    ax = sns.barplot(x=np.arange(min, max + 1), y=errors)
    plt.xlabel("Best K Feature")
    plt.ylabel("RMSE")
    for _, container in enumerate(ax.containers):
        ax.bar_label(container, rotation=45)

    plt.show()


