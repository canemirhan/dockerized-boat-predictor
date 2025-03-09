from . import *

def load_data(path):
    df = pd.read_csv(path,index_col=0)

    df.columns = [re.sub("(?<!^)(?=[A-Z])", "_", col).lower() for col in df.columns]
    df.rename(columns={"total_h_p": "total_hp"}, inplace=True)

    df = df.drop(["id", "seller_id", "zip", "engine_category", "created_date", "created_month", "created_year", "max_engine_year", "min_engine_year","make","model","engine_category",
     "city","state","dry_weight_lb"],axis=1)
    return df

def grab_cols(df:pd.DataFrame,summary=False):
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    if summary:
        print(" SUMMARY ".center(60, "*"))
        print(f"# of categoric variables: {len(cat_cols)}")
        print(list(cat_cols))
        print(f"# of numeric variables: {len(num_cols)}")
        print(list(num_cols))

    return num_cols, cat_cols

def cat_summary(df:pd.DataFrame, catcol:str):
    plt.suptitle(f"{catcol.upper()}")
    plt.subplot(1, 2, 1)
    counts = df[catcol].value_counts().reset_index()
    ax = sns.barplot(counts,
                     x=catcol,
                     y="count",
                     hue=catcol,
                     legend=False)
    for i in range(df[catcol].nunique()):
        ax.bar_label(ax.containers[i])
    plt.xticks(rotation=90)
    plt.subplot(1, 2, 2)
    plt.pie(counts["count"],
            labels=counts[catcol],
            autopct="%1.1f%%")
    plt.show(block=True)

    print(df.groupby(catcol)["price"].mean(), end="\n\n")

def corr_analysis(df:pd.DataFrame,numcols:list):
    corr = df[numcols].corr()
    sns.heatmap(corr,mask=np.triu(corr),linewidths=0.3,cmap="RdBu",annot=True)
    plt.show(block=True)