from . import *

def load_data(path:str):
    """
    Loads our data to notebook/py file by renaming columns and deleting redundant columns.

    Parameters
    ----------
    path: str
        Absolute path or csv file name of the dataset.

    Returns
    -------
    df: pd.DataFrame
        Our dataset converted to pandas dataframe.
    """
    df = pd.read_csv(path,index_col=0)

    # Renaming columns by adding underscore then by making their letters to lowercase thanks to regexp.
    # Example: minEngineYear -> min_engine_year
    df.columns = [re.sub("(?<!^)(?=[A-Z])", "_", col).lower() for col in df.columns]
    df.rename(columns={"total_h_p": "total_hp"}, inplace=True)

    # Dropping redundant columns
    df = df.drop(["id", "seller_id", "zip", "engine_category", "created_date", "created_month", "created_year", "max_engine_year", "min_engine_year","make","model","engine_category",
     "city","state","dry_weight_lb"],axis=1)

    return df

def grab_cols(df:pd.DataFrame,summary=False):
    """
    Captures numeric and categoric variables.

    Parameters
    ----------
    df: pd.DataFrame
    summary: bool, default=False
        Set True if you want to see the result.

    Returns
    -------
    num_cols: list[str]
        Numeric variables.
    cat_cols: list[str]
        Categoric (Object) variables.
    """
    # Grabs columns
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    # Printing summary
    if summary:
        print(" SUMMARY ".center(60, "*"))
        print(f"# of categoric variables: {len(cat_cols)}")
        print(list(cat_cols))
        print(f"# of numeric variables: {len(num_cols)}")
        print(list(num_cols))

    return num_cols, cat_cols

def cat_summary(df:pd.DataFrame, catcol:str):
    """
    Summarizes categoric distribution by plotting bar chart showing counts and pie chart showing ratios.
    After these plots, prints the situation against our dependent/target variable "price" using pd.groupby().

    Parameters
    ----------
    df: pd.DataFrame
    catcol: str
        Categoric variable is to be observed.
    """
    # Bar Chart
    plt.suptitle(f"{catcol.upper()}")
    plt.subplot(1, 2, 1)
    counts = df[catcol].value_counts().reset_index()
    ax = sns.barplot(counts,
                     x=catcol,
                     y="count",
                     hue=catcol,
                     legend=False)

    # For showing values on top of the bars
    for _,container in enumerate(ax.containers):
        ax.bar_label(container)

    plt.xticks(rotation=90)

    # Pie Chart
    plt.subplot(1, 2, 2)
    plt.pie(counts["count"],
            labels=counts[catcol],
            autopct="%1.1f%%")
    plt.show(block=True)

    # Printing groupby cat vs target
    print(df.groupby(catcol)["price"].mean(), end="\n\n")

def num_summary(df:pd.DataFrame, numcol:str):
    """
    Summarizes numeric distribution by plotting histogram chart showing distributions and
    box chart showing outliers.

    Parameters
    ----------
    df: pd.DataFrame
    numcol: str
        Numeric variable is to be observed.
    """
    plt.suptitle(f"{numcol.upper()}")

    # Histogram chart
    plt.subplot(1,2,1)
    sns.histplot(df,x=numcol,kde=True)

    # Box chart
    plt.subplot(1,2,2)
    sns.boxplot(df,x=numcol)

    plt.show(block=True)

def corr_analysis(df:pd.DataFrame,numcols:list):
    """
    Correlation analysis plotting correlation matrix of the numerical variables of the dataframe.

    Parameters
    ----------
    df: pd.DataFrame
    numcols: list[str]
        Numerical variables of the dataframe
    """
    corr = df[numcols].corr()
    sns.heatmap(corr,mask=np.triu(corr),linewidths=0.3,cmap="RdBu",annot=True)
    plt.show(block=True)