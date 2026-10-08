import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok= True)

def perform_eda(df):
    #EDA
    print("="*60)
    print("DATASET INFORMATION")
    print("="*60)

    print(df.shape)
    print(df.info())
    print(df.describe().T)

    print("="*60)
    print("MISSING VALUES")
    print("="*60)

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending = False)

    print(missing)

    if len(missing) > 0:
        plt.figure(figsize=(8, 5))
        missing.plot(kind='bar', color = 'tomato')
        plt.title("Missing Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGE_DIR, "missing_values.png"), dpi = 300, bbox_inches = "tight")
        plt.close()
    else:
        print("="*60)
        print("No Missing Values Found in Dataset.")
        print("="*60)


    print("="*60)
    print("DUPLICATES")
    print("="*60)

    print(df.duplicated().sum())

    plt.figure(figsize = (8,5))

    sns.histplot(df['price'], bins = 40, kde = True)

    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "price_distribution.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize = (8, 5))
    sns.boxplot(x = df['price'])
    plt.title("Price Outliers")
    plt.savefig(os.path.join(IMAGE_DIR, "price_outliers.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize = (12, 10))
    corr = df.select_dtypes(include='number').corr()

    sns.heatmap(
        corr,
        cmap = 'coolwarm',
        center = 0
    )

    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(IMAGE_DIR, "correlation_heatmap.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    corr = df.corr(numeric_only = True)
    df['price'].sort_values()

    print(corr.tail(15))

    plt.figure(figsize = (8,5))

    sns.scatterplot(
        data = df,
        x = 'product_weight_g',
        y = 'price',
        alpha = 0.4
    )
    plt.title("Weight vs Price")
    plt.savefig(os.path.join(IMAGE_DIR, "weight_vs_price.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data = df,
        x = 'freight_value',
        y = 'price',
        alpha = 0.4
    )
    plt.title("Freight value vs Price")
    plt.savefig(os.path.join(IMAGE_DIR, "freight_vsprice.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    plt.figure(figsize = (8, 5))

    sns.countplot(
        data = df,
        x = 'review_score'
    )
    plt.title("Review Score Distribution")
    plt.savefig(os.path.join(IMAGE_DIR, "review_score_distribution.png"), dpi = 300, bbox_inches = "tight")
    plt.show()

    return
