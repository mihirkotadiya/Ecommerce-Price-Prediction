import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
##-----------------Datasets Load---------------------##
def load_data():
    customers = pd.read_csv(DATA_DIR / "customers_dataset.csv")
    geolocation = pd.read_csv(DATA_DIR / "geolocation_dataset.csv")
    items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(DATA_DIR / "order_payments_dataset.csv")
    reviews = pd.read_csv(DATA_DIR / "order_reviews_dataset.csv")
    orders = pd.read_csv(DATA_DIR / "orders_dataset.csv")
    product_category = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")
    products = pd.read_csv(DATA_DIR / "products_dataset.csv")
    sellers = pd.read_csv(DATA_DIR / "sellers_dataset.csv")

    #-------MERGE(items is main table)----------
    df = items.copy()
    df = df.merge(orders, on='order_id', how='left')
    df = df.merge(products, on='product_id', how='left')
    df = df.merge(customers, on='customer_id', how='left')
    df = df.merge(sellers, on='seller_id', how='left')

    reviews_agg = reviews.groupby('order_id').agg({
        'review_score':'mean',
        'review_comment_message':'count'
    }).reset_index()

    df = df.merge(reviews_agg,on = 'order_id', how='left')

    payments_agg = payments.groupby('order_id').agg({
        'payment_value': 'sum',
        'payment_type': 'first'
    }).reset_index()

    df = df.merge(payments_agg, on='order_id', how='left')

    #---DROP USELESS COLUMNS----#
    df.drop(columns=['order_id', 'order_item_id', 'product_id', 'seller_id',
                            'customer_id', 'customer_unique_id', 'review_comment_message'], inplace=True)


    num_cols = df.select_dtypes(include = ['int64', 'float64']).columns

    #----Fill Missing values----
    float_col = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
                'review_score', 'payment_value']
    for col in float_col:
        df[col] = df[col].fillna(df[col].median())
        
    obj_col = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
            'product_category_name', 'payment_type']
    for col in obj_col:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df