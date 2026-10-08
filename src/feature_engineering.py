import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok= True)


def feature_engineering(df):
    #-------------------------Feature Engineering-------------------------------
    date_col = ['order_delivered_carrier_date', 'shipping_limit_date', 'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_customer_date', 'order_estimated_delivery_date']

    for col in date_col:
        df[col] = pd.to_datetime(df[col])

    #Purchase time feature, useful features from list
    df['purchase_year'] = df['order_purchase_timestamp'].dt.year
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_day'] = df['order_purchase_timestamp'].dt.day
    df['purchase_weekday'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    #weekend feature
    df['is_weekend'] = (df['purchase_weekday'] >= 5).astype(int)

    #Delivery Delay
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    #Delivery related features
    df['approval_delay'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds()/3600
    df['shipping_delay_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days
    df['actual_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days

    #Carrier delay
    df['carrier_delay'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.days

    #Delivery perfomance
    df['estimated_gap'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    df['delivery_gap'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days

    #shipping deadlines Feature
    df['shipping_window_days'] = (df['shipping_limit_date'] - df['order_purchase_timestamp']).dt.days


    #----------------BUSINESS FEATURES--------------------
    #1. Product volume
    df['product_volume'] = (df['product_length_cm']*df['product_height_cm']*df['product_width_cm'])

    #2. Product Density
    df['product_density'] = (df['product_weight_g']/(df['product_volume'] + 1))

    #3. Freight per volume
    df['freight_per_kg'] =(df['freight_value']/(df['product_weight_g'] + 1))

    #5. Delivery Difference
    df['delivery_difference'] = (df['estimated_gap'] - df['actual_delivery_days'])

    #6. Heavy Product
    df['heavy_product'] = (df['product_weight_g'] > 5000).astype(int)


    #same state
    df['same_state'] = (df['customer_state'] == df['seller_state']).astype(int)

    #same zip prefix
    df['same_zip'] = (df['customer_zip_code_prefix'] == df['seller_zip_code_prefix']).astype(int)

    #large product
    threshold = df['product_volume'].median()

    df['large_product'] = (df['product_volume'] > threshold).astype(int)

    #Heavy + large
    df['heavy_largr'] = ((df['heavy_product'] == 1) & (df['large_product'] == 1)).astype(int)

    #rush shipping
    df['rush_shipping'] = (df['shipping_window_days'] <= 2).astype(int)

    #late delivery
    df['late_delivery'] = (df['delivery_gap'] > 0).astype(int)

    #Fast approval
    df['fast_approval'] = (df['approval_delay']<1).astype(int)

    #cubic  weight
    df['cubic_weight'] = (df['product_volume']/6000)

    #weekend night purchase
    df['weekend_night'] = ((df['is_weekend'] == 1) & (df['purchase_hour'] >= 18)).astype(int)

    #business hours
    df['business_hour'] = (
        (df['purchase_hour']>=9) &
        (df['purchase_hour'] <= 18)
    ).astype(int)

    #Original columns drop

    df.drop(columns = date_col, inplace=True)
    print("="*70)
    print(f"\nE-COMMERCE PRICE PREDICTION USING MACHINE LEARNING")
    print("="*70)

    #Encoding
    categorical_cols = [
        'order_status',
        'product_category_name',
        'customer_state',
        'seller_state',
        'payment_type'
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df.drop(columns=['customer_city', 'seller_city'], inplace=True, errors='ignore')

    # Convert any remaining non-numeric columns to dummy variables if present
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if remaining_object_cols:
        df = pd.get_dummies(df, columns=remaining_object_cols, drop_first=True)

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        data = df,
        x = 'delivery_delay',
        y = 'price',
        alpha = 0.4
    )
    plt.title("Delivery Delay vs Price")
    plt.savefig(os.path.join(IMAGE_DIR, "delivery_delay_vs_price.png"), dpi = 300, bbox_inches = "tight")
    plt.show()


    #Model Training
    X = df.drop(columns = ['price'], axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nDataset Shape {df.shape}")
    print(f"\nTrain : {len(X_train)}")
    print(f"\nTest : {len(X_test)}")
    print(f"\nFinal Features: {X.shape}")

    print(f"\nModel Perfomance")
    print("-"*70)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return (
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_s,
    X_test_s
)