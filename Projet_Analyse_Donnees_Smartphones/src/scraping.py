import pandas as pd
import os
import numpy as np

def create_smartphone_dataset():
    # Format: Brand, Model, Price_DH, RAM_GB, Storage_GB, Screen_Size_Inch, Battery_mAh
    data = [
        # --- SAMSUNG ---
        ["Samsung", "Galaxy A14", 2200, 4, 64, 6.6, 5000],
        ["Samsung", "Galaxy A24", 2800, 6, 128, 6.5, 5000],
        ["Samsung", "Galaxy A54", 4500, 8, 256, 6.4, 5000],
        ["Samsung", "Galaxy S23 Ultra", 12500, 12, 512, 6.8, 5000],
        ["Samsung", "Galaxy S21 FE", 5800, 8, 128, 6.4, 4500],
        
        # --- APPLE ---
        ["Apple", "iPhone 11", 5200, 4, 128, 6.1, 3110],
        ["Apple", "iPhone 13", 8900, 6, 128, 6.1, 3240],
        ["Apple", "iPhone 14 Pro", 13500, 6, 256, 6.1, 3200],
        ["Apple", "iPhone 15", 11500, 6, 128, 6.1, 3349],
        ["Apple", "iPhone SE 2022", 4800, 4, 64, 4.7, 2018],
        
        # --- XIAOMI ---
        ["Xiaomi", "Redmi 12", 1800, 4, 128, 6.79, 5000],
        ["Xiaomi", "Redmi Note 12", 2300, 6, 128, 6.67, 5000],
        ["Xiaomi", "Redmi Note 12 Pro", 3600, 8, 256, 6.67, 5000],
        ["Xiaomi", "Poco F5", 4400, 12, 256, 6.67, 5000],
        ["Xiaomi", "13T Pro", 8500, 12, 512, 6.67, 5000],
        
        # --- OPPO ---
        ["Oppo", "A17", 1700, 4, 64, 6.56, 5000],
        ["Oppo", "A78", 2600, 8, 128, 6.43, 5000],
        ["Oppo", "Reno 8", 4200, 8, 256, 6.4, 4500],
        ["Oppo", "Reno 10 Pro", 6200, 12, 256, 6.7, 4600],
        ["Oppo", "Find X6 Pro", 10500, 12, 256, 6.82, 5000],
        
        # --- REALME ---
        ["Realme", "C55", 2100, 6, 128, 6.72, 5000],
        ["Realme", "11 Pro+", 4500, 12, 512, 6.7, 5000],
        ["Realme", "C33", 1400, 4, 64, 6.5, 5000],
        ["Realme", "GT Neo 5", 5500, 16, 256, 6.74, 5000],
        
        # --- HUAWEI ---
        ["Huawei", "Nova 9", 3900, 8, 128, 6.57, 4300],
        ["Huawei", "P60 Pro", 11000, 12, 256, 6.67, 4815],
        ["Huawei", "Nova 11i", 2900, 8, 128, 6.8, 5000],
        
        # --- INFINIX & TECNO ---
        ["Infinix", "Hot 30", 1600, 8, 128, 6.78, 5000],
        ["Infinix", "Note 30 Pro", 2800, 8, 256, 6.67, 5000],
        ["Tecno", "Spark 10", 1450, 4, 64, 6.6, 5000],
        ["Tecno", "Camon 20", 2400, 8, 256, 6.67, 5000],
        ["Tecno", "Phantom X2", 6800, 8, 256, 6.8, 5160],
        
        # --- GOOGLE & MOTOROLA ---
        ["Google", "Pixel 7a", 5200, 8, 128, 6.1, 4385],
        ["Google", "Pixel 8 Pro", 12000, 12, 256, 6.7, 5050],
        ["Motorola", "Edge 40", 4900, 8, 256, 6.55, 4400],
        ["Motorola", "Moto G54", 2100, 8, 128, 6.5, 5000],
        
        # --- ANOMALIES & MISSING DATA (Added for Task 3a) ---
        ["Samsung", "Galaxy A04", 1300, np.nan, 32, 6.5, 5000],  # Missing RAM
        ["Xiaomi", "Redmi A2", 1100, 2, np.nan, 6.52, 5000],    # Missing Storage
        ["Realme", "Note 50", 1200, 4, 64, np.nan, 5000],      # Missing Screen Size
        ["FakeBrand", "ErrorPhone", 99999, 1, 1, 1.0, 100],     # Price Anomaly/Outlier
    ]

    columns = ["Brand", "Model", "Price_DH", "RAM_GB", "Storage_GB", "Screen_Size_Inch", "Battery_mAh"]

    df = pd.DataFrame(data, columns=columns)

    # Ensure the directory exists
    os.makedirs("../data", exist_ok=True)

    # Save dataset
    df.to_csv("../data/smartphones.csv", index=False)
    print(f" Success! Dataset created with {len(df)} rows.")
    print("  Included 4 'dirty' rows for the cleaning demonstration (Task 3).")

if __name__ == "__main__":
    create_smartphone_dataset()