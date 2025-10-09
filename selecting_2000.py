import pandas as pd #import pandas library to be able to work with excel

try:
    # reads our dataset with 50,000 rows/recipe
    df = pd.read_excel("modifid_data.xlsx")
    print("Excel file loaded successfully. Total rows:", len(df))

    #samples 2000 random rows from our dataset
    sampled_df = df.sample(n=2000, random_state=42)
    print("✅ Sampled 2000 rows.")

    #enters the sample into a new excel
    sampled_df.to_excel("sampled_recipes.xlsx", index=False)
    print("Saved to 'sampled_recipes.xlsx'.")

except FileNotFoundError:
    print("File not found. Make sure 'big_recipes.xlsx' is in the same folder as this script.")

except Exception as e:
    print("An unexpected error occurred:", e)