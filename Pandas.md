📌 Pandas Cheatsheet 🐼
Pandas is a powerful Python library for data analysis and manipulation.
________________________________________
1️⃣ Import Pandas
import pandas as pd
________________________________________
📂 Creating DataFrames
# From Dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'], 
        'Age': [25, 30, 35], 
        'Salary': [50000, 60000, 70000]}
df = pd.DataFrame(data)

# From List of Lists
df2 = pd.DataFrame([['Alice', 25, 50000], 
                    ['Bob', 30, 60000]], 
                   columns=['Name', 'Age', 'Salary'])

# From CSV / Excel
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")

# From Dictionary with Index
df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])

# Display first 5 rows
print(df.head())

# Display last 5 rows
print(df.tail())
________________________________________
🔍 Basic Data Exploration
print(df.shape)       # (rows, columns)
print(df.columns)     # Column names
print(df.dtypes)      # Data types of each column
print(df.info())      # Summary of the DataFrame
print(df.describe())  # Statistics summary for numerical data
print(df.nunique())   # Count of unique values per column
________________________________________
🎯 Indexing & Selection
# Select a Single Column
print(df['Name'])   # Returns Series
print(df[['Name', 'Salary']])  # Returns DataFrame

# Select Rows by Index
print(df.iloc[0])   # First row (integer position)
print(df.iloc[0:2]) # First two rows

# Select Row by Label
print(df.loc[0])   # First row (by index label)
print(df.loc[df['Age'] > 30])  # Filter rows

# Boolean Indexing
print(df[df['Salary'] > 55000])
________________________________________
🔄 Adding & Removing Columns
# Add a New Column
df['Bonus'] = df['Salary'] * 0.1

# Remove a Column
df.drop('Bonus', axis=1, inplace=True)  # Drop column permanently

# Drop Rows
df.drop(0, axis=0, inplace=True)  # Drop first row
________________________________________
🔄 Handling Missing Values
# Check Missing Values
print(df.isnull().sum())

# Fill Missing Values
df.fillna(value=0, inplace=True)  # Replace NaN with 0
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Replace NaN with mean

# Drop Missing Values
df.dropna(inplace=True)  # Remove rows with NaN
________________________________________
🔢 Sorting & Filtering
# Sort by a Column
df.sort_values('Age', ascending=False, inplace=True)

# Filter Data
df_filtered = df[(df['Age'] > 25) & (df['Salary'] > 50000)]
________________________________________
🔄 Grouping & Aggregation
# Group by Column
grouped = df.groupby('Age').mean()

# Aggregations
df.agg({'Salary': ['min', 'max', 'mean'], 'Age': 'count'})
________________________________________
🔁 Merging, Joining, Concatenation
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Salary': [60000, 70000, 80000]})

# Merge on a Common Column
merged_df = pd.merge(df1, df2, on='ID', how='inner')  # inner, left, right, outer

# Concatenate DataFrames
concat_df = pd.concat([df1, df2], axis=0)  # axis=0 for rows, axis=1 for columns
________________________________________
🔄 Applying Functions
# Apply Function to a Column
df['Age_Category'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Old')

# Apply Function to Entire DataFrame
df = df.applymap(lambda x: x*2 if isinstance(x, (int, float)) else x)
________________________________________
🔠 String Operations
df['Name'] = df['Name'].str.upper()   # Convert to uppercase
df['Name'] = df['Name'].str.replace('A', 'X')  # Replace 'A' with 'X'
df['Name_Length'] = df['Name'].str.len()  # Get string length
________________________________________
📊 Pivot Tables & Crosstab
# Pivot Table
pivot = df.pivot_table(index='Age', values='Salary', aggfunc='mean')

# Crosstab
cross = pd.crosstab(df['Age'], df['Salary'])
________________________________________
📈 Visualization with Pandas
import matplotlib.pyplot as plt

df['Age'].hist()  # Histogram
df.plot(kind='bar', x='Name', y='Salary')  # Bar Chart
df.plot(kind='line', x='Age', y='Salary')  # Line Plot

plt.show()  # Show the plots
________________________________________
🏁 Saving & Loading Data
df.to_csv("output.csv", index=False)  # Save as CSV
df.to_excel("output.xlsx", index=False)  # Save as Excel
df.to_json("output.json", orient="records")  # Save as JSON
________________________________________
💡 Use Cases
✔️ Data Cleaning
✔️ Data Analysis
✔️ Financial Modeling
✔️ Machine Learning Preprocessing
________________________________________
🔹 This Pandas cheatsheet covers essential functions with examples. 🚀

