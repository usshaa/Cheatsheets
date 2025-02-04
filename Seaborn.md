📊 Seaborn Cheatsheet 🎨
Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.
________________________________________
🔹 1. Importing Seaborn & Setup
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set a theme
sns.set_theme(style="darkgrid")
________________________________________
🔹 2. Load Sample Dataset
Seaborn comes with built-in datasets.
# Load an example dataset
df = sns.load_dataset("tips")
print(df.head())
________________________________________
🔹 3. Scatter Plot
sns.scatterplot(x="total_bill", y="tip", data=df, hue="sex", style="time", size="size")
plt.title("Scatter Plot of Total Bill vs Tip")
plt.show()
________________________________________
🔹 4. Line Plot
sns.lineplot(x="day", y="total_bill", data=df, hue="sex", marker="o")
plt.title("Line Plot of Total Bill Over Days")
plt.show()
________________________________________
🔹 5. Bar Plot
sns.barplot(x="day", y="total_bill", data=df, hue="sex", palette="coolwarm")
plt.title("Average Total Bill by Day")
plt.show()
________________________________________
🔹 6. Count Plot (Categorical Frequency)
sns.countplot(x="day", data=df, hue="sex", palette="viridis")
plt.title("Count of Visits Per Day")
plt.show()
________________________________________
🔹 7. Histogram & KDE Plot
sns.histplot(df["total_bill"], bins=20, kde=True, color="blue")
plt.title("Histogram of Total Bill")
plt.show()
________________________________________
🔹 8. Box Plot (Detect Outliers)
sns.boxplot(x="day", y="total_bill", data=df, hue="sex", palette="pastel")
plt.title("Box Plot of Total Bill by Day")
plt.show()
________________________________________
🔹 9. Violin Plot (Combining Boxplot & KDE)
sns.violinplot(x="day", y="total_bill", data=df, hue="sex", split=True)
plt.title("Violin Plot of Total Bill by Day")
plt.show()
________________________________________
🔹 10. Swarm Plot (Categorical Distribution)
sns.swarmplot(x="day", y="total_bill", data=df, hue="sex", dodge=True)
plt.title("Swarm Plot of Total Bill")
plt.show()
________________________________________
🔹 11. Pair Plot (Pairwise Relationships)
sns.pairplot(df, hue="sex", palette="husl")
plt.show()
________________________________________
🔹 12. Heatmap (Correlation Matrix)
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
________________________________________
🔹 13. Facet Grid (Multiple Subplots)
g = sns.FacetGrid(df, col="sex", row="time", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
plt.show()
________________________________________
🔹 14. Regression Plot (Line of Best Fit)
sns.regplot(x="total_bill", y="tip", data=df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Regression Plot of Total Bill vs Tip")
plt.show()
________________________________________
🔹 15. Style & Customization
sns.set_style("whitegrid")  # Set style
sns.set_palette("Set2")  # Change color palette

sns.barplot(x="day", y="total_bill", data=df)
plt.title("Styled Bar Plot")
plt.show()
________________________________________
🎯 Summary of Common Seaborn Plot Types
Plot Type	Function
Scatter Plot	sns.scatterplot()
Line Plot	sns.lineplot()
Bar Chart	sns.barplot()
Count Plot	sns.countplot()
Histogram	sns.histplot()
KDE Plot	sns.kdeplot()
Box Plot	sns.boxplot()
Violin Plot	sns.violinplot()
Swarm Plot	sns.swarmplot()
Pair Plot	sns.pairplot()
Heatmap	sns.heatmap()
Regression Plot	sns.regplot()
________________________________________
🔥 Seaborn makes data visualization elegant and easy! Use these examples to create beautiful and insightful plots! 🚀

