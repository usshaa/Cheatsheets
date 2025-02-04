📊 Matplotlib Cheatsheet 🎨
Matplotlib is a powerful Python library for creating static, animated, and interactive visualizations.
________________________________________
🔹 1. Importing Matplotlib
import matplotlib.pyplot as plt
import numpy as np
________________________________________
🔹 2. Basic Plot
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]

plt.plot(x, y)  # Line plot
plt.title("Basic Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
________________________________________
🔹 3. Customizing Plots
plt.plot(x, y, color='red', linestyle='--', marker='o', markersize=8, linewidth=2, label="Data")
plt.title("Customized Plot", fontsize=14)
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()
________________________________________
🔹 4. Multiple Plots in One Figure
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="Sine Wave", color="blue")
plt.plot(x, y2, label="Cosine Wave", color="green")
plt.legend()
plt.show()
________________________________________
🔹 5. Subplots (Multiple Plots in Grid)
fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of plots

axs[0, 0].plot(x, y1, 'b')  # Top-left
axs[0, 0].set_title("Sine Wave")

axs[0, 1].plot(x, y2, 'r')  # Top-right
axs[0, 1].set_title("Cosine Wave")

axs[1, 0].scatter(x, y1, color='purple')  # Bottom-left
axs[1, 0].set_title("Scatter Plot")

axs[1, 1].bar([1, 2, 3], [5, 7, 3], color='orange')  # Bottom-right
axs[1, 1].set_title("Bar Chart")

plt.tight_layout()  # Adjust spacing
plt.show()
________________________________________
🔹 6. Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color='red', marker='x', s=100, alpha=0.7)
plt.title("Scatter Plot")
plt.show()
________________________________________
🔹 7. Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

plt.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
plt.title("Bar Chart Example")
plt.show()
________________________________________
🔹 8. Histogram
data = np.random.randn(1000)  # 1000 random numbers

plt.hist(data, bins=20, color='purple', edgecolor='black', alpha=0.7)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
________________________________________
🔹 9. Pie Chart
labels = ['A', 'B', 'C', 'D']
sizes = [40, 30, 20, 10]
colors = ['gold', 'lightblue', 'lightcoral', 'green']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart Example")
plt.show()
________________________________________
🔹 10. Adding Text & Annotations
plt.plot(x, y1, label="Sine Wave")
plt.text(2, 0.5, "Peak Point", fontsize=12, color='red')
plt.annotate("Local Min", xy=(7.85, -1), xytext=(6, -0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.legend()
plt.show()
________________________________________
🔹 11. Saving a Figure
plt.plot(x, y1, label="Sine Wave")
plt.title("Sine Wave Example")
plt.savefig("plot.png", dpi=300, bbox_inches='tight')  # Save as PNG
plt.show()
________________________________________
🔹 12. Logarithmic Scale
x = np.linspace(1, 10, 100)
y = np.exp(x)

plt.plot(x, y)
plt.yscale('log')  # Set Y-axis to log scale
plt.title("Logarithmic Scale")
plt.show()
________________________________________
🔹 13. 3D Plot (Matplotlib + mpl_toolkits)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
________________________________________
🎯 Summary of Common Plot Types
Plot Type	Function
Line Plot	plt.plot()
Scatter Plot	plt.scatter()
Bar Chart	plt.bar()
Histogram	plt.hist()
Pie Chart	plt.pie()
Box Plot	plt.boxplot()
Error Bars	plt.errorbar()
3D Plot	Axes3D.plot()
________________________________________
🔥 Matplotlib is a powerful library for visualizing data. Use these examples to create beautiful plots! 🚀

