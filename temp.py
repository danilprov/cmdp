import matplotlib.pyplot as plt

# Example data
x = range(0, 31)
y1 = [val**2 for val in x]
y2 = [val**2 + val for val in x]

marker_indices_line1 = [0, 10, 20]  # Indices for markers on Line 1
marker_indices_line2 = [5, 15, 25]  # Indices for markers on Line 2

plt.plot(x, y1, label='Line 1', marker='o', markevery=marker_indices_line1)
plt.plot(x, y2, label='Line 2', marker='s', markevery=marker_indices_line2)

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()