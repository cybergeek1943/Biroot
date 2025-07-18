import matplotlib.pyplot as plt


def generate_pascal_row(n):
    """Generate the nth row of Pascal's triangle"""
    row = [1]
    for i in range(n):
        row.append(row[i] * (n - i) // (i + 1))
    return row

row = generate_pascal_row(50)
max_elem = max(row)
row = [e/max_elem for e in row]

# Plotting
plt.plot(row, marker='o')
plt.title("Data Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
