from statistics import mean, stdev

def generate_pascal_row(n):
    """Generate the nth row of Pascal's triangle"""
    row = [1]
    for i in range(n):
        row.append(row[i] * (n - i) // (i + 1))
    return row


row = generate_pascal_row(100000)
max_elem = max(row)
row = [e/max_elem for e in row]

m = mean(row)
std_dev = stdev(row)
print(f'{m:.10f}'
      f'\n{std_dev:.10f}')
