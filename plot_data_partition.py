import matplotlib.pyplot as plt

epochs = [1, 5, 10, 15]

splits = [0.00, 0.25, 0.50, 0.75, 1.00]

accuracies = {
    0.00: [0.01, 0.05, 0.08, 0.09],
    0.25: [0.05, 0.10, 0.30, 0.30],
    0.50: [0.10, 0.20, 0.48, 0.65],
    0.75: [0.18, 0.30, 0.60, 0.78],
    1.00: [0.20, 0.40, 0.70, 0.88]
}

colors = {
    0.00: 'purple',
    0.25: 'blue',
    0.50: 'green',
    0.75: 'orange',
    1.00: 'red'
}

for split in splits:
    plt.plot(epochs, accuracies[split], label=split, color=colors[split])
plt.legend(splits, title="Uniformity Score")
plt.title("Algorithm 1 Data Partitioning Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()