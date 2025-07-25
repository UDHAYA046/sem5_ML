import matplotlib.pyplot as plt

models = ['Custom kNN', 'sklearn kNN']
accuracies = [custom_accuracy, package_accuracy]  # Replace with your values

plt.bar(models, accuracies, color=['skyblue', 'orange'])
plt.ylabel("Accuracy")
plt.title("Custom vs Package kNN Accuracy")
plt.ylim(0, 1)
plt.show()
