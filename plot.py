from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Plots the accuracies as bar plots

testacc_svm = loadmat('testacc_svm.mat')
testacc_svm = np.squeeze(testacc_svm["testacc_svm"])
testacc_ffn = loadmat('testacc_ffn.mat')
testacc_ffn = np.squeeze(testacc_ffn["testacc_ffn"])
testacc_rnn = loadmat('testacc_rnn.mat')
testacc_rnn = np.squeeze(testacc_rnn["testacc_rnn"])

bar_width = 0.25
x = np.arange(1,7)
plt.bar(x, testacc_svm, bar_width)
plt.bar(x - bar_width, testacc_ffn, bar_width)
plt.bar(x + bar_width, testacc_rnn, bar_width)
plt.legend(['SVM - RBF', 'Feed Forward NN', 'Proposed Model'])
plt.xlabel('Test subject')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for different subjects')
plt.xticks(x)

plt.tight_layout()
plt.savefig('Accuracies.png', dpi = 300)
plt.show()