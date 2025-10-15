"""
graph.py — quick bar chart of how many images per leukocyte class

what this file does:
- takes the final cleaned dataset split counts (hardcoded here)
- makes a simple bar chart showing image count per class
- saves or shows the figure (depending on what’s uncommented)
"""

import matplotlib.pyplot as plt

# quick manual counts from the processed dataset
counts = {
    "Neutrophil": 568,
    "Eosinophil": 380,
    "Lymphocyte": 371,
    "Monocyte": 180,
}

# unpack the dict so we can plot easily
labels = list(counts.keys())
values = list(counts.values())

# basic matplotlib bar chart
plt.figure(figsize=(6, 4))
plt.bar(labels, values)
plt.title("Image Count by Leukocyte Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()

# can either display or save depending on workflow
# plt.savefig("results/class_counts.png", dpi=150)
plt.show()

print("Saved to results/class_counts.png")
