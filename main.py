import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn_som.som import SOM
from dataCreator import create_sheet, create_data

image_size = 17
n_images = 20

dataset = create_sheet(create_data())["x_0"][100:]
data = []
target = []
for i in range(n_images):
    image = []
    for j in range(image_size):
        image.append(dataset.iloc[image_size * i + j])
        target.append(i % 5)
    data.append(image)
data = np.array(data)

# Build a 5x1 SOM (5 clusters)
som = SOM(m=7, n=1, dim=image_size, random_state=1234)

# Fit it to the data
som.fit(data, shuffle=True, epochs=300000)

# Assign each datapoint to its predicted cluster
predictions_images = som.predict(data)
predictions = []
for i in predictions_images:
    for j in range(image_size):
        predictions.append(i)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
x = [i for i in range(n_images*image_size)]
y = [j for i in data for j in i]
colors = ['red', 'green', 'blue', 'yellow', 'pink', 'grey', 'black']

ax[0].scatter(x, y, c=target, cmap=ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')
plt.show()
