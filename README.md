# 2D Pathfinding with Supervised Learning

This is an independent research project designed to experiement with influencing path creation in 2D grid space based on outside factors, such as the "hastiness" of the model or its "interest" in specific landmarks on the grid.

There are two modes to this data collection: distance and interest. In the distance mode, the algorithm will simply compute the shortest distance using A* and save that path data for model training.

In the interest mode, the pathfinder takes into account how "interesting" each cell in the grid is. A random number of "landmarks" or "points of interest (pois)" are placed inside the grid, each with a random "interest" level (an integer in the range [0, 10)). A gradient is calculated between those points to generate the "interest" values for the remaining cells. A modified version of A* is used to factor in a cell's "interest" value when generating paths. This is akin to taking the scenic route.

If we represent the distance mode as a 0 and the interest mode as a 1, this value can now be thought of as the model's "curiosity".

Running `python3 board.py` will generate a random 2D grid/board, alternate path generation between each mode, and save the data to `data.csv`. Running `python3 train.py` afterwards will process the data and train the neural network.

A lookback of 4 was used to provide context to the model. After training, an accuracy of at least 99% is expected.

This shows that it is possible to influence how a neural network traverses a 2D grid based on how "curious" it is and how "interesting" its surroundings are.