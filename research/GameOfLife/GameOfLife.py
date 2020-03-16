# Game of Life simulation with Tensorflow 2.0
# See Wikipedia for more detail about the game: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

# Size of life board
LIFE_BOARD_ROWS = 200
LIFE_BOARD_COLS = 200

# Initialize status of life board.
def initLives(lifeRatio = 0.5):
	# First generate a random tensor with value from 0 to 1
	template = tf.random.uniform([LIFE_BOARD_ROWS, LIFE_BOARD_COLS], minval = 0, maxval = 1, dtype = tf.float32)
	# Then convert this to a tensor of lives with values 0 or 1
	return tf.cast(template < lifeRatio, tf.float32)

# The filter to count live neighbors using convolution.
# It is a [3, 3] matrix with values:
# [1, 1, 1]
# [1, 0, 1]
# [1, 1, 1]
# To comply with the convention of tf.nn.convolution(), we reshape it to [3, 3, 1, 1]
NEIGHBOR_SUM_FILTER = tf.reshape(tf.cast(tf.constant([1, 1, 1, 1, 0, 1, 1, 1, 1]), tf.float32),  [3, 3, 1, 1])
	
# Itertion of lives.
# Lives is a matrix of int32
@tf.function
def GenerationOfLives(lives):
	paddings = tf.constant([[1, 1,], [1, 1]])
	paddedInput = tf.pad(lives, paddings, "SYMMETRIC") # After the padding, its shape is [LIFE_ROWS + 2, LIFE_COLS + 2]

	# To comply with the convention of tf.nn.convolution(), we reshape it to [1, LIFE_ROWS + 2, LIFE_COLS + 2, 1]
	paddedInput = tf.reshape(paddedInput, [1, LIFE_BOARD_ROWS + 2, LIFE_BOARD_COLS + 2, 1])

	# Using convolution to count neighbors of every cell.
	neighbor_population = tf.nn.convolution(paddedInput, NEIGHBOR_SUM_FILTER, padding = 'VALID')
	neighbor_population = tf.reshape(neighbor_population, [LIFE_BOARD_ROWS, LIFE_BOARD_COLS])
	
	# BEGIN DEFINITION of life rules.
	# Any live cell with two neighbors survives.
	nextGen2Neighbors = tf.math.logical_and(tf.math.equal(lives, 1), tf.math.equal(neighbor_population, 2))
	
	# Any cell with three live neighbors keeps live or becomes a live cell.
	nextGen3Neighbors = tf.math.equal(neighbor_population, 3)
	
	# All other live cells die in the next generation. Similarly, all other dead cells stay dead.
	result = tf.math.logical_or(nextGen2Neighbors, nextGen3Neighbors)
	# END DEFINITION of life rules.
	
	return tf.cast(result, tf.float32)

# Run the simulation, and display generations every UPDATE_INTERVAL	iterations.
currentLives = initLives()

fig, ax = plt.subplots()
displayMatrix = ax.matshow(currentLives)

def updateDisplay(i):
	global currentLives
	currentLives = GenerationOfLives(currentLives)

	displayMatrix.set_array(tf.cast(currentLives, tf.int32))

# Set up mp4 movie export
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Somebody'), bitrate=18000)

# Use interval as 1ms since we do not want delay here to test the performance of simulation.
ani = animation.FuncAnimation(fig, updateDisplay, interval = 100)
plt.show()

#ani.save('gol.mp4', writer=writer)