import numpy as np

np.random.seed(12345)
"""
The argument vectors is N rows of vector with M dimensions.
The return value is a N x N matrix
"""
def correlate(vectors):
    N = vectors.shape[0]
    result = np.zeros((N, N))
    lengths = np.sqrt(np.sum(np.square(vectors), axis = 1))

    for i in range(N):
        result[i, :] = np.dot(vectors, vectors[i, :]) / (lengths[i] * lengths)

    mask = 1 - np.identity(vectors.shape[0])
    result *= mask 

    rating1 = np.sum(np.square(result)) / ( N * (N - 1))
    rating2 = np.max(np.square(result))
    return result, (rating1, rating2)

def cummulativeValue(vectors, power = 2.0):
    signs = np.sign(vectors)
    vectors = np.abs(vectors)
    vectors = np.power(vectors, power)
    vectors *= signs
    sum = np.sum(vectors, axis = 0)
    signs = np.sign(sum)
    sum = np.abs(sum)
    sum = np.power(sum, 1 / power)
    sum *= signs
    return sum


def removeDependent(matrix, remove_rate = 0.005, stop_at = 1e-6):
    print("Energy before {}".format(np.mean(np.square(matrix))))
    lastRating = 1.0
    counter = 1
    firstPass = True
    min_remove_rate = remove_rate
    remove_rate *= 5 
    while True:
        covv, ratings = correlate(matrix)
        if ratings[0] >= lastRating:
            if remove_rate > min_remove_rate:
                remove_rate /= 2
            else:
                break
        if abs(ratings[0] - lastRating) < stop_at:
            break
        lastRating = ratings[0]
        if firstPass:
            print("{} Rating: {}".format(counter, ratings))
            firstPass = False
        counter += 1
        lengths = np.sum(np.square(matrix), axis = 1)
        for i in range(matrix.shape[0]):
            ratio_source_to_this = lengths[i] / lengths
            factor = covv[i] / ratio_source_to_this
            to_be_removed = np.matmul(factor.reshape((matrix.shape[0], 1)), matrix[i].reshape((1, matrix.shape[1]))) * remove_rate
            matrix -= to_be_removed
            matrix[i] += cummulativeValue(to_be_removed, power = 1.113)

    print("{} Rating: {}".format(counter, ratings))
    print("Energy after {}".format(np.mean(np.square(matrix))))
    return matrix



def generateSample(row, column):
    data = np.random.rand(row * column).reshape((row, column)) * 2 - 1
    mask = np.random.randn(row, column)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math

    def analyze(matrix):
        print("Energy {}".format(np.mean(np.square(aa))))
        covv, rating = correlate(aa)

        print("Rating: {}".format(rating))
#        print(covv)

        bins = int(math.sqrt(matrix.shape[0] * matrix.shape[1]))
        plt.hist(covv.reshape(-1), bins = bins)
        plt.show()
    """
    aa = np.random.rand(10 * 5).reshape((10, 5)) - 0.5
    aa = np.vstack([aa, np.arange(5), np.arange(5) + 1])
    analyze(aa)
    aa = removeDependent(aa, remove_rate=0.0001)
    analyze(aa)
    print(aa)
    """
    aa = np.load("layer2.npy")
    analyze(aa)
    aa = removeDependent(aa, remove_rate=0.01)
    analyze(aa)
