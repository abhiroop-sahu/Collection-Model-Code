d = 0.3 # This is the distance decay factor (from the center), represented as lowercase delta in paper
# This class is used for the foveation algorithm
class Sample:
    def __init__(self, x, prefix_sum, x1, y1):
        self.x = x;
        self.x1 = x1
        self.y1 = y1
        self.diff = np.std(x)
        self.dist = math.sqrt(((13 - x1)**2) + ((13 - y1)**2))
        self.key = prefix_sum - (self.dist * d) + (28 * self.diff)

    def __lt__(self, other):
        return self.key > other.key
