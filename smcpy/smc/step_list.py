class StepList():

    def __init__(self):
        self._list = []

    def __getitem__(self, i):
        return self._list[i]

    def add_step(self, step):
        self._list.append(step)
        return None

    def pop_step(self, i=None):
        popped_step = self._list.pop(i)
        return popped_step

    def trim(self, i=None):
        self._list = self._list[:i]
        return None

    def compute_bayes_evidence(self):
        '''
        Computes the Bayes evidence, or normalizing constant according to the eq
            Z_T = Z_0 * prod( sum( unormalized weights for each step ) )
        where Z_0 = int( p(theta) dtheta) = 1.
        '''
        sum_unnormalized = [sum(step.get_weights()) for step in self._list]
        return reduce(lambda x, y: x * y, sum_unnormalized)
