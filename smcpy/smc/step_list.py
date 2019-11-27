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
