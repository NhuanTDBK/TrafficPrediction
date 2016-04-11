class SlidingWindow:
    def __init__(self,data,sliding_number):
        self.data = data
        self.sliding_number = sliding_number
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        if self.index<len(self.data)-self.sliding_number+1:
            self.index+=1
            return self.data[(self.index-1):(self.index-1)+self.sliding_number]
        else:
            raise StopIteration

    def __len__(self):
        return self.data.shape[0]-self.sliding_number+1