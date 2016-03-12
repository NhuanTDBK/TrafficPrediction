class SplitTrainTest():
    def __iter__(self):
        self.output = [(self.train_indices, self.test_indices)]
        yield self.output[0]
    def __init__(self,n_sample=None,train_size=0.7):
        indices = range(n_sample)
        split_index = int(n_sample*train_size)
        self.train_indices = indices[:split_index]
        self.test_indices = indices[split_index:]
        self.loop = 0
        self.n= n_sample

    def next(self):
        if(self.loop<len(self.output)):
            # print self.loop,len(self.output)
            self.loop+=1
            return self.output[self.loop-1]
        else:
            raise StopIteration
    def __len__(self):
        return int(1)