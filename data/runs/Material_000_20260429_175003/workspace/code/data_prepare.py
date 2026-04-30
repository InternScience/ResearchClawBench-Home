class RealisticCrystalDataset:
    def __init__(self,*args,**kwargs):
        self.__dict__.update(kwargs)
    def __len__(self):
        return len(getattr(self,'samples',[]))
    def __getitem__(self, idx):
        return self.samples[idx]
