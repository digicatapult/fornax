import abc
import numpy as np

class Frame:

    def __init__(self, records, names):
        self.records = np.rec.fromrecords(records, names=names)

    def __getitem__(self, indx) -> 'Frame':
        return self.from_record_array(self.records[indx])

    def __len__(self):
        return len(self.records)

    @classmethod
    def from_record_array(cls, rec_array):
        new = cls.__new__(cls)
        new.records = rec_array
        return new


class NeighbourHoodMatchingCosts(Frame):

    names = 'u v uu vv cost'.split()

    def __init__(self, records):
        super().__init__(records, self.names)

    @property
    def u(self):
        """Column of Query node ids u
        
        Returns:
            [int] -- [return the column of query node ids u]
        """
        return self.records.u
