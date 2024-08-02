import math

class EHisto:
    NArr = 0
    Tot = 0.
    histo = [0.]

    def __init__(self,NArr):
        self.NArr = NArr
        self.histo = [0.]*NArr

    def __str__(self):
        string = ' '.join('%.3f' % n for n in self.histo)
        return string
    #return str(self.histo).strip('[]')

    def add(self,pos,val):
        self.histo[pos] += val

    def val(self,pos):
        return self.histo[pos]

    def set(self,pos,val):
        self.histo[pos] = val

    def clear(self):
        for i in range(self.NArr):
            self.histo[i] = 0.

    def Normalize(self):
        Count = 0.
        for i in range(self.NArr):
            Count += self.histo[i]
        if (Count < 1.):
            Count = 1.
        else:
            Count = 1./Count
        for i in range(self.NArr):
            self.histo[i] *= Count

    def EntCalc(self,Count):
        self.Tot = Count
        if (Count < 1.):  Count = 1.
        else: Count = 1./Count
        Entropy = 0.
        for i in range(0,self.NArr):
            freq = self.histo[i]*Count
            if (freq > 0.):
                Entropy -= freq*math.log(freq,2)
        return Entropy

    def setTotal(self,Count):
        self.Tot = Count

    def DistMat(self,histo2):
        NArr = self.NArr
        if(histo2.NArr != NArr):
            return 0.
        #CorrMat= [ [ 0 for i in range(NArr) ] for j in range(NArr) ]
        Count = 0.
        for i in range(NArr):
            for j in range(i+1,NArr):
                dist = math.pow(self.histo[i]-histo2.val(j),2)
                Count += dist
                #CorrMat[i][j] = dist
        return Count

    def Rotate(self):
        tmp = self.histo[0]
        for i in range(self.NArr-1):
            self.histo[i] = self.histo[i+1]
        self.histo[self.NArr-1] = tmp

    def Min(self):
        imin = 0
        Min = self.histo[0]
        for i in range(self.NArr):
            if(self.histo[i] < Min):
                Min = self.histo[i]
                imin = i
        return imin
