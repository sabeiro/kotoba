import math
#import numpy as np
#from scipy import linalg as la
#import matplotlib.pyplot as plt

def Dist(x1,y1):
    ret = (x1[0]-y1[0])*(x1[0]-y1[0])
    ret += (x1[1]-y1[1])*(x1[1]-y1[1])
    ret += (x1[2]-y1[2])*(x1[2]-y1[2])
    return math.sqrt(ret)

def CrNode(file,name):
    string = "CREATE (song {entitytype:'song', name '" + name + "'}) return song";
    file.write(string)


f = open('EntropyTable.dat') 
NFile = 0
entropy = []
FName = []
#entropy = [[int(val) for val in line.split()] for line in lines_list[1:]]
for line in f: # read rest of lines
    line1 = []
    line1.append([(x) for x in line.split()])
    FName.append(line1[0][0])
    entropy.append([float(x) for x in line1[0][1:4]])
    NFile += 1
f.close()
CorrMat= [ [ 0 for i in range(NFile) ] for j in range(NFile) ]
for i in range(NFile):
    for j in range(i+1,NFile):
        x = entropy[i][0:3]
        y = entropy[j][0:3]
        dist = Dist(x,y)
        CorrMat[i][j] = dist
        CorrMat[j][i] = dist


FOut = open('NeoQueries.dat',"w") 
ctype = "song"
FOut.write("CREATE (Entropy:Measure {label:'measure for creativity'})\n")
node = "CREATE "
for i in range(NFile):
    #node += " n={id:'%d" % i + "  ', name:'" + FName[i] + "', type:'" + ctype + "'};\n"
    node += "(" + FName[i] + ":Song { id:'%d', " %i + "name:'" + FName[i] + "', entropy:'%.3f'" %  entropy[i][2] + "}), "
FOut.write(node + "\b\b\n")
RelType = "DIST"
ConnDist = .5
relation = "CREATE "
for i in range(NFile):
    for j in range(i+1,NFile):
        if(CorrMat[i][j] < ConnDist):
            dist = "%.3f" % CorrMat[i][j]
            #relation += "start n1=node:node_auto_index(id='%d" % i + "'), n2=node:node_auto_index(id='%d" % j + "')  create n1-[:" + RelType + "]->n2;"
            #relation += "(%d" % i  + ") - [:" + RelType + "{d: " + dist + "}] -> (%d" % j + "), "
            relation += "(" + FName[i] + ") - [:" + RelType + "{d: " + dist + "}] -> (" + FName[j] + "), "
FOut.write(relation + "\b\b\n")
FOut.write("RETURN Entropy;")
#mn = np.mean(entropy,axis=0)
#entropy -= mn
#C = np.cov(entropy)
#evals, evecs = la.eig(C)
#print evals, evacs


#fig, ax = plt.subplots()
#ax.plot(CorrMat, 'o')
#plt.show()
