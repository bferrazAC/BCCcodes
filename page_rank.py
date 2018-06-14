import numpy as np

vert = ['A','B','C','D']
edges = {'A' : ['B','C'], 'B' : ['C'], 'C' : ['A'], 'D':['C']}

T = len(vert)
beta = 0.8

adj_g = []
for vline in vert:
	n = len(edges[vline])
	line = np.zeros(T)
	for vcolumn in edges[vline]:
		line[vert.index(vcolumn)] = 1.0/n
	# print adj_g
	# print line
	# print '--'
	adj_g.append(line)
adj_g = np.array(adj_g)
# print adj_g

#calc of random jump

pg_matrix = (((1-beta)/T)*np.ones([T,T]) ) + (beta*adj_g)

# pg i+1 = v * pg i

v = [1.0/T for i in range(T)]
v = np.array(v)
print(v)

threshold = .0001
diff = 1
while(diff > threshold):
	v_1 = np.copy(v)
	v = np.dot(v,pg_matrix)
	diff = np.max(np.square(v-v_1))
print v