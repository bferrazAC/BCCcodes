import numpy as np

vert = ['A','B','C','D']
edges = {'A' : ['B','C'], 'B' : ['C'], 'C' : ['A'], 'D':['C']}

# list of nodes that point to a specific one
inv_edges = {}
for l in vert:
	pointed_list = []
	for p in edges:
		if l in edges[p]:
			pointed_list.append(p)
	inv_edges[l] = pointed_list
# print inv_edges

k = 625

z = [1.0 for i in range(len(vert))]
z = np.array(z)

x_0 = np.copy(z)
y_0 = np.copy(z)

for i in range(k):
	x = np.copy(x_0)
	y = np.copy(y_0)
	for i in range(len(vert)):
		node = vert[i]
		point_to = edges[node]

		# if(len(point_to) > 0):
			# aux = [y[vert.index(n)] for n in point_to]
		# else:
		# 	aux = 0

		# somatorio dos hubs dos nos apontados
		x[i] = np.sum([y[vert.index(n)] for n in point_to])
	for i in range(len(vert)):
		node = vert[i]
		pointed_by = inv_edges[node]

		# if(len(pointed_by) > 0):
			# aux = [x[vert.index(n)] for n in pointed_by]
		# else:
			# aux = 0

		# somatorio das autoridades dos nos que apontam
		y[i] = np.sum([x[vert.index(n)] for n in pointed_by])
	
	# normalizacao
	x_0 = x / np.linalg.norm(x)
	y_0 = y / np.linalg.norm(y)

print 'X',x_0,'Y', y_0