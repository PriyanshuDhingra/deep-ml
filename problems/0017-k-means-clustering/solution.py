def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	final_centroids = initial_centroids.copy()
	clusters = [[] for _ in range(k)]
	for _ in range(max_iterations):
		for point in points:
			dist = float('inf')
			closest = 0
			for i in range(len(initial_centroids)):
				temp = sum((initial_centroids[i][k] - point[k])**2 for k in range(len(point)))
				if temp < dist:
					dist = temp
					closest = i
			clusters[closest].append(point)		
		
		for i in range(len(clusters)):
			cluster = clusters[i]
			if cluster:
				final_centroids[i] = tuple(round(sum(point[j] for point in cluster), 4) / len(cluster) for j in range(len(cluster[0])))
				initial_centroids = final_centroids.copy()
			
	return final_centroids