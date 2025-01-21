import numpy as np

def cosine_similarity(A, B):
    """
    Compute cosine similarity between two vectors A and B.
    """
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def cosine_distance(A, B):
    """
    Compute cosine distance between two vectors A and B.
    Cosine distance = 1 - Cosine similarity
    """
    return 1 - cosine_similarity(A, B)

def cosine_distance_matrix(X, Y=None):
    """
    Compute cosine similarity between two sets of vectors.
    
    Parameters:
    - X: np.array of shape (n_samples_X, n_features)
    - Y: np.array of shape (n_samples_Y, n_features), or None
         If None, compute similarity between rows in X.
    
    Returns:
    - similarities: np.array of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X
    
    # Normalize each row vector to have unit length
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Compute the cosine distance matrix
    return  1- np.dot(X_normalized, Y_normalized.T)

def initialize_medoids(X, k):
    """
    Randomly select k medoids from the dataset X.
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return indices

def assign_points_to_medoids(distance_matrix, medoids):
    """
    Assign each point to the nearest medoid based on the distance matrix.
    
    Parameters:
    - distance_matrix: precomputed distance matrix (n_samples, n_samples)
    - medoids: indices of the current medoids
    
    Returns:
    - clusters: list of cluster indices for each data point
    """
    # Get distances from each point to each medoid
    medoid_distances = distance_matrix[:, medoids]
    # Assign each point to the closest medoid
    clusters = np.argmin(medoid_distances, axis=1)
    return clusters

def update_medoids(X, clusters, k, distance_matrix):
    """
    Update medoids by choosing the point in each cluster that minimizes 
    the sum of distances to all other points in the cluster.
    
    Parameters:
    - X: data points (n_samples, n_features)
    - clusters: cluster assignments for each point
    - k: number of clusters
    - distance_matrix: precomputed distance matrix (n_samples, n_samples)
    
    Returns:
    - new_medoids: indices of the updated medoids
    """
    new_medoids = np.zeros(k, dtype=int)
    
    for i in range(k):
        cluster_points = np.where(clusters == i)[0]
        if len(cluster_points) == 0:
            continue
        # Calculate the sum of distances for each point in the cluster to all other points in the cluster
        cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
        total_distances = np.sum(cluster_distances, axis=1)
        # Choose the point with the minimum total distance as the new medoid
        new_medoids[i] = cluster_points[np.argmin(total_distances)]
    
    return new_medoids

def k_medoids(X, k, max_iter=300):
    """
    K-Medoids clustering algorithm using cosine similarity.
    
    Parameters:
    - X: data points (n_samples, n_features)
    - k: number of clusters
    - max_iter: maximum number of iterations
    
    Returns:
    - medoids: indices of the final medoids
    - clusters: cluster assignments for each data point
    """
    # Step 1: Initialize medoids
    medoids_ind = initialize_medoids(X, k)
    
    # Step 2: Precompute cosine distance matrix
    distance_matrix = cosine_distance_matrix(X)
    print("calc distances")
    
    for i in range(max_iter):
        print(f"iter {i}")
        # Step 3: Assign points to the nearest medoids
        clusters = assign_points_to_medoids(distance_matrix, medoids_ind)
        
        # Step 4: Update medoids
        new_medoids_ind = update_medoids(X, clusters, k, distance_matrix)
        
        # If the medoids don't change, convergence is reached
        if np.array_equal(medoids_ind, new_medoids_ind):
            break
        
        medoids_ind = new_medoids_ind
    
    return medoids_ind, clusters