package main

import (
	"container/heap"
	"fmt"
	"math"
	"math/bits"
	"sort"
)

// Result represents a search result with chunk index and similarity score
type Result struct {
	ChunkIndex int
	Score      float64
}

// MaxHeap for KNN search (we want to keep the k smallest distances/highest similarities)
type MaxHeap []Result

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].Score > h[j].Score } // Max heap
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

func (h *MaxHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// DotProduct computes the dot product of two FP32 embeddings (equals cosine similarity for normalized vectors)
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("embeddings must have the same length")
	}

	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

// HammingDistance computes the Hamming distance between two binary embeddings
func HammingDistance(a, b []uint64) int {
	if len(a) != len(b) {
		panic("binary embeddings must have the same length")
	}

	distance := 0
	for i := range a {
		// XOR gives 1 where bits differ, count those bits
		distance += bits.OnesCount64(a[i] ^ b[i])
	}
	return distance
}

// HammingToCosineSimilarity converts Hamming distance to approximate cosine similarity
// Formula: cos(θ) ≈ cos(π * h / d) where h is Hamming distance and d is dimensionality
func HammingToCosineSimilarity(hammingDistance int, dimensionality int) float64 {
	if dimensionality <= 0 {
		panic(fmt.Sprintf("invalid dimensionality: %d", dimensionality))
	}
	// Normalize hamming distance to [0, 1]
	normalized := float64(hammingDistance) / float64(dimensionality)
	// Convert to cosine similarity approximation
	return math.Cos(math.Pi * normalized)
}

// KNN performs K-nearest neighbors search using binary embeddings (Hamming distance)
func KNN(query []uint64, chunks []Chunk, k int, dimensionality int) []Result {
	if k <= 0 || len(chunks) == 0 {
		return []Result{}
	}

	// Use max heap to keep track of top k results (min Hamming distance = max similarity)
	h := &MaxHeap{}
	heap.Init(h)

	for i, chunk := range chunks {
		distance := HammingDistance(query, chunk.Embedding)
		// Convert to similarity (lower distance = higher similarity)
		similarity := HammingToCosineSimilarity(distance, dimensionality)

		if h.Len() < k {
			heap.Push(h, Result{ChunkIndex: i, Score: similarity})
		} else if similarity > (*h)[0].Score {
			// Replace worst result if current is better
			heap.Pop(h)
			heap.Push(h, Result{ChunkIndex: i, Score: similarity})
		}
	}

	// Convert heap to slice and sort by score (descending)
	results := []Result(*h)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}
