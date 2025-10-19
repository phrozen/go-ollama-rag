package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/ollama/ollama/api"
	"github.com/urfave/cli/v3"
)

func generateCommand(ctx context.Context, cmd *cli.Command) error {
	startTime := time.Now()

	input := cmd.String("input")
	chunkSize := cmd.Int("chunk-size")
	chunkOverlap := cmd.Int("chunk-overlap")
	if chunkOverlap == 0 {
		chunkOverlap = chunkSize / 4
	}
	model := cmd.String("model")
	dimensions := cmd.Int("dimensions")
	output := cmd.String("output")

	if output == "" {
		output = strings.ReplaceAll(input, ".md", ".embeddings.json")
	}

	// Read input file
	data, err := os.ReadFile(input)
	if err != nil {
		return fmt.Errorf("failed to read input file: %w", err)
	}

	spinner := Spinner("Splitting markdown...")
	chunks, err := SplitMarkdownWithSize(string(data), chunkSize, chunkOverlap)
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to split markdown: %v", err))
		return err
	}
	if len(chunks) == 0 {
		spinner.Error("No chunks generated from input file")
		return fmt.Errorf("no chunks generated from input file")
	}
	spinner.Success(fmt.Sprintf("Split into %d chunks", len(chunks)))

	// Save chunks for inspection (optional) uncomment for debugging
	// os.WriteFile(strings.ReplaceAll(input, ".md", ".chunks.md"), []byte(strings.Join(chunks, "\n\n---\n\n")), os.ModePerm)

	spinner = Spinner("Generating embeddings with binary quantization (%s)...", model)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to create API client: %v", err))
		return err
	}

	doc := Document{
		Source:     filepath.Base(input),
		Model:      model,
		Dimensions: dimensions,
		SHA256:     fmt.Sprintf("%x", sha256.Sum256(data)),
		Timestamp:  time.Now().UTC(),
		Chunks:     make([]Chunk, 0, len(chunks)),
	}

	for i, chunk := range chunks {
		spinner.WithMessage(fmt.Sprintf("Generating embeddings %d/%d (%s)...", i+1, len(chunks), model))
		embeddings, err := client.Embed(ctx, &api.EmbedRequest{
			Model:      model,
			Input:      chunk,
			Dimensions: dimensions,
		})
		if err != nil {
			spinner.Error(fmt.Sprintf("Failed to create embeddings for chunk %d: %v", i, err))
			return err
		}
		// Always quantize (it's ~60x faster and provides 32x memory reduction)
		doc.Chunks = append(doc.Chunks, Chunk{
			Text:      chunk,
			Embedding: BinaryQuantize(embeddings.Embeddings[0]),
		})
	}
	spinner.Success(fmt.Sprintf("Generated embeddings for %d chunks (with binary quantization)", len(chunks)))

	spinner = Spinner("Saving document with embeddings...")
	data, err = json.MarshalIndent(doc, "", "    ")
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to marshal document: %v", err))
		return err
	}

	err = os.WriteFile(output, data, 0644)
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to write embeddings file: %v", err))
		return err
	}
	spinner.Success(fmt.Sprintf("Saved document with embeddings to %s", output))

	elapsed := time.Since(startTime)
	fmt.Printf("\n⏱️  Total time: %s\n", elapsed.Round(time.Millisecond))

	return nil
}

func searchCommand(ctx context.Context, cmd *cli.Command) error {
	startTime := time.Now()

	input := cmd.String("input")
	query := cmd.String("query")
	k := cmd.Int("k")

	// Load embeddings document
	data, err := os.ReadFile(input)
	if err != nil {
		return fmt.Errorf("failed to read embeddings file: %w", err)
	}

	var doc Document
	if err := json.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("failed to unmarshal embeddings: %w", err)
	}

	if doc.Dimensions != len(doc.Chunks[0].Embedding)*64 {
		return fmt.Errorf("dimension mismatch: expected %d bits, got %d",
			len(doc.Chunks[0].Embedding)*64, doc.Dimensions)
	}

	spinner := Spinner("Generating query embedding...")
	client, err := api.ClientFromEnvironment()
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to create API client: %v", err))
		return err
	}

	embeddings, err := client.Embed(ctx, &api.EmbedRequest{
		Model:      doc.Model,
		Input:      query,
		Dimensions: doc.Dimensions,
	})
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to create query embedding: %v", err))
		return err
	}
	queryEmbedding := embeddings.Embeddings[0]
	queryBinary := BinaryQuantize(queryEmbedding)
	spinner.Success("Generated query embedding")

	// Perform KNN search
	var results []Result

	spinner = Spinner("Searching with binary embeddings (Hamming distance)...")

	results = KNN(queryBinary, doc.Chunks, k, doc.Dimensions)
	spinner.Success(fmt.Sprintf("Found %d results using binary search", len(results)))

	// Print results with glamour
	fmt.Printf("\n🔍 Top %d results for query: %q \n\n", len(results), query)

	for i, result := range results {
		fmt.Printf("📄 Result %d (Score: %.4f)\n", i+1, result.Score)
		out, err := glamour.Render(doc.Chunks[result.ChunkIndex].Text, "dark")
		if err != nil {
			return fmt.Errorf("failed to render result: %w", err)
		}
		fmt.Println(out)
		fmt.Println(strings.Repeat("─", 80))
	}

	elapsed := time.Since(startTime)
	fmt.Printf("\n⏱️  Total time: %s\n", elapsed.Round(time.Millisecond))

	return nil
}

func askCommand(ctx context.Context, cmd *cli.Command) error {
	startTime := time.Now()

	input := cmd.String("input")
	model := cmd.String("model")
	query := cmd.String("query")
	k := cmd.Int("k")

	// Load embeddings document
	data, err := os.ReadFile(input)
	if err != nil {
		return fmt.Errorf("failed to read embeddings file: %w", err)
	}

	var doc Document
	if err := json.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("failed to unmarshal embeddings: %w", err)
	}

	spinner := Spinner("Generating query embedding...")
	client, err := api.ClientFromEnvironment()
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to create API client: %v", err))
		return err
	}

	embeddings, err := client.Embed(ctx, &api.EmbedRequest{
		Model:      doc.Model,
		Input:      query,
		Dimensions: doc.Dimensions,
	})
	if err != nil {
		spinner.Error(fmt.Sprintf("Failed to create query embedding: %v", err))
		return err
	}
	queryEmbedding := embeddings.Embeddings[0]
	queryBinary := BinaryQuantize(queryEmbedding)
	spinner.Success("Generated query embedding")

	// Perform KNN search
	spinner = Spinner("Retrieving relevant context...")
	results := KNN(queryBinary, doc.Chunks, k, doc.Dimensions)
	spinner.Success(fmt.Sprintf("Retrieved %d context chunks", len(results)))

	// Build context from retrieved chunks
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Use the following context to answer the question:\n\n")
	for i, result := range results {
		chunk := doc.Chunks[result.ChunkIndex]
		contextBuilder.WriteString(fmt.Sprintf("Context %d:\n%s\n\n", i+1, strings.TrimSpace(chunk.Text)))
	}
	contextBuilder.WriteString(fmt.Sprintf("Question: %s", query))

	spinner = Spinner("Asking LLM (%s)...", model)

	// Stream the response
	streaming := false
	req := &api.GenerateRequest{
		Model:  model,
		Prompt: contextBuilder.String(),
		Stream: &streaming,
	}

	err = client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		out, err := glamour.Render(resp.Response, "dark")
		if err != nil {
			return fmt.Errorf("failed to render response: %w", err)
		}
		spinner.Success("Response:")
		fmt.Print(out)
		return nil
	})

	if err != nil {
		spinner.Error(fmt.Sprintf("failed to generate response: %v", err))
		return fmt.Errorf("failed to generate response: %w", err)
	}

	fmt.Println()

	elapsed := time.Since(startTime)
	fmt.Printf("\n⏱️  Total time: %s\n", elapsed.Round(time.Millisecond))

	return nil
}
