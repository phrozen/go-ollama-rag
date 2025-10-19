package main

import (
	"context"
	"fmt"
	"os"

	"github.com/urfave/cli/v3"
)

func main() {
	app := &cli.Command{
		Name:    "rag",
		Usage:   "RAG (Retrieval-Augmented Generation) tool using Ollama embeddings",
		Version: "0.1.0",
		Commands: []*cli.Command{
			{
				Name:  "generate",
				Usage: "Generate embeddings for a markdown file with binary quantization",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "input",
						Aliases:  []string{"i"},
						Usage:    "Input markdown file path",
						Required: true,
					},
					&cli.IntFlag{
						Name:    "chunk-size",
						Aliases: []string{"c"},
						Usage:   "Chunk size for splitting text",
						Value:   1024,
					},
					&cli.IntFlag{
						Name:    "chunk-overlap",
						Aliases: []string{"l"},
						Usage:   "Chunk overlap size (default: 1/4 of chunk-size)",
						Value:   0, // 0 means auto (chunkSize/4)
					},
					&cli.StringFlag{
						Name:    "model",
						Aliases: []string{"m"},
						Usage:   "Ollama embedding model (e.g., mxbai-embed-large, nomic-embed-text)",
						Value:   "qwen3-embedding",
					},
					&cli.IntFlag{
						Name:    "dimensions",
						Aliases: []string{"d"},
						Usage:   "Embedding dimensions",
						Value:   4096,
					},
					&cli.StringFlag{
						Name:    "output",
						Aliases: []string{"o"},
						Usage:   "Output JSON file path (default: <input>.embeddings.json)",
					},
				},
				Action: generateCommand,
			},
			{
				Name:  "search",
				Usage: "Find and print top K embedding results for a query",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "input",
						Aliases:  []string{"i"},
						Usage:    "Input JSON embeddings file",
						Required: true,
					},
					&cli.StringFlag{
						Name:     "query",
						Aliases:  []string{"q"},
						Usage:    "Search query",
						Required: true,
					},
					&cli.IntFlag{
						Name:    "k",
						Usage:   "Number of top results to return",
						Value:   5,
						Aliases: []string{"n"},
					},
				},
				Action: searchCommand,
			},
			{
				Name:  "ask",
				Usage: "Ask an LLM using RAG for context",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "input",
						Aliases:  []string{"i"},
						Usage:    "Input JSON embeddings file",
						Required: true,
					},
					&cli.StringFlag{
						Name:    "model",
						Aliases: []string{"m"},
						Usage:   "Ollama chat model to use",
						Value:   "qwen3",
					},
					&cli.StringFlag{
						Name:     "query",
						Aliases:  []string{"q"},
						Usage:    "Question to ask",
						Required: true,
					},
					&cli.IntFlag{
						Name:    "k",
						Usage:   "Number of context chunks to retrieve",
						Value:   5,
						Aliases: []string{"n"},
					},
				},
				Action: askCommand,
			},
		},
	}

	if err := app.Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
