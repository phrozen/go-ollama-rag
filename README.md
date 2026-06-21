# go-ollama-rag

Full RAG (Retrieval-Augmented Generation) CLI tool example using Ollama with binary quantization support.

Loosely followed this tutorial and implemented it with binary quantization:

https://k33g.hashnode.dev/rag-from-scratch-with-go-and-ollama

Binary quantization is extremely fast (~60x faster than DotProduct on normalized vectors and ~120x faster than CosineSimilarity) and requires no previous normalization as is unnafected by magnitude. Uses a lot less memory (32x less storage and RAM), but loses accuracy in the process, which can be compensated by using large dimensionality vectors to maintain semantic accuracy (`qwen3-embedding` at 4096 dimensions).

See https://github.com/phrozen/hybrid-agentic-rag for a complete agentic hybrid pipeline example.

## Features

- 🚀 Generate embeddings from markdown files with text splitting
- 🔍 KNN with binary quantization with large dimensionality embeddings (`qwen3-embeddings:8b`: 4096 dimensions)
- 💬 Ask questions with automatic context retrieval (RAG + LLM `qwen3:8b`)
- 📦 Binary quantization always enabled - 32x memory reduction 
- ⚡ Efficient Hamming distance computation for binary embeddings (`uint64`) - ~60x faster than DotProduct ~120x faster than CosineSimilarity
- 🎨 Markdown rendering with [Glamour](https://github.com/charmbracelet/glamour)
- 💫 Spinners and progress indicators with [Clime](https://github.com/alperdrsnn/clime)

## Installation

```bash
git clone https://github.com/phrozen/go-ollama-rag.git
cd go-ollama-rag
go build -o rag
```

Or using [Task](https://taskfile.dev):
```bash
task build
```

## Quick Start

```bash
# 1. Generate embeddings from a markdown file (always includes binary quantization)
./rag generate -i data/chronicles-of-aethelgard.md
# Or: task generate

# 2. Search for similar content 
./rag search -i data/chronicles-of-aethelgard.embeddings.json -q "Which are the monstrous races?" -k 3
# Or: task search QUERY="Which are the monstrous races?"

# 3. Ask questions with RAG
./rag ask -i data/chronicles-of-aethelgard.embeddings.json -q "Explain the biological compatibility of Human species"
# Or: task ask QUERY="Explain the biological compatibility of Human species"
```

## Commands

### Generate
Generate embeddings with automatic binary quantization:
```bash
./rag generate -i <input.md> [options]
# Or: task generate
```
- `--chunk-size` | `-c`: Chunk size for splitting (default: `1024`)
- `--chunk-overlap` | `-l`: Chunk overlap size (default: `1/4 of chunk-size`)
- `--model` | `-m`: Embedding model (default: `qwen3-embedding`)
- `--dimensions` | `-d`: Embedding dimensions (default: `4096`)
- `--output` | `-o`: Output file path (default: `<input>.embeddings.json`)

**Note**: Binary quantization is always enabled for optimal memory usage.

#### Example

```bash
# Pick a different model and set dimensions, don't go too low
./rag generate --model qwen3-embedding:4b --dimensions 2560 -i data/chronicles-of-aethelgard.md
```

### Search
Find and print top K similar chunks with beautiful markdown rendering:

```bash
./rag search -i <embeddings.json> -q "query" [options]
# Or: task search QUERY="your query" K=5
```
- `-k`: Number of results (default: `5`)

### Ask
Ask questions using RAG to an LLM model (streaming disabled in favor of Markdown rendering with Glamour):

```bash
./rag ask -i <embeddings.json> -q "question" [options]
# Or: task ask QUERY="your question" K=5
```

- `--model` | `-m`: Chat model (default: `qwen3`)
- `-k`: Number of context chunks (default: `5`)

#### Example

```bash
# You can pick a larger model to answer the questions
./rag ask --model qwen3:8b -i data/chronicles-of-aethelgard.embeddings.json -q "Explain the biological compatibility of Human species"
```

## Using Taskfile

This project includes a [Taskfile](https://taskfile.dev) for common operations:

```bash
# Show all available tasks
task --list

# Build the project
task build

# Format, test, and build
task all
```

## Requirements

- Go 1.25+
- [Ollama](https://ollama.ai/) running locally
- Embedding models (e.g., `qwen3-embedding`)
- Chat models (e.g., `qwen3`)

## Binary Quantization

Binary quantization is **always enabled** for optimal efficiency:
- 32x compression ratio (4 bytes → 1 bit per dimension)
- Fast Hamming distance computation using bitwise operations on `uint64` type (~60-120x faster)
- Preserves directional information for similarity search at high dimensionality (4096)
- Base64-encoded for portable JSON storage

## Example


### Generate embeddings (default settings)

```bash
❯ ./rag generate -i data/chronicles-of-aethelgard.md                                                                               
✓ Split into 109 chunks
✓ Generated embeddings for 109 chunks (with binary quantization)
✓ Saved document with embeddings to data/chronicles-of-aethelgard.embeddings.json

⏱️  Total time: 53.021s
```

### Search for chunks using KNN [default settings] (optional)

```bash
❯ ./rag search -i data/chronicles-of-aethelgard.embeddings.json -q "Explain the biological compatibility of Human species" -k 1
✓ Generated query embedding
✓ Found 1 results using binary search

🔍 Top 1 results for query: "Explain the biological compatibility of Human species" 

📄 Result 1 (Score: 0.5537)

  ### Other Species                                                           
                                                                              
  Humans generally display more openness to "monstrous" species compared to   
  other races, often establishing diplomatic relations with orcs, goblinoids, 
  and other traditionally isolated groups. This adaptability sometimes creates
  tension with more conservative species.                                     
                                                                              
  ## Biological Compatibility                                                 
                                                                              
  Humans demonstrate unique genetic flexibility, capable of producing viable  
  offspring with various humanoid species. This characteristic has profound   
  implications:                                                               
                                                                              
  • Half-elves represent the most common mixed heritage, often serving as     
  bridges between human and elven communities                                 
  • Half-orcs emerge from either peaceful unions or historical conflicts,     
  facing varying degrees of acceptance                                        
  • Rarer combinations (human-dwarf, human-halfling) occur but typically      
  require magical intervention                                                
                                                                              
  This biological adaptability influences human society, leading to complex   
  inheritance laws and social structures accommodating mixed-heritage         
  individuals.                                                                
                                                                              
  ## Regional Distribution and Variations                                     


────────────────────────────────────────────────────────────────────────────────

⏱️  Total time: 312ms
```

### Ask LLM model with RAG [default settings]

```bash
❯ ./rag ask -i data/chronicles-of-aethelgard.embeddings.json -q "Explain the biological compatibility of Human species"        
✓ Generated query embedding
✓ Retrieved 5 context chunks
✓ Response:

  The biological compatibility of the Human species, as detailed in the       
  context, highlights their unique genetic flexibility, enabling them to      
  produce viable offspring with various humanoid species. This adaptability   
  has significant social and legal implications. Key points include:          
                                                                              
  1. Common Hybrids:                                                          
    • Half-elves are the most prevalent mixed-heritage offspring, often       
    serving as cultural and diplomatic bridges between human and elven        
    communities.                                                              
    • Half-orcs arise from both peaceful unions and historical conflicts,     
    though their acceptance varies across societies.                          
  2. Rare Combinations:                                                       
    • Hybrids such as human-dwarves and human-halflings are less common and   
    typically require magical intervention to occur, reflecting biological or 
    societal barriers.                                                        
  3. Social and Legal Implications:                                           
    • The prevalence of mixed-heritage individuals has led to complex         
    inheritance laws and social structures, necessitating frameworks to       
    address identity, rights, and integration within human societies.         
                                                                              
                                                                              
  This genetic adaptability underscores Humans' role as a versatile species,  
  capable of interbreeding with others, which shapes their societal dynamics  
  and historical interactions with other races.                               



⏱️  Total time: 18.378s
```
