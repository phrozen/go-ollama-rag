package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"time"
)

type Document struct {
	Source     string    `json:"source"`
	Model      string    `json:"model"`
	Dimensions int       `json:"dimensions"`
	SHA256     string    `json:"sha256"`
	Timestamp  time.Time `json:"timestamp"`
	Chunks     []Chunk   `json:"chunks"`
}

type Chunk struct {
	Text      string   `json:"text"`
	Embedding []uint64 `json:"embedding"`
}

// chunkJSON is used for custom JSON marshaling
type chunkJSON struct {
	Text      string `json:"text"`
	Embedding string `json:"embedding,omitempty"` // Base64 encoded
}

// MarshalJSON converts Binary []uint64 to Base64-encoded bytes
func (e Chunk) MarshalJSON() ([]byte, error) {
	ej := chunkJSON{
		Text:      e.Text,
		Embedding: "",
	}

	// Convert []uint64 to []byte and Base64 encode
	if len(e.Embedding) > 0 {
		bytes := uint64SliceToBytes(e.Embedding)
		ej.Embedding = base64.StdEncoding.EncodeToString(bytes)
	}

	return json.Marshal(ej)
}

// UnmarshalJSON converts Base64-encoded bytes back to []uint64
func (e *Chunk) UnmarshalJSON(data []byte) error {
	var ej chunkJSON
	if err := json.Unmarshal(data, &ej); err != nil {
		return err
	}

	e.Text = ej.Text

	// Decode Base64 and convert back to []uint64
	if ej.Embedding != "" {
		bytes, err := base64.StdEncoding.DecodeString(ej.Embedding)
		if err != nil {
			return err
		}
		e.Embedding = bytesToUint64Slice(bytes)
	}

	return nil
}

// uint64SliceToBytes converts []uint64 to []byte using little-endian encoding
func uint64SliceToBytes(values []uint64) []byte {
	bytes := make([]byte, len(values)*8)
	for i, val := range values {
		binary.LittleEndian.PutUint64(bytes[i*8:(i+1)*8], val)
	}
	return bytes
}

// bytesToUint64Slice converts []byte to []uint64 using little-endian encoding
func bytesToUint64Slice(bytes []byte) []uint64 {
	numUint64s := len(bytes) / 8
	values := make([]uint64, numUint64s)
	for i := 0; i < numUint64s; i++ {
		values[i] = binary.LittleEndian.Uint64(bytes[i*8 : (i+1)*8])
	}
	return values
}

// BinaryQuantize converts a float32 embedding to a binary representation using simple thresholding
func BinaryQuantize(fp32 []float32) []uint64 {
	// Calculate how many uint64s we need (64 bits per uint64)
	numUint64s := (len(fp32) + 63) / 64
	binary := make([]uint64, numUint64s)

	// Pack bits: 1 if value >= 0, 0 otherwise
	for i, val := range fp32 {
		if val >= 0 {
			// Calculate which uint64 and which bit position
			uint64Index := i / 64
			bitPosition := i % 64

			// Set the bit to 1 using little-endian order
			binary[uint64Index] |= (1 << bitPosition)
		}
		// Negative values leave the bit as 0 (default)
	}

	return binary
}
