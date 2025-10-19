package main

import (
	"fmt"

	"github.com/alperdrsnn/clime"
	"github.com/tmc/langchaingo/textsplitter"
)

func Spinner(format string, args ...any) *clime.Spinner {
	spinner := clime.NewSpinner().
		WithMessage(fmt.Sprintf(format, args...)).
		WithColor(clime.BrightMagentaColor).
		Start()
	return spinner
}

func SplitMarkdownWithSize(content string, chunkSize int, chunkOverlap int) ([]string, error) {

	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	)
	/*
		// Markdown splitter implementation produces less quality chunks overall
		// due to excessive splitting on headings etc.
		// Uncomment to use more advanced markdown splitting.
		splitter := textsplitter.NewMarkdownTextSplitter(
			textsplitter.WithChunkSize(chunkSize),
			textsplitter.WithChunkOverlap(128),
			textsplitter.WithHeadingHierarchy(true),
			textsplitter.WithKeepSeparator(true),
			textsplitter.WithJoinTableRows(true),
			textsplitter.WithReferenceLinks(true),
			textsplitter.WithCodeBlocks(true),
		)
	*/
	return splitter.SplitText(content)
}
