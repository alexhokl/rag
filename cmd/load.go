package cmd

import (
	"context"
	"fmt"

	"github.com/amikos-tech/chroma-go/types"
	"github.com/spf13/cobra"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

type loadOptions struct {
	documentPath         string
	databaseURL          string
	databaseName         string
	filePattern          string
	splitterChunkSize    int
	splitterChunkOverlap int
	embeddingModelName   string
}

var loadOpts loadOptions

var loadCmd = &cobra.Command{
	Use:   "load",
	Short: "Load documents into a vector database",
	RunE:  runLoad,
}

func init() {
	rootCmd.AddCommand(loadCmd)

	flags := loadCmd.Flags()
	flags.StringVarP(&loadOpts.documentPath, "document-path", "f", "", "Path to document file(s)")
	flags.StringVarP(&loadOpts.databaseURL, "database-url", "d", "http://localhost:8000", "URL of vector database")
	flags.StringVarP(&loadOpts.databaseName, "database-name", "n", "", "Name of vector database")
	flags.StringVarP(&loadOpts.filePattern, "file-pattern", "p", "**/*.md", "File glob pattern")
	flags.IntVar(&loadOpts.splitterChunkSize, "splitter-chunk-size", 1500, "Chunk size for splitter")
	flags.IntVar(&loadOpts.splitterChunkOverlap, "splitter-chunk-overlap", 300, "Chunk overlap for splitter")
	flags.StringVarP(&loadOpts.embeddingModelName, "embedding-model", "e", "nomic-embed-text", "Name of embedding model")

	loadCmd.MarkFlagRequired("document-path")
}

func runLoad(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	fmt.Printf(
		"about to load documents from [%s] to create a vector database [%s] at [%s]...\n",
		loadOpts.documentPath,
		loadOpts.databaseName,
		loadOpts.databaseURL,
	)

	documents, err := RetrieveDocuments(loadOpts.documentPath, loadOpts.filePattern)
	if err != nil {
		return fmt.Errorf("failed to retrieve documents: %w", err)
	}
	fmt.Printf("retrieved [%d] documents\n", len(documents))

	splitter := CreateTextSplitter(loadOpts.splitterChunkSize, loadOpts.splitterChunkOverlap)
	_, err = CreateDatabase(
		ctx,
		splitter,
		documents,
		loadOpts.embeddingModelName,
		loadOpts.databaseName,
		loadOpts.databaseURL,
	)
	if err != nil {
		return fmt.Errorf("failed to create database: %w", err)
	}

	fmt.Println("vector database created")
	return nil
}

func RetrieveDocuments(documentPath, filePattern string) ([]schema.Document, error) {
	loader := documentloaders.NewNotionDirectory(documentPath)
	return loader.Load()
}

func CreateTextSplitter(chunkSize, chunkOverlap int) textsplitter.TextSplitter {
	return textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(chunkSize),
		textsplitter.WithChunkOverlap(chunkOverlap),
	)
}

func CreateDatabase(
	ctx context.Context,
	splitter textsplitter.TextSplitter,
	documents []schema.Document,
	embeddingModelName string,
	databaseName string,
	databaseURL string) (vectorstores.VectorStore, error) {
	splittedDocuments, err := splitDocuments(splitter, documents)
	if err != nil {
		return nil, fmt.Errorf("failed to split documents: %w", err)
	}

	embedClient, err := ollama.New(
		ollama.WithModel(embeddingModelName),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load LLM model: %w", err)
	}
	embedder, err := embeddings.NewEmbedder(embedClient)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	store, err := chroma.New(
		chroma.WithChromaURL(databaseURL),
		chroma.WithNameSpace(databaseName),
		chroma.WithEmbedder(embedder),
		chroma.WithDistanceFunction(types.COSINE),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}
	storedDocuments, err := store.AddDocuments(ctx, splittedDocuments)
	if err != nil {
		return nil, fmt.Errorf("failed to store documents: %w", err)
	}
	fmt.Printf("stored [%d] splitted documents\n", len(storedDocuments))
	return store, nil
}

func splitDocuments(textSplitter textsplitter.TextSplitter, documents []schema.Document) ([]schema.Document, error) {
	var splittedDocuments []schema.Document
	for _, doc := range documents {
		splittedTexts, err := textSplitter.SplitText(doc.PageContent)
		if err != nil {
			return nil, err
		}
		for _, text := range splittedTexts {
			splittedDocuments = append(splittedDocuments, schema.Document{
				PageContent: text,
				Metadata:    doc.Metadata,
			})
		}
	}
	return splittedDocuments, nil
}
