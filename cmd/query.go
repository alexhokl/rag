package cmd

import (
	"context"
	"fmt"

	chroma "github.com/amikos-tech/chroma-go"
	"github.com/amikos-tech/chroma-go/pkg/embeddings/ollama"
	"github.com/spf13/cobra"
)

const DEFAULT_OLLAMA_URL = "http://localhost:11434"

type queryOptions struct {
	databaseURL        string
	databaseName       string
	embeddingModelName string
	question           string
	resultCount        int32
}

var queryOpts queryOptions

var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Query against documents stored in the specified vector database",
	RunE:  runQuery,
}

func init() {
	rootCmd.AddCommand(queryCmd)

	flags := queryCmd.Flags()
	flags.StringVarP(&queryOpts.databaseURL, "database-url", "d", "http://localhost:8000", "URL of vector database")
	flags.StringVarP(&queryOpts.databaseName, "database-name", "n", "", "Name of vector database")
	flags.StringVarP(&queryOpts.embeddingModelName, "model", "e", "nomic-embed-text", "Name of embedding model")
	flags.StringVarP(&queryOpts.question, "question", "q", "", "Query")
	flags.Int32VarP(&queryOpts.resultCount, "result-count", "r", 5, "Number of results to return")

	queryCmd.MarkFlagRequired("question")
	queryCmd.MarkFlagRequired("database-name")
}

func runQuery(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	embeddingFunction, err := ollama.NewOllamaEmbeddingFunction(
		ollama.WithBaseURL(DEFAULT_OLLAMA_URL),
		ollama.WithModel(queryOpts.embeddingModelName),
	)
	if err != nil {
		return fmt.Errorf("failed to create Ollama embedding function: %w", err)
	}

	datastore, err := chroma.NewClient(queryOpts.databaseURL)
	if err != nil {
		return fmt.Errorf("failed to connect to vector database: %w", err)
	}
	collection, err := datastore.GetCollection(ctx, queryOpts.databaseName, embeddingFunction)
	if err != nil {
		return fmt.Errorf("failed to get collection [%s]: %w", queryOpts.databaseName, err)
	}

	results, err := collection.Query(
		ctx,
		[]string{queryOpts.question},
		queryOpts.resultCount,
		nil,
		nil,
		nil,
	)
	if err != nil {
		return fmt.Errorf("failed to query collection [%s]: %w", queryOpts.databaseName, err)
	}

	for index, ids := range results.Ids {
		for innerIndex, id := range ids {
			fmt.Printf("Result %d\n", (innerIndex + 1))
			fmt.Printf("Document ID: %s\n", id)
			fmt.Printf("Score: %f\n", 1 - results.Distances[index][innerIndex])
			fmt.Printf("Source: %s\n", results.Metadatas[index][innerIndex]["source"])
			fmt.Printf("Document: %s\n\n", results.Documents[index][innerIndex])
		}
	}

	return nil
}
