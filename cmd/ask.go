package cmd

import (
	"context"
	"fmt"
	"strings"

	"github.com/alexhokl/helper/jsonhelper"
	"github.com/alexhokl/helper/ollamahelper"
	"github.com/spf13/cobra"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

const NUM_DOCUMENTS_TO_BE_USED = 5
const DEFAULT_SEARCH_SCORE_THRESHOLD = 0.5

type askOptions struct {
	databaseURL        string
	databaseName       string
	modelName          string
	question           string
	embeddingModelName string
}

type GraderResponse struct {
	Score string `json:"score"`
}

var askOpts askOptions

var askCmd = &cobra.Command{
	Use:   "ask",
	Short: "Ask a question on the documents stored in the specified vector database",
	RunE:  runAsk,
}

func init() {
	rootCmd.AddCommand(askCmd)

	flags := askCmd.Flags()
	flags.StringVarP(&askOpts.databaseURL, "database-url", "d", "http://localhost:8000", "URL of vector database")
	flags.StringVarP(&askOpts.databaseName, "database-name", "n", "", "Name of vector database")
	flags.StringVarP(&askOpts.embeddingModelName, "embedding-model", "e", "nomic-embed-text", "Name of embedding model")
	flags.StringVarP(&askOpts.modelName, "model", "m", "", "Name of the model")
	flags.StringVarP(&askOpts.question, "question", "q", "", "Question to ask")

	askCmd.MarkFlagRequired("model")
	askCmd.MarkFlagRequired("question")
	askCmd.MarkFlagRequired("database-name")
}

func runAsk(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}

	embedder, err := ollamahelper.GetEmbedder(askOpts.embeddingModelName)
	if err != nil {
		return fmt.Errorf("unable to load embedder: %w", err)
	}
	store, err := chroma.New(
		chroma.WithChromaURL(askOpts.databaseURL),
		chroma.WithNameSpace(askOpts.databaseName),
		chroma.WithEmbedder(embedder),
	)
	if err != nil {
		return fmt.Errorf("unable to load database: %w", err)
	}
	retriever := vectorstores.ToRetriever(
		store,
		NUM_DOCUMENTS_TO_BE_USED,
		vectorstores.WithScoreThreshold(DEFAULT_SEARCH_SCORE_THRESHOLD),
	)

	documents, err := retriever.GetRelevantDocuments(ctx, askOpts.question)
	if err != nil {
		return fmt.Errorf("failed to retrieve documents: %w", err)
	}

	if len(documents) == 0 {
		fmt.Println("No reference documents found")
		return nil
	}

	fmt.Printf("Found %d document sections from database\n", len(documents))
	// for i, doc := range documents {
	// 	fmt.Printf("Document %d. [score:%f] [source:%s]\n", i+1, doc.Score, doc.Metadata["source"])
	// }

	llm, err := ollama.New(
		ollama.WithModel(askOpts.modelName),
	)
	if err != nil {
		return fmt.Errorf("unable to connect to a model using Ollama: %w", err)
	}

	relevantDocuments := []schema.Document{}
	for _, doc := range documents {
		graderSystemPrompt := `
			You are a grader assessing relevance of a retrieved document to a user question.
			If the document contains keywords related to the user question,
			grade it as relevant. It does not need to be a stringent test.
			The goal is to filter out erroneous retrievals.
			Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
			Provide the binary score as a JSON with a single key 'score' and no premable or explanation.`
		graderUserPrompt := fmt.Sprintf("Here is the retrieved document: %s \n\nHere is the user question: %s", doc.PageContent, askOpts.question)

		gradingResponse, err := ollamahelper.GenerateContent(ctx, llm, graderSystemPrompt, graderUserPrompt)
		if err != nil {
			return fmt.Errorf("unable to grade reference document: %w", err)
		}
		var graderResponse GraderResponse
		errParse := jsonhelper.ParseJSONString(gradingResponse, &graderResponse)
		if errParse != nil {
			return fmt.Errorf("unable to parse grading response: %w", errParse)
		}
		if graderResponse.Score == "yes" {
			relevantDocuments = append(relevantDocuments, doc)
		}
	}

	if len(relevantDocuments) == 0 {
		fmt.Println("No relevant documents found")
		return nil
	}

	fmt.Printf("About to answer your question using %d relevant document sections...\n\n\n", len(relevantDocuments))

	systemPrompt := `
		You are an assistant for question-answering tasks.
		Use the following pieces of retrieved documentation to answer the question.
		Please write in full sentences with correct spelling and punctuation. If it makes sense, use lists.
		If the documentation does not contain the answer, just respond that you are unable to find an answer.
	    Explain the reasoning as well.`
	relevantTexts := make([]string, 0, len(relevantDocuments))
	for _, doc := range relevantDocuments {
		relevantTexts = append(relevantTexts, doc.PageContent)
	}
	userPrompt := fmt.Sprintf("Documentation: %s \n\nQuestion: %s \n\nAnswer: ", strings.Join(relevantTexts, " ; "), askOpts.question)
	err = ollamahelper.GenerateStreamingContent(ctx, llm, systemPrompt, userPrompt, func(_ context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	})
	if err != nil {
		return fmt.Errorf("unable to generate answer: %w", err)
	}

	return nil
}
