package documentloaders

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/langchaingo/schema"
)

// MarkdownDirectoryLoader is a document loader that reads markdown content from a directory.
type MarkdownDirectoryLoader struct {
	filePath      string
	encoding      string
	baseSourceURL string
}

// NewMarkdownDirectory creates a new MarkdownDirectoryLoader with the given file path and encoding.
func NewMarkdownDirectory(filePath string, baseSourceURL string, encoding ...string) *MarkdownDirectoryLoader {
	defaultEncoding := "utf-8"

	if len(encoding) > 0 {
		return &MarkdownDirectoryLoader{
			filePath:      filePath,
			encoding:      encoding[0],
			baseSourceURL: baseSourceURL,
		}
	}

	return &MarkdownDirectoryLoader{
		filePath:      filePath,
		encoding:      defaultEncoding,
		baseSourceURL: baseSourceURL,
	}
}

// Load retrieves data from a directory and returns a list of schema.Document objects.
func (n *MarkdownDirectoryLoader) Load() ([]schema.Document, error) {
	filePaths := make([]string, 0)
	err := filepath.Walk(n.filePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		if filepath.Ext(info.Name()) == ".md" {
			filePaths = append(filePaths, path)
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("unable to loop through directory [%s]: %w", n.filePath, err)
	}

	documents := make([]schema.Document, 0, len(filePaths))
	for _, filePath := range filePaths {
		text, err := os.ReadFile(filePath)
		if err != nil {
			return nil, err
		}

		relativePath, err := filepath.Rel(n.filePath, filePath)
		if err != nil {
			return nil, fmt.Errorf("unable to get relative path for file [%s]: %w", filePath, err)
		}

		metadata := map[string]interface{}{"source": relativePath}
		if n.baseSourceURL != "" {
			metadata["source_url"] = filepath.Join(n.baseSourceURL, relativePath)
		}
		documents = append(
			documents,
			schema.Document{
				PageContent: string(text),
				Metadata: metadata,
			})
	}

	return documents, nil
}
