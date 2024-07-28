package documentloaders_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/alexhokl/rag/documentloaders"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/schema"
	"github.com/zeebo/assert"
)

func TestMarkdownDirectoryLoader_Load(t *testing.T) {
	t.Parallel()

	// Create a temporary test directory
	tempDir, err := os.MkdirTemp("", "markdown_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create sample Markdown files in the temporary directory
	testFiles := []struct {
		name          string
		content       string
		baseSourceURL string
		expected      schema.Document
	}{
		{
			name:          "test1.md",
			content:       "# Test Document 1\nThis is test document 1.",
			baseSourceURL: "",
			expected: schema.Document{
				PageContent: "# Test Document 1\nThis is test document 1.",
				Metadata:    map[string]interface{}{"source": "test1.md"},
			},
		},
		{
			name:          "test2.md",
			content:       "# Test Document 2\nThis is test document 2.",
			baseSourceURL: "",
			expected: schema.Document{
				PageContent: "# Test Document 2\nThis is test document 2.",
				Metadata:    map[string]interface{}{"source": "test2.md"},
			},
		},
		{
			name:    "test3.md",
			content: "# Test Document 3\nThis is test document 3.",
			baseSourceURL: "https://github.com",
			expected: schema.Document{
				PageContent: "# Test Document 3\nThis is test document 3.",
				Metadata:    map[string]interface{}{"source": "test3.md", "source_url": "https:/github.com/test3.md"},
			},
		},
	}

	for _, file := range testFiles {
		filePath := filepath.Join(tempDir, file.name)
		err := os.WriteFile(filePath, []byte(file.content), 0o600)
		require.NoError(t, err)
	}

	for i, expected := range testFiles {
		loader := documentloaders.NewMarkdownDirectory(tempDir, expected.baseSourceURL)

		docs, err := loader.Load()
		require.NoError(t, err)

		require.Len(t, docs, len(testFiles))
		assert.Equal(t, expected.expected.PageContent, docs[i].PageContent)
		assert.Equal(t, expected.expected.Score, docs[i].Score)
		for key, value := range expected.expected.Metadata {
			assert.Equal(t, value, docs[i].Metadata[key])
		}
	}
}
