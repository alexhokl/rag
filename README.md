# rag

## Description

This is an application allows user asking questions against documents stored in
a vector database and presents the answers using user-specified large language
model (LLM) via [Ollama](https://ollama.com/). This application also allows user
loading documents from specified sources into a vector database. This
capabilities are useful for building retrieval augmented generation (RAG)
applications where custom documents are used to generate answers to user
questions or queries.

This application assumes vector database and Ollama are setup locally.

## Usage

```
Retrieval augmented generation application

Usage:
  rag [command]

Available Commands:
  ask         Ask a question on the documents stored in the specified vector database
  load        Load documents into a vector database
  query       Query against documents stored in the specified vector database
```

## Libraries

- [tmc/langchaingo](https://github.com/tmc/langchaingo)
- [amikos-tech/chroma-go](https://github.com/amikos-tech/chroma-go)
