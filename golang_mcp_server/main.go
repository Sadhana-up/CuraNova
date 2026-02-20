package main

import (
	"log"
	"net/http"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	// ─── Create server ────────────────────────────────────────────────────────
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "Demo",
		Version: "v1.0.0",
	}, nil)

	// ─── Register everything (defined in their own files) ─────────────────────
	RegisterTools(server)     // tools.go     — add, multiply, word_count, reverse, temp convert
	RegisterResources(server) // resources.go — server info, greeting://{name}, user://{id}
	RegisterPrompts(server)   // prompts.go   — greet_user, code_review, summarize, translate

	// ─── Streamable HTTP transport ────────────────────────────────────────────
	handler := mcp.NewStreamableHTTPHandler(
		func(r *http.Request) *mcp.Server { return server },
		nil,
	)

	http.HandleFunc("/mcp", handler.ServeHTTP)

	log.Println("✅ MCP server running at http://localhost:8080/mcp")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}