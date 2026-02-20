package main

import (
	"context"
	"fmt"
	"runtime"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ═══════════════════════════════════════════════════════════════════
//  RESOURCE 1: Dynamic greeting by name
//  URI pattern: greeting://{name}
//  Example:     greeting://Alice  → "Hello, Alice!"
// ═══════════════════════════════════════════════════════════════════

func GetGreeting(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
	name := req.Params.URI[len("greeting://"):]
	if name == "" {
		name = "World"
	}
	return &mcp.ReadResourceResult{
		Contents: []*mcp.ResourceContents{
			{
				URI:      req.Params.URI,
				MIMEType: "text/plain",
				Text:     fmt.Sprintf("Hello, %s! Welcome to the Go MCP server.", name),
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  RESOURCE 2: Server info (static)
//  URI: info://server
// ═══════════════════════════════════════════════════════════════════

func GetServerInfo(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
	info := fmt.Sprintf(`Server:    Demo MCP Server
Version:   v1.0.0
Go:        %s
OS:        %s
Arch:      %s
StartTime: %s
`, runtime.Version(), runtime.GOOS, runtime.GOARCH, time.Now().Format(time.RFC1123))

	return &mcp.ReadResourceResult{
		Contents: []*mcp.ResourceContents{
			{
				URI:      req.Params.URI,
				MIMEType: "text/plain",
				Text:     info,
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  RESOURCE 3: User profile by ID
//  URI pattern: user://{id}
//  Example:     user://42  → JSON profile
// ═══════════════════════════════════════════════════════════════════

func GetUserProfile(ctx context.Context, req *mcp.ReadResourceRequest) (*mcp.ReadResourceResult, error) {
	id := req.Params.URI[len("user://"):]

	// In a real app this would hit a database
	profile := fmt.Sprintf(`{
  "id": "%s",
  "name": "User %s",
  "email": "user%s@example.com",
  "joined": "2024-01-01",
  "status": "active"
}`, id, id, id)

	return &mcp.ReadResourceResult{
		Contents: []*mcp.ResourceContents{
			{
				URI:      req.Params.URI,
				MIMEType: "application/json",
				Text:     profile,
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  RegisterResources — called from main.go
// ═══════════════════════════════════════════════════════════════════

func RegisterResources(server *mcp.Server) {
	// Static resource — fixed URI
	server.AddResource(&mcp.Resource{
		URI:         "info://server",
		Name:        "server_info",
		Description: "Live info about this MCP server (Go version, OS, uptime)",
		MIMEType:    "text/plain",
	}, GetServerInfo)

	// Dynamic resource templates — URI contains a variable {name} / {id}
	server.AddResourceTemplate(&mcp.ResourceTemplate{
		URITemplate: "greeting://{name}",
		Name:        "greeting",
		Description: "Get a personalized greeting — use any name, e.g. greeting://Alice",
		MIMEType:    "text/plain",
	}, GetGreeting)

	server.AddResourceTemplate(&mcp.ResourceTemplate{
		URITemplate: "user://{id}",
		Name:        "user_profile",
		Description: "Get a user profile by ID — e.g. user://42",
		MIMEType:    "application/json",
	}, GetUserProfile)
}