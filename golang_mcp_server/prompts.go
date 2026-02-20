package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ═══════════════════════════════════════════════════════════════════
//  PROMPT 1: Greet user
//  Args: name (required), style (optional: friendly/formal/casual)
// ═══════════════════════════════════════════════════════════════════

func GreetUserPrompt(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	name := req.Params.Arguments["name"]
	style := req.Params.Arguments["style"]

	styles := map[string]string{
		"friendly": "Please write a warm and friendly greeting",
		"formal":   "Please write a formal and professional greeting",
		"casual":   "Please write a casual and relaxed greeting",
	}

	instruction, ok := styles[style]
	if !ok {
		instruction = styles["friendly"]
	}

	return &mcp.GetPromptResult{
		Description: "Generates a personalized greeting in a chosen style",
		Messages: []*mcp.PromptMessage{
			{
				Role:    "user",
				Content: &mcp.TextContent{Text: fmt.Sprintf("%s for someone named %s.", instruction, name)},
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  PROMPT 2: Code review
//  Args: code (required), language (optional), focus (optional)
// ═══════════════════════════════════════════════════════════════════

func CodeReviewPrompt(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	code := req.Params.Arguments["code"]
	language := req.Params.Arguments["language"]
	focus := req.Params.Arguments["focus"]

	if language == "" {
		language = "the given programming language"
	}
	if focus == "" {
		focus = "overall code quality, readability, performance, and security"
	}

	prompt := fmt.Sprintf(`Please review the following %s code. Focus on: %s.

Provide:
1. A brief summary of what the code does
2. Any bugs or issues found
3. Suggestions for improvement
4. A rating from 1-10

Code to review:
`+"```"+`
%s
`+"```", language, focus, code)

	return &mcp.GetPromptResult{
		Description: "Generates a code review prompt for the provided code",
		Messages: []*mcp.PromptMessage{
			{
				Role:    "user",
				Content: &mcp.TextContent{Text: prompt},
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  PROMPT 3: Summarize text
//  Args: text (required), length (optional: short/medium/long),
//        format (optional: bullet/paragraph)
// ═══════════════════════════════════════════════════════════════════

func SummarizePrompt(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	text := req.Params.Arguments["text"]
	length := req.Params.Arguments["length"]
	format := req.Params.Arguments["format"]

	lengthInstructions := map[string]string{
		"short":  "in 1-2 sentences",
		"medium": "in a short paragraph (3-5 sentences)",
		"long":   "in detail, covering all key points",
	}
	formatInstructions := map[string]string{
		"bullet":    "Use bullet points.",
		"paragraph": "Use flowing prose paragraphs.",
	}

	lengthInstr, ok := lengthInstructions[length]
	if !ok {
		lengthInstr = lengthInstructions["medium"]
	}
	formatInstr, ok := formatInstructions[format]
	if !ok {
		formatInstr = formatInstructions["paragraph"]
	}

	prompt := fmt.Sprintf("Please summarize the following text %s. %s\n\nText:\n%s",
		lengthInstr, formatInstr, text)

	return &mcp.GetPromptResult{
		Description: "Generates a summarization prompt with configurable length and format",
		Messages: []*mcp.PromptMessage{
			{
				Role:    "user",
				Content: &mcp.TextContent{Text: prompt},
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  PROMPT 4: Translate text
//  Args: text (required), target_language (required),
//        tone (optional: formal/casual)
// ═══════════════════════════════════════════════════════════════════

func TranslatePrompt(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	text := req.Params.Arguments["text"]
	targetLang := req.Params.Arguments["target_language"]
	tone := req.Params.Arguments["tone"]

	toneInstr := ""
	if strings.ToLower(tone) == "formal" {
		toneInstr = " Use formal language."
	} else if strings.ToLower(tone) == "casual" {
		toneInstr = " Use casual, conversational language."
	}

	prompt := fmt.Sprintf("Translate the following text to %s.%s Only return the translation, nothing else.\n\nText:\n%s",
		targetLang, toneInstr, text)

	return &mcp.GetPromptResult{
		Description: "Generates a translation prompt for the given text and target language",
		Messages: []*mcp.PromptMessage{
			{
				Role:    "user",
				Content: &mcp.TextContent{Text: prompt},
			},
		},
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  RegisterPrompts — called from main.go
// ═══════════════════════════════════════════════════════════════════

func RegisterPrompts(server *mcp.Server) {
	server.AddPrompt(&mcp.Prompt{
		Name:        "greet_user",
		Description: "Generate a personalized greeting in friendly, formal, or casual style",
		Arguments: []*mcp.PromptArgument{
			{Name: "name", Description: "The person's name", Required: true},
			{Name: "style", Description: "Tone: friendly | formal | casual (default: friendly)", Required: false},
		},
	}, GreetUserPrompt)

	server.AddPrompt(&mcp.Prompt{
		Name:        "code_review",
		Description: "Generate a thorough code review prompt for any programming language",
		Arguments: []*mcp.PromptArgument{
			{Name: "code", Description: "The code to review", Required: true},
			{Name: "language", Description: "Programming language (e.g. Go, Python)", Required: false},
			{Name: "focus", Description: "What to focus on (e.g. security, performance)", Required: false},
		},
	}, CodeReviewPrompt)

	server.AddPrompt(&mcp.Prompt{
		Name:        "summarize",
		Description: "Generate a summarization prompt with configurable length and format",
		Arguments: []*mcp.PromptArgument{
			{Name: "text", Description: "The text to summarize", Required: true},
			{Name: "length", Description: "Summary length: short | medium | long (default: medium)", Required: false},
			{Name: "format", Description: "Output format: bullet | paragraph (default: paragraph)", Required: false},
		},
	}, SummarizePrompt)

	server.AddPrompt(&mcp.Prompt{
		Name:        "translate",
		Description: "Generate a translation prompt for any target language",
		Arguments: []*mcp.PromptArgument{
			{Name: "text", Description: "The text to translate", Required: true},
			{Name: "target_language", Description: "Target language (e.g. Spanish, Japanese)", Required: true},
			{Name: "tone", Description: "Tone: formal | casual (optional)", Required: false},
		},
	}, TranslatePrompt)
}