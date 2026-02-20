package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ═══════════════════════════════════════════════════════════════════
//  TOOL 1: Add two numbers
// ═══════════════════════════════════════════════════════════════════

type AddInput struct {
	A int `json:"a" jsonschema:"the first number"`
	B int `json:"b" jsonschema:"the second number"`
}

type AddOutput struct {
	Result int `json:"result"`
}

func Add(ctx context.Context, req *mcp.CallToolRequest, input AddInput) (
	*mcp.CallToolResult, AddOutput, error,
) {
	return nil, AddOutput{Result: input.A + input.B}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  TOOL 2: Multiply two numbers
// ═══════════════════════════════════════════════════════════════════

type MultiplyInput struct {
	A float64 `json:"a" jsonschema:"the first number"`
	B float64 `json:"b" jsonschema:"the second number"`
}

type MultiplyOutput struct {
	Result float64 `json:"result"`
}

func Multiply(ctx context.Context, req *mcp.CallToolRequest, input MultiplyInput) (
	*mcp.CallToolResult, MultiplyOutput, error,
) {
	return nil, MultiplyOutput{Result: input.A * input.B}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  TOOL 3: Word count
// ═══════════════════════════════════════════════════════════════════

type WordCountInput struct {
	Text string `json:"text" jsonschema:"the text to count words in"`
}

type WordCountOutput struct {
	Words      int `json:"words"`
	Characters int `json:"characters"`
	Lines      int `json:"lines"`
}

func WordCount(ctx context.Context, req *mcp.CallToolRequest, input WordCountInput) (
	*mcp.CallToolResult, WordCountOutput, error,
) {
	words := 0
	if strings.TrimSpace(input.Text) != "" {
		words = len(strings.Fields(input.Text))
	}
	return nil, WordCountOutput{
		Words:      words,
		Characters: len(input.Text),
		Lines:      len(strings.Split(input.Text, "\n")),
	}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  TOOL 4: Reverse a string
// ═══════════════════════════════════════════════════════════════════

type ReverseInput struct {
	Text string `json:"text" jsonschema:"the text to reverse"`
}

type ReverseOutput struct {
	Reversed string `json:"reversed"`
}

func ReverseString(ctx context.Context, req *mcp.CallToolRequest, input ReverseInput) (
	*mcp.CallToolResult, ReverseOutput, error,
) {
	runes := []rune(input.Text)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return nil, ReverseOutput{Reversed: string(runes)}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  TOOL 5: Temperature converter
// ═══════════════════════════════════════════════════════════════════

type TempConvertInput struct {
	Value float64 `json:"value" jsonschema:"the temperature value to convert"`
	From  string  `json:"from"  jsonschema:"source unit: celsius, fahrenheit, or kelvin"`
	To    string  `json:"to"    jsonschema:"target unit: celsius, fahrenheit, or kelvin"`
}

type TempConvertOutput struct {
	Result float64 `json:"result"`
	Unit   string  `json:"unit"`
}

func ConvertTemperature(ctx context.Context, req *mcp.CallToolRequest, input TempConvertInput) (
	*mcp.CallToolResult, TempConvertOutput, error,
) {
	// Convert everything to Celsius first
	var celsius float64
	switch strings.ToLower(input.From) {
	case "celsius":
		celsius = input.Value
	case "fahrenheit":
		celsius = (input.Value - 32) * 5 / 9
	case "kelvin":
		celsius = input.Value - 273.15
	default:
		return nil, TempConvertOutput{}, fmt.Errorf("unknown unit: %s", input.From)
	}

	// Convert from Celsius to target
	var result float64
	switch strings.ToLower(input.To) {
	case "celsius":
		result = celsius
	case "fahrenheit":
		result = celsius*9/5 + 32
	case "kelvin":
		result = celsius + 273.15
	default:
		return nil, TempConvertOutput{}, fmt.Errorf("unknown unit: %s", input.To)
	}

	return nil, TempConvertOutput{Result: result, Unit: input.To}, nil
}

// ═══════════════════════════════════════════════════════════════════
//  RegisterTools — called from main.go
// ═══════════════════════════════════════════════════════════════════

func RegisterTools(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{
		Name:        "add",
		Description: "Add two integers together",
	}, Add)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "multiply",
		Description: "Multiply two numbers together",
	}, Multiply)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "word_count",
		Description: "Count words, characters, and lines in a text",
	}, WordCount)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "reverse_string",
		Description: "Reverse any string of text",
	}, ReverseString)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "convert_temperature",
		Description: "Convert temperatures between celsius, fahrenheit, and kelvin",
	}, ConvertTemperature)
}