// ──────────────────────────────────────────────────────────
// Types for the ADK Bidi-streaming chat UI
// ──────────────────────────────────────────────────────────

/** A single chat message displayed in the message area */
export interface ChatMessage {
  id: string;
  type: "user" | "agent" | "system" | "image" | "tool-stream";
  text: string;
  imageDataUrl?: string;
  isPartial?: boolean;
  isInterrupted?: boolean;
  isTranscription?: boolean;
}

/** Console log entry */
export interface ConsoleEntry {
  id: string;
  timestamp: string;
  type: "outgoing" | "incoming" | "error";
  content: string;
  data?: Record<string, unknown> | null;
  emoji?: string;
  author?: string;
  isAudio?: boolean;
}

/** ADK event received from the server via WebSocket */
export interface AdkEvent {
  author?: string;
  turnComplete?: boolean;
  interrupted?: boolean;
  inputTranscription?: {
    text?: string;
    finished?: boolean;
  };
  outputTranscription?: {
    text?: string;
    finished?: boolean;
  };
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
  content?: {
    parts?: AdkPart[];
  };
}

export interface AdkPart {
  text?: string;
  inlineData?: {
    mimeType?: string;
    data?: string;
  };
  executableCode?: {
    code?: string;
    language?: string;
  };
  codeExecutionResult?: {
    outcome?: string;
    output?: string;
  };
}

/** Connection status */
export type ConnectionStatus = "connecting" | "connected" | "disconnected";
