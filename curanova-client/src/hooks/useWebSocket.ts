"use client";
// ──────────────────────────────────────────────────────────
// useWebSocket — manages the main ADK WebSocket connection.
//
// New architecture:
//   • ALL messages (text, URL, uploaded images) go through the
//     main ADK WebSocket (/ws/{user_id}/{session_id}).
//   • The Gemini agent decides whether a medical tool call is
//     needed based on the message content.
//   • When the agent calls a tool, it emits a medical_stream_trigger
//     event. This hook catches that event and opens a direct
//     WebSocket to /ws/analyze, streaming tokens into the green
//     "Medical Analysis" bubble.
// ──────────────────────────────────────────────────────────

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AdkEvent,
  ChatMessage,
  ConnectionStatus,
  ConsoleEntry,
} from "@/types/chat";
import {
  randomId,
  formatTimestamp,
  cleanCJKSpaces,
  sanitizeEventForDisplay,
  base64ToArray,
} from "@/lib/chat-utils";

interface UseWebSocketOptions {
  serverUrl: string;
  enableProactivity: boolean;
  enableAffectiveDialog: boolean;
  showAudioEvents: boolean;
  onAudioData?: (buffer: ArrayBuffer) => void;
}

interface UseWebSocketReturn {
  messages: ChatMessage[];
  consoleEntries: ConsoleEntry[];
  connectionStatus: ConnectionStatus;
  sendTextMessage: (text: string) => void;
  sendImage: (base64Data: string, imageDataUrl?: string, prompt?: string) => void;
  sendImageUpload: (file: File, prompt?: string) => void;
  sendAudioChunk: (pcmData: ArrayBuffer) => void;
  clearConsole: () => void;
}

const userId = "demo-user";
const sessionId = "demo-session-" + randomId();

export function useWebSocket({
  serverUrl,
  enableProactivity,
  enableAffectiveDialog,
  showAudioEvents,
  onAudioData,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [consoleEntries, setConsoleEntries] = useState<ConsoleEntry[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("connecting");

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Mutable refs for streaming state (avoid re-renders on every token)
  const currentMessageIdRef = useRef<string | null>(null);
  const currentMessageTextRef = useRef<string>("");
  const currentInputTranscriptionIdRef = useRef<string | null>(null);
  const currentInputTranscriptionTextRef = useRef<string>("");
  const currentOutputTranscriptionIdRef = useRef<string | null>(null);
  const currentOutputTranscriptionTextRef = useRef<string>("");
  const inputTranscriptionFinishedRef = useRef<boolean>(false);

  // Refs for callbacks that may change
  const onAudioDataRef = useRef(onAudioData);
  onAudioDataRef.current = onAudioData;
  const showAudioEventsRef = useRef(showAudioEvents);
  showAudioEventsRef.current = showAudioEvents;

  // ── Direct-streaming state for medical image analysis ──
  const analyzeWsRef = useRef<WebSocket | null>(null);
  const analyzeStreamTextRef = useRef<string>("");
  const analyzeStreamIdRef = useRef<string | null>(null);

  // ── Console entry helper ──
  const addConsoleEntry = useCallback(
    (
      type: ConsoleEntry["type"],
      content: string,
      data: Record<string, unknown> | null = null,
      emoji?: string,
      author?: string,
      isAudio = false
    ) => {
      if (isAudio && !showAudioEventsRef.current) return;
      const entry: ConsoleEntry = {
        id: randomId(),
        timestamp: formatTimestamp(),
        type,
        content,
        data,
        emoji,
        author,
        isAudio,
      };
      setConsoleEntries((prev) => [...prev, entry]);
    },
    []
  );

  // ── Message helpers ──
  const addSystemMessage = useCallback((text: string) => {
    setMessages((prev) => [
      ...prev,
      { id: randomId(), type: "system", text },
    ]);
  }, []);

  const addUserMessage = useCallback((text: string) => {
    setMessages((prev) => [
      ...prev,
      { id: randomId(), type: "user", text, isPartial: false },
    ]);
  }, []);

  const addUserImageMessage = useCallback((imageDataUrl: string) => {
    setMessages((prev) => [
      ...prev,
      { id: randomId(), type: "image", text: "", imageDataUrl },
    ]);
  }, []);

  // Update or create an agent message by id
  const upsertAgentMessage = useCallback(
    (
      msgId: string,
      text: string,
      isPartial: boolean,
      opts?: Partial<ChatMessage>
    ) => {
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === msgId);
        if (idx === -1) {
          return [
            ...prev,
            { id: msgId, type: "agent", text, isPartial, ...opts },
          ];
        }
        const updated = [...prev];
        updated[idx] = { ...updated[idx], text, isPartial, ...opts };
        return updated;
      });
    },
    []
  );

  // Update or create a tool-stream message (green Medical Analysis bubble)
  const upsertToolStreamMessage = useCallback(
    (msgId: string, text: string, isPartial: boolean) => {
      setMessages((prev) => {
        const idx = prev.findIndex((m) => m.id === msgId);
        if (idx === -1) {
          return [
            ...prev,
            { id: msgId, type: "tool-stream", text, isPartial },
          ];
        }
        const updated = [...prev];
        updated[idx] = { ...updated[idx], text, isPartial };
        return updated;
      });
    },
    []
  );

  // Mark a message as interrupted
  const markInterrupted = useCallback((msgId: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === msgId ? { ...m, isPartial: false, isInterrupted: true } : m
      )
    );
  }, []);

  // Finalize a message (remove partial state)
  const finalizeMessage = useCallback((msgId: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === msgId ? { ...m, isPartial: false } : m))
    );
  }, []);

  // ── Direct analysis via /ws/analyze ──────────────────────────────────────
  // Called when the agent emits a medical_stream_trigger event.
  // payload matches the /ws/analyze JSON schema:
  //   { prompt, image_url?, image_url_2?, image_b64?, image_b64_2?, max_new_tokens? }
  const startDirectAnalysis = useCallback(
    (payload: {
      prompt: string;
      image_url?: string;
      image_url_2?: string;
      image_b64?: string;
      image_b64_2?: string;
      max_new_tokens?: number;
    }) => {
      const wsProtocol = serverUrl.startsWith("https") ? "wss:" : "ws:";
      const host = serverUrl.replace(/^https?:\/\//, "");
      const analyzeUrl = `${wsProtocol}//${host}/ws/analyze`;

      const streamId = randomId();
      analyzeStreamIdRef.current = streamId;
      analyzeStreamTextRef.current = "";

      upsertToolStreamMessage(streamId, "", true);

      addConsoleEntry(
        "outgoing",
        "Medical Stream Triggered",
        { endpoint: analyzeUrl, ...payload },
        "🩺",
        "system"
      );

      const ws = new WebSocket(analyzeUrl);
      analyzeWsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ max_new_tokens: 500, ...payload }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.token) {
          analyzeStreamTextRef.current += data.token;
          upsertToolStreamMessage(streamId, analyzeStreamTextRef.current, true);
        } else if (data.status === "done") {
          upsertToolStreamMessage(streamId, analyzeStreamTextRef.current, false);
          addConsoleEntry(
            "incoming",
            "Medical Analysis Complete",
            {
              tokens_received: analyzeStreamTextRef.current.length,
              preview: analyzeStreamTextRef.current.substring(0, 100) + "...",
            },
            "✅",
            "tool"
          );
          analyzeStreamIdRef.current = null;
          analyzeStreamTextRef.current = "";
          ws.close();
          analyzeWsRef.current = null;
        } else if (data.error) {
          upsertToolStreamMessage(
            streamId,
            analyzeStreamTextRef.current + "\n[Error]: " + data.error,
            false
          );
          addConsoleEntry(
            "error",
            "Analysis Error: " + data.error,
            data,
            "⚠️",
            "tool"
          );
          analyzeStreamIdRef.current = null;
          analyzeStreamTextRef.current = "";
          ws.close();
          analyzeWsRef.current = null;
        }
      };

      ws.onerror = () => {
        addConsoleEntry(
          "error",
          "Analyze WebSocket Error",
          { error: "connection error" },
          "⚠️",
          "system"
        );
      };

      ws.onclose = () => {
        if (analyzeStreamIdRef.current === streamId) {
          upsertToolStreamMessage(streamId, analyzeStreamTextRef.current, false);
        }
        analyzeWsRef.current = null;
      };
    },
    [serverUrl, addConsoleEntry, upsertToolStreamMessage]
  );

  // ── Build the main WebSocket URL ──
  const buildWsUrl = useCallback(() => {
    const wsProtocol = serverUrl.startsWith("https") ? "wss:" : "ws:";
    const host = serverUrl.replace(/^https?:\/\//, "");
    let url = `${wsProtocol}//${host}/ws/${userId}/${sessionId}`;
    const params = new URLSearchParams();
    if (enableProactivity) params.append("proactivity", "true");
    if (enableAffectiveDialog) params.append("affective_dialog", "true");
    const qs = params.toString();
    if (qs) url += "?" + qs;
    return url;
  }, [serverUrl, enableProactivity, enableAffectiveDialog]);

  // ── Handle incoming ADK event (or medical_stream_trigger) ──
  const handleAdkEvent = useCallback(
    (rawData: string) => {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(rawData);
      } catch {
        return;
      }

      // ── Medical stream trigger from agent tool call ──────────────────────
      if (parsed.medical_stream_trigger === true) {
        const payload = parsed.payload as Parameters<typeof startDirectAnalysis>[0];
        addConsoleEntry(
          "incoming",
          "Agent triggered medical analysis",
          { payload },
          "🩺",
          "agent"
        );
        startDirectAnalysis(payload);
        return;
      }

      const adkEvent = parsed as AdkEvent;

      // ── Build console summary ──
      let eventSummary = "Event";
      let eventEmoji = "📨";
      const author = adkEvent.author || "system";

      if (adkEvent.turnComplete) {
        eventSummary = "Turn Complete";
        eventEmoji = "✅";
      } else if (adkEvent.interrupted) {
        eventSummary = "Interrupted";
        eventEmoji = "⏸️";
      } else if (adkEvent.inputTranscription) {
        const t = adkEvent.inputTranscription.text || "";
        eventSummary = `Input Transcription: "${t.length > 60 ? t.substring(0, 60) + "..." : t}"`;
        eventEmoji = "📝";
      } else if (adkEvent.outputTranscription) {
        const t = adkEvent.outputTranscription.text || "";
        eventSummary = `Output Transcription: "${t.length > 60 ? t.substring(0, 60) + "..." : t}"`;
        eventEmoji = "📝";
      } else if (adkEvent.usageMetadata) {
        const u = adkEvent.usageMetadata;
        eventSummary = `Token Usage: ${(u.totalTokenCount || 0).toLocaleString()} total`;
        eventEmoji = "📊";
      } else if (adkEvent.content?.parts) {
        const hasText = adkEvent.content.parts.some((p) => p.text);
        const hasAudio = adkEvent.content.parts.some((p) => p.inlineData);
        const hasExecCode = adkEvent.content.parts.some((p) => p.executableCode);
        const hasCodeResult = adkEvent.content.parts.some((p) => p.codeExecutionResult);

        if (hasExecCode) {
          const cp = adkEvent.content.parts.find((p) => p.executableCode);
          if (cp?.executableCode) {
            const code = cp.executableCode.code || "";
            const lang = cp.executableCode.language || "unknown";
            const trunc = code.length > 60 ? code.substring(0, 60).replace(/\n/g, " ") + "..." : code.replace(/\n/g, " ");
            eventSummary = `Executable Code (${lang}): ${trunc}`;
            eventEmoji = "💻";
          }
        }
        if (hasCodeResult) {
          const rp = adkEvent.content.parts.find((p) => p.codeExecutionResult);
          if (rp?.codeExecutionResult) {
            const outcome = rp.codeExecutionResult.outcome || "UNKNOWN";
            const output = rp.codeExecutionResult.output || "";
            const truncOut = output.length > 60 ? output.substring(0, 60).replace(/\n/g, " ") + "..." : output.replace(/\n/g, " ");
            eventSummary = `Code Execution Result (${outcome}): ${truncOut}`;
            eventEmoji = outcome === "OUTCOME_OK" ? "✅" : "❌";
          }
        }
        if (hasText) {
          const tp = adkEvent.content.parts.find((p) => p.text);
          if (tp?.text) {
            const t = tp.text;
            eventSummary = `Text: "${t.length > 80 ? t.substring(0, 80) + "..." : t}"`;
          } else {
            eventSummary = "Text Response";
          }
          eventEmoji = "💭";
        }
        if (hasAudio) {
          const ap = adkEvent.content.parts.find((p) => p.inlineData);
          if (ap?.inlineData) {
            const mime = ap.inlineData.mimeType || "unknown";
            const dataLen = ap.inlineData.data?.length || 0;
            const byteSize = Math.floor(dataLen * 0.75);
            eventSummary = `Audio Response: ${mime} (${byteSize.toLocaleString()} bytes)`;
          } else {
            eventSummary = "Audio Response";
          }
          eventEmoji = "🔊";

          const sanitized = sanitizeEventForDisplay(adkEvent as unknown as Record<string, unknown>);
          addConsoleEntry("incoming", eventSummary, sanitized, eventEmoji, author, true);
        }
      }

      // Log non-audio-only events
      const isAudioOnly =
        adkEvent.content?.parts?.some((p) => p.inlineData) &&
        !adkEvent.content?.parts?.some((p) => p.text);
      if (!isAudioOnly) {
        const sanitized = sanitizeEventForDisplay(adkEvent as unknown as Record<string, unknown>);
        addConsoleEntry("incoming", eventSummary, sanitized, eventEmoji, author);
      }

      // ── Turn complete ──
      if (adkEvent.turnComplete === true) {
        if (currentMessageIdRef.current) {
          finalizeMessage(currentMessageIdRef.current);
        }
        if (currentOutputTranscriptionIdRef.current) {
          finalizeMessage(currentOutputTranscriptionIdRef.current);
        }
        currentMessageIdRef.current = null;
        currentMessageTextRef.current = "";
        currentOutputTranscriptionIdRef.current = null;
        currentOutputTranscriptionTextRef.current = "";
        inputTranscriptionFinishedRef.current = false;
        return;
      }

      // ── Interrupted ──
      if (adkEvent.interrupted === true) {
        if (currentMessageIdRef.current) {
          markInterrupted(currentMessageIdRef.current);
        }
        if (currentOutputTranscriptionIdRef.current) {
          markInterrupted(currentOutputTranscriptionIdRef.current);
        }
        currentMessageIdRef.current = null;
        currentMessageTextRef.current = "";
        currentOutputTranscriptionIdRef.current = null;
        currentOutputTranscriptionTextRef.current = "";
        inputTranscriptionFinishedRef.current = false;
        return;
      }

      // ── Input transcription ──
      if (adkEvent.inputTranscription?.text) {
        const transcriptionText = adkEvent.inputTranscription.text;
        const isFinished = adkEvent.inputTranscription.finished;

        if (inputTranscriptionFinishedRef.current) return;

        if (!currentInputTranscriptionIdRef.current) {
          const id = randomId();
          currentInputTranscriptionIdRef.current = id;
          currentInputTranscriptionTextRef.current = cleanCJKSpaces(transcriptionText);
          upsertAgentMessage(id, currentInputTranscriptionTextRef.current, !isFinished, {
            type: "user",
            isTranscription: true,
          });
        } else {
          if (!currentOutputTranscriptionIdRef.current && !currentMessageIdRef.current) {
            if (isFinished) {
              currentInputTranscriptionTextRef.current = cleanCJKSpaces(transcriptionText);
            } else {
              currentInputTranscriptionTextRef.current = cleanCJKSpaces(
                currentInputTranscriptionTextRef.current + transcriptionText
              );
            }
            upsertAgentMessage(
              currentInputTranscriptionIdRef.current,
              currentInputTranscriptionTextRef.current,
              !isFinished,
              { type: "user", isTranscription: true }
            );
          }
        }

        if (isFinished) {
          currentInputTranscriptionIdRef.current = null;
          currentInputTranscriptionTextRef.current = "";
          inputTranscriptionFinishedRef.current = true;
        }
        return;
      }

      // ── Output transcription ──
      if (adkEvent.outputTranscription?.text) {
        const transcriptionText = adkEvent.outputTranscription.text;
        const isFinished = adkEvent.outputTranscription.finished;

        if (currentInputTranscriptionIdRef.current && !currentOutputTranscriptionIdRef.current) {
          finalizeMessage(currentInputTranscriptionIdRef.current);
          currentInputTranscriptionIdRef.current = null;
          currentInputTranscriptionTextRef.current = "";
          inputTranscriptionFinishedRef.current = true;
        }

        if (!currentOutputTranscriptionIdRef.current) {
          const id = randomId();
          currentOutputTranscriptionIdRef.current = id;
          currentOutputTranscriptionTextRef.current = transcriptionText;
          upsertAgentMessage(id, transcriptionText, !isFinished, { isTranscription: true });
        } else {
          if (isFinished) {
            currentOutputTranscriptionTextRef.current = transcriptionText;
          } else {
            currentOutputTranscriptionTextRef.current += transcriptionText;
          }
          upsertAgentMessage(
            currentOutputTranscriptionIdRef.current,
            currentOutputTranscriptionTextRef.current,
            !isFinished,
            { isTranscription: true }
          );
        }

        if (isFinished) {
          currentOutputTranscriptionIdRef.current = null;
          currentOutputTranscriptionTextRef.current = "";
        }
        return;
      }

      // ── Content events (text or audio) ──
      if (adkEvent.content?.parts) {
        if (
          currentInputTranscriptionIdRef.current &&
          !currentMessageIdRef.current &&
          !currentOutputTranscriptionIdRef.current
        ) {
          finalizeMessage(currentInputTranscriptionIdRef.current);
          currentInputTranscriptionIdRef.current = null;
          currentInputTranscriptionTextRef.current = "";
          inputTranscriptionFinishedRef.current = true;
        }

        for (const part of adkEvent.content.parts) {
          // Audio
          if (part.inlineData) {
            const mime = part.inlineData.mimeType || "";
            if (mime.startsWith("audio/pcm") && part.inlineData.data) {
              onAudioDataRef.current?.(base64ToArray(part.inlineData.data));
            }
          }

          // Text
          if (part.text) {
            if (!currentMessageIdRef.current) {
              const id = randomId();
              currentMessageIdRef.current = id;
              currentMessageTextRef.current = part.text;
              upsertAgentMessage(id, part.text, true);
            } else {
              currentMessageTextRef.current += part.text;
              upsertAgentMessage(currentMessageIdRef.current, currentMessageTextRef.current, true);
            }
          }
        }
      }
    },
    [
      startDirectAnalysis,
      addConsoleEntry,
      upsertAgentMessage,
      finalizeMessage,
      markInterrupted,
    ]
  );

  // ── Connect WebSocket ──
  const connect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const url = buildWsUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionStatus("connected");
      addSystemMessage("Connected to ADK streaming server");
      addConsoleEntry("incoming", "WebSocket Connected", { userId, sessionId, url }, "🔌", "system");
    };

    ws.onmessage = (event) => {
      // Pass raw string so handleAdkEvent can detect medical_stream_trigger
      handleAdkEvent(event.data);
    };

    ws.onclose = () => {
      setConnectionStatus("disconnected");
      addSystemMessage("Connection closed. Reconnecting in 5 seconds...");
      addConsoleEntry(
        "error",
        "WebSocket Disconnected",
        { status: "Connection closed", reconnecting: true },
        "🔌",
        "system"
      );
      reconnectTimerRef.current = setTimeout(() => {
        addConsoleEntry("outgoing", "Reconnecting to ADK server...", { userId, sessionId }, "🔄", "system");
        connect();
      }, 5000);
    };

    ws.onerror = () => {
      setConnectionStatus("disconnected");
      addConsoleEntry("error", "WebSocket Error", { message: "Connection error occurred" }, "⚠️", "system");
    };
  }, [buildWsUrl, addSystemMessage, addConsoleEntry, handleAdkEvent]);

  // ── Send text message (ALL text goes to agent, including URLs) ──
  const sendTextMessage = useCallback(
    (message: string) => {
      addUserMessage(message);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "text", text: message }));
        addConsoleEntry("outgoing", "User Message: " + message, null, "💬", "user");
      }
    },
    [addUserMessage, addConsoleEntry]
  );

  // ── Send image (camera capture) → agent via main WebSocket ──
  // The before_model_callback in agent.py intercepts the inline_data and
  // rewrites it as a directive. Agent then decides whether to call a tool.
  const sendImage = useCallback(
    (base64Data: string, imageDataUrl?: string, prompt?: string) => {
      const dataUrl = imageDataUrl || `data:image/jpeg;base64,${base64Data}`;
      addUserImageMessage(dataUrl);

      addConsoleEntry(
        "outgoing",
        "Image Sent → Agent (agent will decide analysis)",
        { mimeType: "image/jpeg", b64_len: base64Data.length },
        "📷",
        "user"
      );

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "image",
            data: base64Data,
            mimeType: "image/jpeg",
            prompt: prompt || "",
          })
        );
      }
    },
    [addUserImageMessage, addConsoleEntry]
  );

  // ── Send image upload (File object) → agent via main WebSocket ──
  // Reads the file, converts to base64, sends as { type: "image" } JSON.
  // Agent's before_model_callback intercepts inline_data and decides.
  const sendImageUpload = useCallback(
    (file: File, prompt?: string) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        const base64 = result.split(",")[1];
        const mimeType = result.split(";")[0].replace("data:", "") || "image/jpeg";
        const dataUrl = result;

        addUserImageMessage(dataUrl);

        addConsoleEntry(
          "outgoing",
          `File Upload → Agent (${file.name})`,
          { name: file.name, size: file.size, mime: mimeType },
          "📎",
          "user"
        );

        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(
            JSON.stringify({
              type: "image",
              data: base64,
              mimeType,
              prompt: prompt || "",
            })
          );
        }
      };
      reader.readAsDataURL(file);
    },
    [addUserImageMessage, addConsoleEntry]
  );

  // ── Send audio chunk (binary) ──
  const sendAudioChunk = useCallback((pcmData: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(pcmData);
    }
  }, []);

  // ── Clear console ──
  const clearConsole = useCallback(() => {
    setConsoleEntries([]);
  }, []);

  // ── Effect: connect on mount & reconnect on config change ──
  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      analyzeWsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enableProactivity, enableAffectiveDialog]);

  return {
    messages,
    consoleEntries,
    connectionStatus,
    sendTextMessage,
    sendImage,
    sendImageUpload,
    sendAudioChunk,
    clearConsole,
  };
}
