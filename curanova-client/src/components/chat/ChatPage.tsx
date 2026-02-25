"use client";

import {
  FormEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Mic,
  MicOff,
  Camera,
  Paperclip,
  X,
  Wifi,
  WifiOff,
  Loader2,
  Stethoscope,
  Sparkles,
  Settings2,
  Bot,
  User,
  ArrowDown,
} from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAudio } from "@/hooks/useAudio";
import CameraModal from "@/components/chat/CameraModal";
import type { ChatMessage } from "@/types/chat";

const SERVER_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ── Typing dots animation ── */
function TypingDots() {
  return (
    <span className="inline-flex items-center gap-[3px] ml-1.5 align-middle">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="inline-block w-[5px] h-[5px] rounded-full bg-white/40"
          animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            delay: i * 0.15,
            ease: "easeInOut",
          }}
        />
      ))}
    </span>
  );
}

/* ── Empty state ── */
function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="flex flex-col items-center justify-center h-full select-none"
    >
      <motion.div
        animate={{ rotate: [0, 5, -5, 0] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        className="w-20 h-20 rounded-3xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-white/[0.06] flex items-center justify-center mb-6 backdrop-blur-sm"
      >
        <Sparkles className="w-9 h-9 text-violet-400/70" />
      </motion.div>
      <h2 className="text-xl font-semibold text-white/80 mb-2 tracking-tight">
        CuraNova
      </h2>
      <p className="text-sm text-white/30 max-w-xs text-center leading-relaxed">
        Medical AI assistant — ask anything, share images, or use voice
      </p>
    </motion.div>
  );
}

/* ── Single message bubble ── */
function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.type === "system") {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex justify-center py-1.5"
      >
        <span className="text-[11px] text-white/25 bg-white/[0.03] px-3 py-1 rounded-full">
          {msg.text}
        </span>
      </motion.div>
    );
  }

  if (msg.type === "image") {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-end gap-2"
      >
        <div className="max-w-[280px] rounded-2xl overflow-hidden ring-1 ring-white/[0.08]">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={msg.imageDataUrl}
            alt="Captured"
            className="w-full h-auto object-cover"
          />
        </div>
        {msg.text && (
          <div className="max-w-[75%] px-4 py-2.5 rounded-2xl bg-violet-500/90 text-white rounded-tr-md">
            <p className="text-[13.5px] leading-relaxed whitespace-pre-wrap break-words">
              {msg.text}
            </p>
          </div>
        )}
      </motion.div>
    );
  }

  if (msg.type === "tool-stream") {
    return (
      <motion.div
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        className="flex gap-2.5 items-start max-w-[85%]"
      >
        <div className="shrink-0 w-7 h-7 rounded-lg bg-emerald-500/15 border border-emerald-500/20 flex items-center justify-center mt-0.5">
          <Stethoscope className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div className="bg-emerald-500/[0.08] border border-emerald-500/[0.12] rounded-2xl rounded-tl-md px-4 py-3 min-w-0">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-emerald-400/70 mb-1.5 block">
            Medical Analysis
          </span>
          <p className="text-[13.5px] leading-relaxed text-white/85 whitespace-pre-wrap break-words">
            {msg.text}
            {msg.isPartial && <TypingDots />}
          </p>
        </div>
      </motion.div>
    );
  }

  const isUser = msg.type === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8, x: isUser ? 8 : -8 }}
      animate={{ opacity: 1, y: 0, x: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={`flex items-start gap-2.5 ${isUser ? "flex-row-reverse" : ""} ${msg.isInterrupted ? "opacity-50" : ""}`}
    >
      {/* Avatar */}
      <div
        className={`shrink-0 w-7 h-7 rounded-lg flex items-center justify-center mt-0.5 ${isUser
          ? "bg-violet-500/15 border border-violet-500/20"
          : "bg-white/[0.06] border border-white/[0.08]"
          }`}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5 text-violet-400" />
        ) : (
          <Bot className="w-3.5 h-3.5 text-white/50" />
        )}
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[75%] min-w-0 px-4 py-2.5 rounded-2xl ${isUser
          ? "bg-violet-500/90 text-white rounded-tr-md"
          : "bg-white/[0.06] border border-white/[0.06] text-white/85 rounded-tl-md"
          } ${msg.isTranscription ? "ring-1 ring-white/10" : ""}`}
      >
        {msg.isTranscription && isUser && (
          <Mic className="inline w-3 h-3 mr-1 opacity-50 -mt-0.5" />
        )}
        <p className="text-[13.5px] leading-relaxed whitespace-pre-wrap break-words">
          {msg.text}
          {msg.isPartial && !isUser && <TypingDots />}
        </p>
        {msg.isInterrupted && (
          <span className="text-[10px] italic text-white/30 mt-1 block">
            interrupted
          </span>
        )}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   ChatPage — Main Component
   ═══════════════════════════════════════════════════════ */
export default function ChatPage() {
  const [enableProactivity, setEnableProactivity] = useState(false);
  const [enableAffectiveDialog, setEnableAffectiveDialog] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const {
    isAudioActive,
    startAudio,
    playAudioData,
  } = useAudio({
    onRecordedChunk: (pcmData) => sendAudioChunk(pcmData),
  });

  const {
    messages,
    connectionStatus,
    sendTextMessage,
    sendImage,
    sendImageUpload,
    sendAudioChunk,
  } = useWebSocket({
    serverUrl: SERVER_URL,
    enableProactivity,
    enableAffectiveDialog,
    showAudioEvents: false,
    onAudioData: isAudioActive ? playAudioData : undefined,
  });

  const [inputValue, setInputValue] = useState("");
  // Staged attachments — files are held here until user hits Send
  const [pendingFiles, setPendingFiles] = useState<
    { file: File; previewUrl: string }[]
  >([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Detect if user scrolled up
  const handleScroll = () => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    setShowScrollBtn(!atBottom);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const text = inputValue.trim();

    // If there are staged image attachments, send them WITH the prompt so
    // the agent receives both the image and the clinical question together.
    if (pendingFiles.length > 0) {
      const prompt = text || undefined;
      // Send all pending images — agent's before_model_callback will group
      // them into a single turn and decide whether to do a tool call.
      pendingFiles.forEach(({ file }) => sendImageUpload(file, prompt));
      setPendingFiles([]);
      setInputValue("");
      if (fileInputRef.current) fileInputRef.current.value = "";
      inputRef.current?.focus();
      return;
    }

    // Plain text message (including image URLs — agent decides)
    if (!text) return;
    sendTextMessage(text);
    setInputValue("");
    inputRef.current?.focus();
  };

  const handleCameraCapture = useCallback(
    (base64Data: string, imageDataUrl: string) => {
      const prompt = inputValue.trim() || undefined;
      setInputValue("");
      sendImage(base64Data, imageDataUrl, prompt);
    },
    [sendImage, inputValue]
  );

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []).filter((f) =>
        f.type.startsWith("image/")
      );
      if (!files.length) return;

      // Stage files (max 2 total across all batches)
      setPendingFiles((prev) => {
        const remaining = 2 - prev.length;
        if (remaining <= 0) return prev;
        const toAdd = files.slice(0, remaining).map((file) => ({
          file,
          previewUrl: URL.createObjectURL(file),
        }));
        return [...prev, ...toAdd];
      });

      // Reset file input so the same file can trigger onChange again
      if (fileInputRef.current) fileInputRef.current.value = "";
      // Focus input so user can type their prompt
      inputRef.current?.focus();
    },
    [] // no deps — only uses refs and setState
  );

  const isConnected = connectionStatus === "connected";
  const hasMessages =
    messages.filter((m) => m.type !== "system").length > 0;

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0f] text-white overflow-hidden">
      {/* ── Header ── */}
      <header className="relative z-10 flex items-center justify-between px-5 py-3 border-b border-white/[0.06] bg-[#0a0a0f]/80 backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight">CuraNova</h1>
            <p className="text-[10px] text-white/30 -mt-0.5">Medical AI Assistant</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Connection */}
          <div className="flex items-center gap-1.5 text-[11px] text-white/40 mr-1">
            {isConnected ? (
              <Wifi className="w-3.5 h-3.5 text-emerald-400/70" />
            ) : connectionStatus === "connecting" ? (
              <Loader2 className="w-3.5 h-3.5 text-amber-400/70 animate-spin" />
            ) : (
              <WifiOff className="w-3.5 h-3.5 text-red-400/70" />
            )}
            <span className={isConnected ? "text-emerald-400/50" : connectionStatus === "connecting" ? "text-amber-400/50" : "text-red-400/50"}>
              {isConnected ? "Live" : connectionStatus === "connecting" ? "Connecting" : "Offline"}
            </span>
          </div>

          {/* Settings toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowSettings((s) => !s)}
            className={`p-2 rounded-xl transition-colors ${showSettings ? "bg-white/10" : "hover:bg-white/[0.04]"}`}
          >
            <Settings2 className="w-4 h-4 text-white/40" />
          </motion.button>
        </div>
      </header>

      {/* ── Settings dropdown ── */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden border-b border-white/[0.06] bg-white/[0.02]"
          >
            <div className="px-5 py-3 flex items-center gap-6 text-xs text-white/50">
              <label className="flex items-center gap-2 cursor-pointer select-none group">
                <input
                  type="checkbox"
                  checked={enableProactivity}
                  onChange={(e) => setEnableProactivity(e.target.checked)}
                  className="w-3.5 h-3.5 rounded accent-violet-500 cursor-pointer"
                />
                <span className="group-hover:text-white/70 transition-colors">Proactivity</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer select-none group">
                <input
                  type="checkbox"
                  checked={enableAffectiveDialog}
                  onChange={(e) => setEnableAffectiveDialog(e.target.checked)}
                  className="w-3.5 h-3.5 rounded accent-violet-500 cursor-pointer"
                />
                <span className="group-hover:text-white/70 transition-colors">Affective Dialog</span>
              </label>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Messages area ── */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto chat-scroll"
      >
        {!hasMessages ? (
          <EmptyState />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 flex flex-col gap-3">
            {/* Messages */}
            <AnimatePresence mode="popLayout">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}
            </AnimatePresence>

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Scroll-to-bottom FAB ── */}
      <AnimatePresence>
        {showScrollBtn && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            onClick={scrollToBottom}
            className="absolute bottom-24 left-1/2 -translate-x-1/2 z-20 w-9 h-9 rounded-full bg-white/10 border border-white/10 backdrop-blur-md flex items-center justify-center hover:bg-white/15 transition-colors"
          >
            <ArrowDown className="w-4 h-4 text-white/60" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* ── Input bar ── */}
      <div className="relative z-10 border-t border-white/[0.06] bg-[#0a0a0f]/90 backdrop-blur-xl">
        {/* Hidden file input for image uploads */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          id="image-upload-input"
          onChange={handleFileUpload}
        />

        {/* Staged image previews */}
        <AnimatePresence>
          {pendingFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="max-w-3xl mx-auto px-4 pt-2 flex gap-2 overflow-hidden"
            >
              {pendingFiles.map(({ previewUrl }, idx) => (
                <motion.div
                  key={previewUrl}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.85 }}
                  className="relative shrink-0"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt={`Attachment ${idx + 1}`}
                    className="w-14 h-14 rounded-xl object-cover ring-1 ring-white/10"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      URL.revokeObjectURL(previewUrl);
                      setPendingFiles((prev) =>
                        prev.filter((_, i) => i !== idx)
                      );
                    }}
                    className="absolute -top-1.5 -right-1.5 w-4.5 h-4.5 bg-black/80 border border-white/20 rounded-full flex items-center justify-center text-white/70 hover:text-white transition-colors"
                    title="Remove"
                  >
                    <X className="w-2.5 h-2.5" />
                  </button>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
        <form
          onSubmit={handleSubmit}
          className="max-w-3xl mx-auto px-4 py-3 flex items-center gap-2"
        >
          {/* Camera */}
          <CameraModal onCapture={handleCameraCapture} />

          {/* File upload */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => fileInputRef.current?.click()}
            className="p-2.5 rounded-xl hover:bg-white/[0.06] text-white/35 hover:text-white/60 transition-all"
            title="Upload image (max 2)"
          >
            <Paperclip className="w-[18px] h-[18px]" />
          </motion.button>

          {/* Audio */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            disabled={isAudioActive}
            onClick={() => startAudio()}
            className={`p-2.5 rounded-xl transition-all ${isAudioActive
              ? "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20"
              : "hover:bg-white/[0.06] text-white/35 hover:text-white/60"
              }`}
            title={isAudioActive ? "Audio active" : "Start audio"}
          >
            {isAudioActive ? (
              <Mic className="w-[18px] h-[18px]" />
            ) : (
              <MicOff className="w-[18px] h-[18px]" />
            )}
          </motion.button>

          {/* Text input */}
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Message CuraNova..."
              autoComplete="off"
              className="w-full bg-white/[0.04] border border-white/[0.07] rounded-xl px-4 py-2.5 text-[13.5px] text-white placeholder-white/20 outline-none focus:border-violet-500/40 focus:bg-white/[0.06] transition-all"
            />
          </div>

          {/* Send */}
          <motion.button
            type="submit"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.9 }}
            disabled={!isConnected || (!inputValue.trim() && pendingFiles.length === 0)}
            className="p-2.5 rounded-xl bg-violet-500/90 text-white disabled:opacity-20 disabled:cursor-not-allowed hover:bg-violet-500 transition-all"
          >
            <Send className="w-[18px] h-[18px]" />
          </motion.button>
        </form>
      </div>
    </div>
  );
}
