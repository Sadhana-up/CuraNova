"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, X, Send } from "lucide-react";

interface CameraModalProps {
  onCapture: (base64Data: string, imageDataUrl: string) => void;
}

export default function CameraModal({ onCapture }: CameraModalProps) {
  const [isOpen, setIsOpen] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const open = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 768 },
          height: { ideal: 768 },
          facingMode: "user",
        },
      });
      streamRef.current = stream;
      setIsOpen(true);
    } catch (err) {
      console.error("Camera access failed:", err);
    }
  }, []);

  useEffect(() => {
    if (isOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [isOpen]);

  const close = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsOpen(false);
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current || !streamRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.85);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(",")[1];
          onCapture(base64, imageDataUrl);
        };
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.85
    );

    close();
  }, [onCapture, close]);

  return (
    <>
      {/* Trigger */}
      <motion.button
        type="button"
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={open}
        className="p-2.5 rounded-xl hover:bg-white/[0.06] text-white/35 hover:text-white/60 transition-all"
        title="Open camera"
      >
        <Camera className="w-[18px] h-[18px]" />
      </motion.button>

      {/* Modal — portaled to body so backdrop-blur parents don't break fixed positioning */}
      {typeof document !== "undefined" &&
        createPortal(
          <AnimatePresence>
            {isOpen && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
                onClick={(e) => {
                  if (e.target === e.currentTarget) close();
                }}
              >
                <motion.div
                  initial={{ scale: 0.92, opacity: 0, y: 10 }}
                  animate={{ scale: 1, opacity: 1, y: 0 }}
                  exit={{ scale: 0.92, opacity: 0, y: 10 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="bg-[#141418] border border-white/[0.08] rounded-2xl shadow-2xl w-full max-w-lg mx-4 overflow-hidden"
                >
                  {/* Header */}
                  <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06]">
                    <h3 className="text-sm font-semibold text-white/80">
                      Camera
                    </h3>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={close}
                      className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-white/[0.08] transition-colors"
                    >
                      <X className="w-4 h-4 text-white/50" />
                    </motion.button>
                  </div>

                  {/* Preview */}
                  <div className="bg-black flex items-center justify-center min-h-[320px]">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full max-h-[420px] object-contain"
                    />
                  </div>

                  {/* Footer */}
                  <div className="flex items-center justify-end gap-2 px-5 py-3.5 border-t border-white/[0.06]">
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={close}
                      className="px-4 py-2 rounded-xl text-sm text-white/50 hover:bg-white/[0.06] transition-colors"
                    >
                      Cancel
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={capture}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm bg-violet-500/90 text-white hover:bg-violet-500 transition-colors"
                    >
                      <Send className="w-3.5 h-3.5" />
                      Capture & Send
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>,
          document.body
        )}
    </>
  );
}
