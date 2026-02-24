"use client";
// ──────────────────────────────────────────────────────────
// useAudio — manages Web Audio API worklets for recording
// and playback of PCM audio, matching the original exactly.
// ──────────────────────────────────────────────────────────

import { useCallback, useRef, useState } from "react";

interface UseAudioOptions {
  /** Called when a recorded PCM chunk is ready to send to the server */
  onRecordedChunk: (pcmData: ArrayBuffer) => void;
}

interface UseAudioReturn {
  isAudioActive: boolean;
  startAudio: () => Promise<void>;
  /** Feed decoded PCM ArrayBuffer from the server into the player */
  playAudioData: (buffer: ArrayBuffer) => void;
  /** Signal end-of-audio to the player (e.g. on interrupt) */
  stopPlayback: () => void;
}

export function useAudio({ onRecordedChunk }: UseAudioOptions): UseAudioReturn {
  const [isAudioActive, setIsAudioActive] = useState(false);

  const playerNodeRef = useRef<AudioWorkletNode | null>(null);
  const playerCtxRef = useRef<AudioContext | null>(null);
  const recorderNodeRef = useRef<AudioWorkletNode | null>(null);
  const recorderCtxRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);

  // Keep a stable ref for the callback
  const onRecordedChunkRef = useRef(onRecordedChunk);
  onRecordedChunkRef.current = onRecordedChunk;

  /** Convert Float32 samples to 16-bit PCM Int16Array buffer */
  const convertFloat32ToPCM = useCallback(
    (inputData: Float32Array): ArrayBuffer => {
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        pcm16[i] = inputData[i] * 0x7fff;
      }
      return pcm16.buffer;
    },
    []
  );

  const startAudio = useCallback(async () => {
    try {
      // ── Player (24 kHz output) ──
      const playerCtx = new AudioContext({ sampleRate: 24000 });
      await playerCtx.audioWorklet.addModule("/audio/pcm-player-processor.js");
      const playerNode = new AudioWorkletNode(
        playerCtx,
        "pcm-player-processor"
      );
      playerNode.connect(playerCtx.destination);
      playerNodeRef.current = playerNode;
      playerCtxRef.current = playerCtx;

      // ── Recorder (16 kHz input) ──
      const recorderCtx = new AudioContext({ sampleRate: 16000 });
      await recorderCtx.audioWorklet.addModule(
        "/audio/pcm-recorder-processor.js"
      );
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1 },
      });
      const source = recorderCtx.createMediaStreamSource(micStream);
      const recorderNode = new AudioWorkletNode(
        recorderCtx,
        "pcm-recorder-processor"
      );
      source.connect(recorderNode);

      recorderNode.port.onmessage = (event: MessageEvent) => {
        const pcmData = convertFloat32ToPCM(event.data as Float32Array);
        onRecordedChunkRef.current(pcmData);
      };

      recorderNodeRef.current = recorderNode;
      recorderCtxRef.current = recorderCtx;
      micStreamRef.current = micStream;

      setIsAudioActive(true);
    } catch (err) {
      console.error("Failed to start audio:", err);
    }
  }, [convertFloat32ToPCM]);

  const playAudioData = useCallback((buffer: ArrayBuffer) => {
    playerNodeRef.current?.port.postMessage(new Int16Array(buffer));
  }, []);

  const stopPlayback = useCallback(() => {
    playerNodeRef.current?.port.postMessage({ command: "endOfAudio" });
  }, []);

  return { isAudioActive, startAudio, playAudioData, stopPlayback };
}
