// ──────────────────────────────────────────────────────────
// Utility helpers for WebSocket and message handling
// ──────────────────────────────────────────────────────────

/** Generate a random short ID */
export function randomId(): string {
  return Math.random().toString(36).substring(7);
}

/** Format current time as HH:MM:SS.mmm */
export function formatTimestamp(): string {
  const now = new Date();
  return now.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
  });
}

/**
 * Detect an image URL in a message string.
 * Returns the URL if found, null otherwise.
 */
export function extractImageUrl(text: string): string | null {
  // Match common image URL patterns
  const urlRegex =
    /(https?:\/\/[^\s]+\.(?:png|jpg|jpeg|gif|bmp|webp|tif|tiff|svg))/i;
  const match = text.match(urlRegex);
  if (match) return match[1];

  // Also match any https URL (user may paste a URL without extension)
  const genericUrl = /(https?:\/\/[^\s]+)/i;
  const gMatch = text.match(genericUrl);
  if (gMatch) return gMatch[1];

  return null;
}

/**
 * Clean spaces between CJK characters.
 * Removes spaces between Japanese/Chinese/Korean characters while preserving spaces around Latin text.
 */
export function cleanCJKSpaces(text: string): string {
  const cjkPattern =
    /[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]/;

  return text.replace(/(\S)\s+(?=\S)/g, (match, char1: string) => {
    const nextCharMatch = text.match(new RegExp(char1 + "\\s+(.)", "g"));
    if (nextCharMatch && nextCharMatch.length > 0) {
      const char2 = nextCharMatch[0].slice(-1);
      if (cjkPattern.test(char1) && cjkPattern.test(char2)) {
        return char1;
      }
    }
    return match;
  });
}

/**
 * Sanitize event data for console display
 * (replace large audio data with summary)
 */
export function sanitizeEventForDisplay(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  event: Record<string, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Record<string, any> {
  const sanitized = JSON.parse(JSON.stringify(event));

  if (sanitized.content && sanitized.content.parts) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    sanitized.content.parts = sanitized.content.parts.map((part: any) => {
      if (part.inlineData && part.inlineData.data) {
        const byteSize = Math.floor(part.inlineData.data.length * 0.75);
        return {
          ...part,
          inlineData: {
            ...part.inlineData,
            data: `(${byteSize.toLocaleString()} bytes)`,
          },
        };
      }
      return part;
    });
  }

  return sanitized;
}

/**
 * Decode Base64 data to ArrayBuffer.
 * Handles both standard base64 and base64url encoding.
 */
export function base64ToArray(base64: string): ArrayBuffer {
  let standardBase64 = base64.replace(/-/g, "+").replace(/_/g, "/");
  while (standardBase64.length % 4) {
    standardBase64 += "=";
  }
  const binaryString = window.atob(standardBase64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}
