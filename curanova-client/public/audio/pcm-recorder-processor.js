class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Buffer size of 4096 samples = ~256ms of audio at 16kHz
    // This reduces WebSocket traffic and prevents API rate limits/errors
    this.bufferSize = 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    if (inputs.length > 0 && inputs[0].length > 0) {
      // Use the first channel
      const inputChannel = inputs[0][0];

      // Accumulate samples in the buffer
      for (let i = 0; i < inputChannel.length; i++) {
        this.buffer[this.bufferIndex++] = inputChannel[i];

        // When buffer is full, send it and reset
        if (this.bufferIndex >= this.bufferSize) {
          const bufferCopy = new Float32Array(this.buffer);
          this.port.postMessage(bufferCopy);
          this.bufferIndex = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-recorder-processor", PCMProcessor);
