import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# ─────────────────────────────────────────────────────────────
# HTML/JS/CSS Application
# ─────────────────────────────────────────────────────────────
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CuraNova Stream Client</title>
    <!-- Markdown Parser -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Tailwind CSS for styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; }
        .input-dark { background-color: #1e293b; border: 1px solid #334155; color: white; }
        .input-dark:focus { outline: none; border-color: #38bdf8; }
        .btn-primary { background-color: #0ea5e9; color: white; transition: 0.2s; }
        .btn-primary:hover { background-color: #0284c7; }
        .btn-disabled { background-color: #475569; cursor: not-allowed; }
        
        /* Markdown Styles */
        .prose h1, .prose h2, .prose h3 { color: #38bdf8; margin-top: 1em; margin-bottom: 0.5em; font-weight: bold; }
        .prose ul { list-style-type: disc; padding-left: 1.5em; margin-bottom: 1em; }
        .prose p { margin-bottom: 1em; line-height: 1.6; }
        .prose strong { color: #f0f9ff; font-weight: 700; }
        
        /* Blink cursor effect for streaming */
        .cursor::after { content: '▋'; animation: blink 1s step-start infinite; color: #38bdf8; }
        @keyframes blink { 50% { opacity: 0; } }
    </style>
</head>
<body class="flex flex-col items-center min-h-screen p-6">

    <div class="w-full max-w-4xl bg-slate-800 p-8 rounded-xl shadow-2xl border border-slate-700">
        
        <!-- Header -->
        <div class="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
            <h1 class="text-3xl font-bold text-sky-400">CuraNova <span class="text-slate-400 text-lg">Live Stream</span></h1>
            <div class="text-xs text-slate-500">WebSocket Client</div>
        </div>

        <!-- Configuration -->
        <div class="grid grid-cols-1 gap-4 mb-6">
            <div>
                <label class="block text-sm font-medium text-slate-400 mb-1">Colab Ngrok URL</label>
                <input type="text" id="ngrokUrl" 
                    class="w-full p-2 rounded input-dark" 
                    placeholder="https://xxxx-xx-xx-xx-xx.ngrok-free.app"
                    value="https://a161-136-118-65-76.ngrok-free.app"> 
                <p class="text-xs text-slate-500 mt-1">Copy the public URL from your Colab logs here.</p>
            </div>
        </div>

        <!-- Input Type Switcher -->
        <div class="flex gap-4 mb-4">
            <label class="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="inputType" value="url" checked onchange="toggleInput()" class="accent-sky-500">
                <span>Image URL</span>
            </label>
            <label class="flex items-center gap-2 cursor-pointer">
                <input type="radio" name="inputType" value="upload" onchange="toggleInput()" class="accent-sky-500">
                <span>Upload File</span>
            </label>
        </div>

        <!-- Image Input Section -->
        <div class="mb-4">
            <!-- URL Input -->
            <div id="urlInputContainer">
                <label class="block text-sm font-medium text-slate-400 mb-1">Image URL</label>
                <input type="text" id="imageUrl" 
                    class="w-full p-2 rounded input-dark"
                    value="https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png">
            </div>

            <!-- File Upload -->
            <div id="fileInputContainer" class="hidden">
                <label class="block text-sm font-medium text-slate-400 mb-1">Upload X-Ray</label>
                <input type="file" id="imageFile" accept="image/*" 
                    class="w-full p-2 rounded input-dark file:mr-4 file:py-1 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-sky-500 file:text-white hover:file:bg-sky-600">
            </div>
        </div>

        <!-- Prompt Section -->
        <div class="mb-6">
            <label class="block text-sm font-medium text-slate-400 mb-1">Prompt</label>
            <textarea id="prompt" rows="2" class="w-full p-2 rounded input-dark">Describe this chest X-ray. What do you see?</textarea>
        </div>

        <!-- Action Button -->
        <button id="analyzeBtn" onclick="startAnalysis()" class="w-full py-3 rounded-lg font-bold text-lg btn-primary shadow-lg shadow-sky-500/20">
            Analyze Image
        </button>

        <!-- Status & Output -->
        <div class="mt-8">
            <div id="status" class="text-sm text-slate-400 mb-2 font-mono h-6">Ready.</div>
            <div id="outputContainer" class="bg-slate-900 p-6 rounded-lg border border-slate-700 min-h-[200px] shadow-inner">
                <div id="outputContent" class="prose prose-invert max-w-none text-gray-300"></div>
                <span id="cursor" class="hidden cursor"></span>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let accumulatedText = "";

        function toggleInput() {
            const type = document.querySelector('input[name="inputType"]:checked').value;
            if (type === 'url') {
                document.getElementById('urlInputContainer').classList.remove('hidden');
                document.getElementById('fileInputContainer').classList.add('hidden');
            } else {
                document.getElementById('urlInputContainer').classList.add('hidden');
                document.getElementById('fileInputContainer').classList.remove('hidden');
            }
        }

        function setStatus(msg, color='text-slate-400') {
            const el = document.getElementById('status');
            el.className = `text-sm mb-2 font-mono h-6 ${color}`;
            el.textContent = msg;
        }

        async function startAnalysis() {
            const btn = document.getElementById('analyzeBtn');
            const outputDiv = document.getElementById('outputContent');
            const cursor = document.getElementById('cursor');
            const ngrokUrl = document.getElementById('ngrokUrl').value.replace(/\/$/, ""); // remove trailing slash
            
            if (!ngrokUrl) {
                alert("Please enter your Colab Ngrok URL");
                return;
            }

            // Reset UI
            btn.disabled = true;
            btn.classList.add('btn-disabled');
            accumulatedText = "";
            outputDiv.innerHTML = "";
            cursor.classList.remove('hidden');
            setStatus("⏳ Connecting...");

            // Determine Input Type
            const inputType = document.querySelector('input[name="inputType"]:checked').value;
            let wsEndpoint = "";
            let payload = {};

            try {
                if (inputType === 'url') {
                    wsEndpoint = "/ws/analyze/image-url";
                    payload = {
                        image_url: document.getElementById('imageUrl').value,
                        prompt: document.getElementById('prompt').value,
                        max_new_tokens: 500
                    };
                    connectWebSocket(ngrokUrl, wsEndpoint, payload);
                } else {
                    wsEndpoint = "/ws/analyze/image-base64";
                    const fileInput = document.getElementById('imageFile');
                    if (fileInput.files.length === 0) {
                        alert("Please select a file.");
                        resetUI();
                        return;
                    }
                    setStatus("⏳ Processing image...");
                    const base64 = await convertFileToBase64(fileInput.files[0]);
                    payload = {
                        image_b64: base64,
                        prompt: document.getElementById('prompt').value,
                        max_new_tokens: 500
                    };
                    connectWebSocket(ngrokUrl, wsEndpoint, payload);
                }
            } catch (err) {
                console.error(err);
                setStatus("❌ Error: " + err.message, "text-red-500");
                resetUI();
            }
        }

        function convertFileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    // Remove the "data:image/png;base64," prefix
                    const result = reader.result;
                    const b64 = result.split(',')[1];
                    resolve(b64);
                };
                reader.onerror = error => reject(error);
                reader.readAsDataURL(file);
            });
        }

        function connectWebSocket(baseUrl, endpoint, payload) {
            // Convert https -> wss
            const wsBase = baseUrl.replace("https://", "wss://").replace("http://", "ws://");
            const wsUrl = `${wsBase}${endpoint}`;

            console.log("Connecting to:", wsUrl);

            if (ws) ws.close();
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                setStatus("✓ Connected! Streaming...", "text-green-400");
                console.log("Sending payload:", payload);
                ws.send(JSON.stringify(payload));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.token) {
                    accumulatedText += data.token;
                    // Render Markdown live
                    document.getElementById('outputContent').innerHTML = marked.parse(accumulatedText);
                    // Auto-scroll
                    window.scrollTo(0, document.body.scrollHeight);
                } else if (data.status === 'done') {
                    setStatus("✓ Analysis Complete", "text-sky-400");
                    ws.close();
                    resetUI();
                } else if (data.error) {
                    setStatus("❌ Server Error: " + data.error, "text-red-500");
                    ws.close();
                    resetUI();
                }
            };

            ws.onerror = (error) => {
                console.error("WebSocket Error:", error);
                setStatus("❌ WebSocket Error (Check console)", "text-red-500");
                resetUI();
            };

            ws.onclose = () => {
                console.log("Connection closed");
                if (document.getElementById('status').textContent.includes('Connecting')) {
                     setStatus("❌ Connection closed unexpectedly", "text-red-500");
                     resetUI();
                }
            };
        }

        function resetUI() {
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = false;
            btn.classList.remove('btn-disabled');
            document.getElementById('cursor').classList.add('hidden');
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_client():
    return html_content

if __name__ == "__main__":
    print("🚀 Starting Web Client...")
    print("👉 Open http://localhost:8080 in your browser")
    uvicorn.run(app, host="127.0.0.1", port=8080)