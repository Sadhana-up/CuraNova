let mode = 'url';
let ws = null;
let startTime = 0;
let tokenCount = 0;

// Initialize
window.onload = () => {
    updateClock();
    setInterval(updateClock, 1000);
    updatePreviewFromUrl();
};

function updateClock() {
    const now = new Date();
    document.getElementById('clock').innerText = now.toISOString().split('T')[1].split('.')[0] + " UTC";
}

function setMode(newMode) {
    mode = newMode;
    
    // Update active button state
    const buttons = document.querySelectorAll('.terminal-btn');
    buttons.forEach(b => b.classList.remove('active'));
    // Find button with correct onclick handler text part (simple check)
    Array.from(buttons).find(b => b.getAttribute('onclick').includes(`'${newMode}'`)).classList.add('active');

    if (mode === 'url') {
        document.getElementById('url-input-group').classList.remove('hidden');
        document.getElementById('file-input-group').classList.add('hidden');
        updatePreviewFromUrl();
    } else {
        document.getElementById('url-input-group').classList.add('hidden');
        document.getElementById('file-input-group').classList.remove('hidden');
        updatePreviewFromFile();
    }
}

// Image Previews
function updatePreviewFromUrl() {
    const url = document.getElementById('imgUrl').value;
    const img = document.getElementById('preview');
    const container = document.getElementById('previewContainer');
    
    if(url) { 
        img.src = url; 
        container.classList.remove('hidden');
    } else {
        container.classList.add('hidden');
    }
}

function updatePreviewFromFile() {
    const fileInput = document.getElementById('imgFile');
    const file = fileInput.files[0];
    const img = document.getElementById('preview');
    const container = document.getElementById('previewContainer');
    const nameDisplay = document.getElementById('fileNameDisplay');

    if (file) {
        nameDisplay.innerText = file.name.toUpperCase();
        const reader = new FileReader();
        reader.onload = (e) => { 
            img.src = e.target.result; 
            container.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        nameDisplay.innerText = "NO_FILE_CHOSEN";
        container.classList.add('hidden');
    }
}

// Helper: Get Base64 without prefix
const getBase64 = (file) => new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(file);
});

async function runInference() {
    const ngrok = document.getElementById('ngrokUrl').value.replace(/\/$/, "");
    const prompt = document.getElementById('prompt').value;
    const output = document.getElementById('output');
    const outputContainer = document.getElementById('output-container');
    const btn = document.getElementById('runBtn');
    const statsRow = document.getElementById('statsRow');

    if (!ngrok) return alert("SYSTEM_ERROR: NGROK_URL_MISSING");

    // Reset State
    if (ws) ws.close();
    output.innerHTML = "";
    outputContainer.classList.remove('hidden');
    statsRow.classList.remove('hidden');
    
    btn.disabled = true;
    btn.innerText = "ACCESSING_MAINFRAME...";
    
    tokenCount = 0;
    startTime = performance.now();
    let fullText = "";

    // Prepare Payload
    let endpoint = "";
    let payload = { prompt: prompt, max_new_tokens: 500 };

    try {
        if (mode === 'url') {
            endpoint = "/ws/analyze/image-url";
            payload.image_url = document.getElementById('imgUrl').value;
        } else {
            endpoint = "/ws/analyze/image-base64";
            const file = document.getElementById('imgFile').files[0];
            if (!file) throw new Error("SYSTEM_ERROR: INPUT_FILE_MISSING");
            payload.image_b64 = await getBase64(file);
        }

        // Connect
        const wsUrl = ngrok.replace("https://", "wss://").replace("http://", "ws://") + endpoint;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("Connected");
            btn.innerText = "TRANSMITTING_DATA...";
            ws.send(JSON.stringify(payload));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.token) {
                tokenCount++;
                fullText += data.token;
                // Using marked for markdown, but styled for terminal
                output.innerHTML = marked.parse(fullText);
                updateStats();
                
                // Auto-scroll to bottom of terminal log
                const log = document.getElementById('terminal-log');
                log.scrollTop = log.scrollHeight;
            } else if (data.status === 'done') {
                ws.close();
                btn.disabled = false;
                btn.innerText = ">> EXECUTE_ANALYSIS <<";
                output.innerHTML += "<br><br><span style='color: white;'>> TRANSMISSION_COMPLETE.</span>";
            } else if (data.error) {
                output.innerHTML += `<br><span style="color:red">>> ERROR: ${data.error}</span>`;
                ws.close();
                btn.disabled = false;
                btn.innerText = "RETRY_ANALYSIS";
            }
        };

        ws.onerror = (e) => {
            console.error(e);
            output.innerHTML = "<span style='color:red'>xx CONNECTION_FAILURE. CHECK_NGROK_URL. xx</span>";
            btn.disabled = false;
            btn.innerText = "RETRY_ANALYSIS";
        };

        ws.onclose = () => {
             if(btn.disabled) {
                btn.disabled = false;
                btn.innerText = ">> EXECUTE_ANALYSIS <<";
             }
        }

    } catch (err) {
        alert(err.message);
        btn.disabled = false;
        btn.innerText = ">> EXECUTE_ANALYSIS <<";
    }
}

function updateStats() {
    const elapsed = (performance.now() - startTime) / 1000;
    const tps = elapsed > 0 ? tokenCount / elapsed : 0;
    
    document.getElementById('timeVal').innerText = elapsed.toFixed(1) + "s";
    document.getElementById('tokenVal').innerText = tokenCount;
    document.getElementById('tpsVal').innerText = tps.toFixed(1) + " t/s";
}
