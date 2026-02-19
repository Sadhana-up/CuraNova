let mode = 'url';
let ws = null;
let startTime = 0;
let tokenCount = 0;

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
    
    // Toggle Buttons
    document.getElementById('btn-url').className = mode === 'url' ? 'active' : '';
    document.getElementById('btn-up').className = mode === 'upload' ? 'active' : '';

    // Toggle Inputs
    if (mode === 'url') {
        document.getElementById('url-input-wrapper').classList.remove('hidden');
        document.getElementById('file-input-wrapper').classList.add('hidden');
        updatePreviewFromUrl();
    } else {
        document.getElementById('url-input-wrapper').classList.add('hidden');
        document.getElementById('file-input-wrapper').classList.remove('hidden');
        updatePreviewFromFile();
    }
}

function updatePreviewFromUrl() {
    const url = document.getElementById('imgUrl').value;
    const img = document.getElementById('preview');
    
    if(url && mode === 'url') { 
        img.src = url; 
        img.onload = () => img.classList.add('visible');
        img.onerror = () => img.classList.remove('visible');
        if(img.complete) img.classList.add('visible');
    } else {
        img.classList.remove('visible');
    }
}

function updatePreviewFromFile() {
    const file = document.getElementById('imgFile').files[0];
    const img = document.getElementById('preview');
    const nameDisplay = document.getElementById('fileNameDisplay');

    if (file) {
        nameDisplay.innerText = file.name;
        const reader = new FileReader();
        reader.onload = (e) => { 
            img.src = e.target.result; 
            img.classList.add('visible');
        };
        reader.readAsDataURL(file);
    } else {
        nameDisplay.innerText = "NO_DATA";
        img.classList.remove('visible');
    }
}

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
    
    if (!ngrok) return alert(">> ERROR: MISSING_UPLINK_URL");

    if (ws) ws.close();
    output.innerHTML = "";
    outputContainer.classList.remove('hidden');
    
    btn.disabled = true;
    btn.querySelector('.btn-content').innerText = "PROCESSING...";
    
    tokenCount = 0;
    startTime = performance.now();
    let fullText = "";

    let payload = { prompt: prompt, max_new_tokens: 500 };

    try {
        if (mode === 'url') {
            payload.image_url = document.getElementById('imgUrl').value;
            var endpoint = "/ws/analyze/image-url";
        } else {
            const file = document.getElementById('imgFile').files[0];
            if (!file) throw new Error("ERROR: NO_SOURCE_FILE");
            payload.image_b64 = await getBase64(file);
            var endpoint = "/ws/analyze/image-base64";
        }

        const wsUrl = ngrok.replace("https://", "wss://").replace("http://", "ws://") + endpoint;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => ws.send(JSON.stringify(payload));

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.token) {
                tokenCount++;
                fullText += data.token;
                output.innerHTML = marked.parse(fullText);
                updateStats();
                
                // Smooth Scroll to bottom
                const log = document.getElementById('terminal-log');
                log.scrollTo({ top: log.scrollHeight, behavior: 'smooth' });

            } else if (data.status === 'done') {
                ws.close();
                resetBtn();
            } else if (data.error) {
                output.innerHTML += `<br><span style="color:#ff4444">>> ERROR: ${data.error}</span>`;
                ws.close();
                resetBtn();
            }
        };

        ws.onerror = (e) => {
            output.innerHTML = "<span style='color:#ff4444'>xx CONNECTION_FAILURE xx</span>";
            resetBtn();
        };

    } catch (err) {
        alert(err.message);
        resetBtn();
    }
}

function resetBtn() {
    const btn = document.getElementById('runBtn');
    btn.disabled = false;
    btn.querySelector('.btn-content').innerText = "INITIALIZE_ANALYSIS";
}

function updateStats() {
    const elapsed = (performance.now() - startTime) / 1000;
    const tps = elapsed > 0 ? tokenCount / elapsed : 0;
    document.getElementById('timeVal').innerText = (elapsed * 1000).toFixed(0) + "ms";
    document.getElementById('tpsDisplay').innerText = "SPEED: " + tps.toFixed(1) + " T/S";
}