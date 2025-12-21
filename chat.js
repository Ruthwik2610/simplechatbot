(function checkAuth() {
    const user = localStorage.getItem('currentUser');
    if (!user) {
        window.location.href = 'index.html';
    }
})();

const chatHistory = [
    { id: 1, title: "React Component Help", date: "Today" },
    { id: 2, title: "Marketing Email Draft", date: "Today" }
];

let isGenerating = false;
let currentFile = null;
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB Limit

document.addEventListener('DOMContentLoaded', () => {
    renderHistory();
    setupEventListeners();
    setupFileHandling();
    autoResizeTextarea();
    if (window.hljs) hljs.highlightAll();
});

// --- File Handling Logic ---
function setupFileHandling() {
    const attachmentBtn = document.getElementById('attachmentBtn');
    const fileInput = document.getElementById('fileInput');

    attachmentBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.size > MAX_FILE_SIZE) {
            alert(`File is too large. Maximum size is 5MB.`);
            fileInput.value = '';
            return;
        }

        currentFile = file;
        renderFilePreview(file);
    });
}

function renderFilePreview(file) {
    const previewArea = document.getElementById('filePreviewArea');
    const fileSizeFormatted = (file.size / 1024).toFixed(1) + ' KB';
    
    previewArea.innerHTML = `
        <div class="file-chip">
            <i class="fas fa-file-alt"></i>
            <div class="file-info">
                <span class="file-name">${file.name}</span>
                <span class="file-size">${fileSizeFormatted}</span>
            </div>
            <i class="fas fa-times remove-file-btn" onclick="clearFile()"></i>
        </div>
    `;
}

window.clearFile = function() {
    currentFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('filePreviewArea').innerHTML = '';
}

// Helper to read file content as text
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        // FIX 1: Reject with a clear Error object, not the event object
        reader.onerror = () => reject(new Error("Failed to read file content")); 
        reader.readAsText(file);
    });
}

// --- Main Chat Logic ---
async function handleSend() {
    if (isGenerating) return;
    
    const input = document.getElementById('messageInput');
    const text = input.value.trim();
    
    if (!text && !currentFile) return;

    // UI Updates
    document.getElementById('emptyState').style.display = 'none';
    input.value = '';
    input.style.height = 'auto'; 
    
    // 1. Show User Message
    let userDisplayHtml = text;
    if (currentFile) {
        userDisplayHtml = `
            <div class="file-chip" style="margin-bottom: 10px; background:white;">
                <i class="fas fa-file-code" style="color:#DA7756"></i>
                <span style="font-size:13px; font-weight:500;">${currentFile.name}</span>
            </div>
            <br>${text}
        `;
    }
    addMessageToDom('user', userDisplayHtml);
    addToHistory(text || "File Analysis");

    // 2. Prepare Payload
    isGenerating = true;
    const loadingId = addLoadingIndicator();
    
    try {
        let fullContentForAI = text;

        if (currentFile) {
            try {
                const fileContent = await readFileAsText(currentFile);
                fullContentForAI = `
I have attached a file named "${currentFile.name}".
Here is the content of the file:
\`\`\`
${fileContent}
\`\`\`

My Question: ${text}
`;
            } catch (err) {
                console.error("Error reading file:", err);
                fullContentForAI += `\n\n[Error: Could not read file content]`;
            }
        }
        
        clearFile(); 

        // 3. Send to API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: fullContentForAI })
        });

        // FIX 2: Check if response is JSON, otherwise read text
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
             throw new Error("Server returned non-JSON response. Check your API route.");
        }

        const data = await response.json();
        
        // FIX 3: Safe error extraction
        if (data.error) {
            const errMsg = typeof data.error === 'string' ? data.error : JSON.stringify(data.error);
            throw new Error(errMsg);
        }

        const aiText = data.choices[0].message.content;
        removeLoadingIndicator(loadingId);
        await streamResponse(aiText);

    } catch (error) {
        console.error("Chat Error:", error);
        removeLoadingIndicator(loadingId);
        
        // FIX 4: Final safety net to prevent [object Object]
        let displayError = "An unknown error occurred.";
        if (error instanceof Error) {
            displayError = error.message;
        } else if (typeof error === 'string') {
            displayError = error;
        } else {
            try {
                displayError = JSON.stringify(error);
            } catch (e) {
                displayError = "Critical error (could not stringify)";
            }
        }
        
        addMessageToDom('ai', `**Error:** ${displayError}`);
    }
    
    isGenerating = false;
}

// --- Helpers ---
function setupEventListeners() {
    const sendBtn = document.getElementById('sendBtn');
    const input = document.getElementById('messageInput');
    const newChatBtn = document.getElementById('newChatBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const logoutBtn = document.getElementById('logoutBtn');

    if (logoutBtn) logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('currentUser');
        window.location.href = 'index.html';
    });

    sendBtn.addEventListener('click', handleSend);
    newChatBtn.addEventListener('click', startNewChat);
    if(clearChatBtn) clearChatBtn.addEventListener('click', startNewChat);

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
}

window.usePrompt = function(text) {
    const input = document.getElementById('messageInput');
    input.value = text;
    handleSend();
}

function startNewChat() {
    const container = document.getElementById('messagesArea');
    const emptyState = document.getElementById('emptyState');
    container.querySelectorAll('.message-wrapper').forEach(msg => msg.remove());
    emptyState.style.display = 'flex';
    clearFile();
    document.getElementById('messageInput').focus();
}

function addMessageToDom(role, contentOrHtml) {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper';
    
    const iconClass = role === 'user' ? 'user' : 'ai';
    const iconChar = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-cube"></i>';
    
    const isHtml = contentOrHtml.includes('<div') || contentOrHtml.includes('<br>');
    
    div.innerHTML = `
        <div class="role-icon ${iconClass}">${iconChar}</div>
        <div class="message-content">
            ${isHtml ? contentOrHtml : `<p>${contentOrHtml}</p>`}
        </div>
    `;
    
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function addLoadingIndicator() {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper loading-msg';
    div.id = 'loading-' + Date.now();
    div.innerHTML = `
        <div class="role-icon ai"><i class="fas fa-cube"></i></div>
        <div class="message-content">
            <div class="typing-dot" style="display:inline-block; width:8px; height:8px; background:#DA7756; border-radius:50%; animation: pulse 1s infinite;"></div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div.id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

async function streamResponse(fullText) {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper';
    div.innerHTML = `
        <div class="role-icon ai"><i class="fas fa-cube"></i></div>
        <div class="message-content"></div>
    `;
    container.appendChild(div);
    
    const contentArea = div.querySelector('.message-content');
    let currentText = "";
    const chars = fullText.split('');
    
    for (let char of chars) {
        currentText += char;
        contentArea.innerHTML = marked.parse(currentText);
        contentArea.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        scrollToBottom();
        await new Promise(r => setTimeout(r, Math.random() * 20 + 10)); 
    }
}

function addToHistory(title) { /* Mock implementation */ }
function renderHistory() { /* Mock implementation */ }
function scrollToBottom() {
    const container = document.getElementById('messagesArea');
    container.scrollTop = container.scrollHeight;
}
function autoResizeTextarea() {
    const tx = document.getElementById('messageInput');
    tx.addEventListener("input", function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}
