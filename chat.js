(function checkAuth() {
    const user = localStorage.getItem('currentUser');
    if (!user) {
        window.location.href = 'index.html';
    }
})();

// --- Global State ---
let isGenerating = false;
let currentFile = null;
let isDebugMode = false; // New state for Debug Toggle
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5 MB Limit

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupFileHandling();
    autoResizeTextarea();
    if (window.hljs) hljs.highlightAll();

    // Initialize Debug Toggle Listener
    const debugToggle = document.getElementById('debugModeToggle');
    if (debugToggle) {
        debugToggle.addEventListener('change', (e) => {
            isDebugMode = e.target.checked;
        });
    }
});

// --- File Handling Logic ---
function setupFileHandling() {
    const attachmentBtn = document.getElementById('attachmentBtn');
    const fileInput = document.getElementById('fileInput');

    if (attachmentBtn && fileInput) {
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
    const fileInput = document.getElementById('fileInput');
    const previewArea = document.getElementById('filePreviewArea');
    if (fileInput) fileInput.value = '';
    if (previewArea) previewArea.innerHTML = '';
}

function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
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

        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
             throw new Error("Server returned non-JSON response.");
        }

        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }

        let aiText = data.choices[0].message.content;

        // --- ROUTING LOGIC START ---
        // Default Identity
        let agentIdentity = { 
            name: "Team Orchestrator", 
            icon: "fa-network-wired", 
            class: "ai-default", 
            rawTag: "None" 
        };
        
        // Detect Tags
        if (aiText.includes('[[TECH]]')) {
            agentIdentity = { name: "Tech Agent", icon: "fa-code", class: "ai-tech", rawTag: "[[TECH]]" };
            aiText = aiText.replace('[[TECH]]', '').trim();
        } else if (aiText.includes('[[DATA]]')) {
            agentIdentity = { name: "Data Analyst", icon: "fa-chart-bar", class: "ai-data", rawTag: "[[DATA]]" };
            aiText = aiText.replace('[[DATA]]', '').trim();
        } else if (aiText.includes('[[DOCS]]')) {
            agentIdentity = { name: "Docs Writer", icon: "fa-book", class: "ai-docs", rawTag: "[[DOCS]]" };
            aiText = aiText.replace('[[DOCS]]', '').trim();
        } else if (aiText.includes('[[TEAM]]')) {
             // Keeps default orchestrator identity
             aiText = aiText.replace('[[TEAM]]', '').trim();
             agentIdentity.rawTag = "[[TEAM]]";
        }
        // --- ROUTING LOGIC END ---

        removeLoadingIndicator(loadingId);
        
        // 4. Render Response with Agent Badge
        await streamResponseWithAgent(aiText, agentIdentity);

    } catch (error) {
        console.error("Chat Error:", error);
        removeLoadingIndicator(loadingId);
        
        let displayError = error.message || "An unknown error occurred.";
        addMessageToDom('ai', `**Error:** ${displayError}`);
    }
    
    isGenerating = false;
}

// --- Enhanced Renderer ---
async function streamResponseWithAgent(fullText, agentInfo) {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper';
    
    // 1. Create Badge HTML
    const badgeHtml = `
        <div class="agent-badge ${agentInfo.class}">
            <i class="fas ${agentInfo.icon}"></i> ${agentInfo.name}
        </div>
    `;

    // 2. Create Debug HTML (only if enabled)
    let debugHtml = '';
    if (isDebugMode) {
        debugHtml = `
            <div class="debug-box">
                <div class="debug-header"><i class="fas fa-terminal"></i> Routing Debugger</div>
                <div class="debug-row"><span>Raw Tag:</span> <code>${agentInfo.rawTag}</code></div>
                <div class="debug-row"><span>Route Target:</span> <strong>${agentInfo.name}</strong></div>
                <div class="debug-row"><span>Status:</span> <span style="color:#22c55e">Active</span></div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="role-icon ai"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            ${debugHtml}
            ${badgeHtml}
            <div class="markdown-body"></div>
        </div>
    `;
    container.appendChild(div);
    
    const contentArea = div.querySelector('.markdown-body');
    let currentText = "";
    const chars = fullText.split('');
    
    // Stream characters
    for (let char of chars) {
        currentText += char;
        contentArea.innerHTML = marked.parse(currentText);
        
        // Syntax Highlighting
        contentArea.querySelectorAll('pre code').forEach((block) => {
            if (window.hljs) hljs.highlightElement(block);
        });
        
        scrollToBottom();
        // Dynamic typing speed
        await new Promise(r => setTimeout(r, 5)); 
    }
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

    if (sendBtn) sendBtn.addEventListener('click', handleSend);
    
    if (newChatBtn) newChatBtn.addEventListener('click', startNewChat);
    if (clearChatBtn) clearChatBtn.addEventListener('click', startNewChat);

    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });
    }
}

window.usePrompt = function(text) {
    const input = document.getElementById('messageInput');
    if (input) {
        input.value = text;
        handleSend();
    }
}

function startNewChat() {
    const container = document.getElementById('messagesArea');
    const emptyState = document.getElementById('emptyState');
    
    // Remove all message wrappers except empty state
    const messages = container.querySelectorAll('.message-wrapper');
    messages.forEach(msg => msg.remove());
    
    emptyState.style.display = 'flex';
    clearFile();
    
    const input = document.getElementById('messageInput');
    if (input) input.focus();
}

function addMessageToDom(role, contentOrHtml) {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper';
    
    const iconClass = role === 'user' ? 'user' : 'ai';
    const iconChar = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
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
        <div class="role-icon ai"><i class="fas fa-robot"></i></div>
        <div class="message-content">
            <div class="typing-dot" style="display:inline-block; width:8px; height:8px; background:#DA7756; border-radius:50%; animation: pulse 1s infinite;"></div>
            <span style="font-size:13px; color:#666; margin-left:8px;">Consulting team...</span>
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

function scrollToBottom() {
    const container = document.getElementById('messagesArea');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function autoResizeTextarea() {
    const tx = document.getElementById('messageInput');
    if (tx) {
        tx.addEventListener("input", function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
}
