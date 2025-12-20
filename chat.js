// Data: Mock History
const chatHistory = [
    { id: 1, title: "React Component Help", date: "Today" },
    { id: 2, title: "Marketing Email Draft", date: "Today" },
    { id: 3, title: "Python Data Analysis", date: "Yesterday" },
    { id: 4, title: "Book Recommendations", date: "Previous 7 Days" }
];

// Data: Mock AI Responses
// In a real app, this would come from an API (e.g., OpenAI)
const aiResponses = [
    "I can certainly help with that! Here is a **Python function** to calculate the Fibonacci sequence using recursion:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    \n    sequence = [0, 1]\n    while len(sequence) < n:\n        next_val = sequence[-1] + sequence[-2]\n        sequence.append(next_val)\n    return sequence\n\n# Example usage\nprint(fibonacci(10))\n```\n\nIs there anything else you'd like to optimize in this code?",
    "**Quantum computing** is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than on classical computers.\n\nKey concepts include:\n1. **Superposition**: Ability to be in multiple states at once.\n2. **Entanglement**: Linking particles so one affects the other instantly.",
    "Sure! Here are a few **dinner ideas** based on a healthy diet:\n\n* **Grilled Salmon** with roasted asparagus and quinoa.\n* **Chicken Stir-fry** with bell peppers, broccoli, and a ginger-soy glaze.\n* **Vegetable Curry** with chickpeas and coconut milk."
];

let isGenerating = false;

document.addEventListener('DOMContentLoaded', () => {
    renderHistory();
    setupEventListeners();
    autoResizeTextarea();
    
    // Initialize Highlight.js
    if (window.hljs) hljs.highlightAll();
});

function renderHistory() {
    const historyList = document.getElementById('historyList');
    const groups = { "Today": [], "Yesterday": [], "Previous 7 Days": [] };
    
    chatHistory.forEach(chat => {
        if(groups[chat.date]) groups[chat.date].push(chat);
    });

    let html = '';
    for (const [date, items] of Object.entries(groups)) {
        if (items.length === 0) continue;
        html += `<div class="history-group-title">${date}</div>`;
        items.forEach(item => {
            html += `<div class="history-item"><i class="far fa-message"></i> &nbsp; ${item.title}</div>`;
        });
    }
    historyList.innerHTML = html;
}

function setupEventListeners() {
    const sendBtn = document.getElementById('sendBtn');
    const input = document.getElementById('messageInput');
    const newChatBtn = document.getElementById('newChatBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');

    sendBtn.addEventListener('click', handleSend);
    newChatBtn.addEventListener('click', startNewChat);
    
    if(clearChatBtn) {
        clearChatBtn.addEventListener('click', startNewChat);
    }

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
}

// Global function for suggestion chips
window.usePrompt = function(text) {
    const input = document.getElementById('messageInput');
    input.value = text;
    // Trigger auto resize
    input.style.height = 'auto';
    input.style.height = (input.scrollHeight) + 'px';
    handleSend();
}

function startNewChat() {
    const container = document.getElementById('messagesArea');
    const emptyState = document.getElementById('emptyState');
    
    // Clear all message wrappers except empty state
    const messages = container.querySelectorAll('.message-wrapper');
    messages.forEach(msg => msg.remove());
    
    emptyState.style.display = 'flex';
    document.getElementById('messageInput').focus();
}

async function handleSend() {
    if (isGenerating) return;
    
    const input = document.getElementById('messageInput');
    const text = input.value.trim();
    if (!text) return;

    // UI Updates
    document.getElementById('emptyState').style.display = 'none';
    input.value = '';
    input.style.height = 'auto'; 
    addMessageToDom('user', text);
    addToHistory(text);
    
    // Call Vercel Backend
    isGenerating = true;
    const loadingId = addLoadingIndicator();
    
    try {
        // We fetch our own internal API route
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        
        if (data.error) throw new Error(data.error);

        const aiText = data.choices[0].message.content;
        
        removeLoadingIndicator(loadingId);
        await streamResponse(aiText);

    } catch (error) {
        console.error(error);
        removeLoadingIndicator(loadingId);
        addMessageToDom('ai', `**Error:** ${error.message}`);
    }
    
    isGenerating = false;
}

function addMessageToDom(role, content) {
    const container = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'message-wrapper';
    
    const iconClass = role === 'user' ? 'user' : 'ai';
    const iconChar = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
    
    // For user messages, simple text. For AI, we will parse markdown later in streamResponse
    div.innerHTML = `
        <div class="role-icon ${iconClass}">${iconChar}</div>
        <div class="message-content">
            <p>${content.replace(/\n/g, '<br>')}</p>
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
            <div class="typing-dot" style="display:inline-block; width:8px; height:8px; background:white; border-radius:50%; animation: pulse 1s infinite;"></div>
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
        <div class="role-icon ai"><i class="fas fa-robot"></i></div>
        <div class="message-content"></div>
    `;
    container.appendChild(div);
    
    const contentArea = div.querySelector('.message-content');
    let currentText = "";
    
    // Split into smaller chunks to simulate streaming token by token
    const chars = fullText.split('');
    
    for (let char of chars) {
        currentText += char;
        
        // Only parse markdown periodically or at end to save performance, 
        // but for this demo, we can parse mostly at the end or simple replace
        // For smoother effect, we just insert text and parse final
        
        contentArea.innerHTML = marked.parse(currentText);
        
        // Apply syntax highlighting to code blocks that are closed
        contentArea.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        scrollToBottom();
        
        // Random typing speed (10-30ms)
        await new Promise(r => setTimeout(r, Math.random() * 20 + 10)); 
    }
}

function addToHistory(title) {
    // Just prepend a visual item to the sidebar
    const historyList = document.getElementById('historyList');
    const newItem = document.createElement('div');
    newItem.className = 'history-item active';
    newItem.innerHTML = `<i class="far fa-message"></i> &nbsp; ${title.substring(0, 20)}${title.length > 20 ? '...' : ''}`;
    
    // Insert after the "Today" header
    const todayHeader = historyList.querySelector('.history-group-title');
    if (todayHeader) {
        todayHeader.after(newItem);
    }
}

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