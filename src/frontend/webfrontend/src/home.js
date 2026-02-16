// Home page (Chat) functionality
document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const messagesContainer = document.getElementById('messages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const newChatBtn = document.getElementById('newChatBtn');
    const chatHistory = document.getElementById('chatHistory');
    const userNameDisplay = document.getElementById('userName');
    const userAvatarDisplay = document.getElementById('userAvatar');
    
    let currentChatId = null;
    let chats = [];
    
    // Check authentication
    const sessionId = localStorage.getItem('sessionId');
    const userId = localStorage.getItem('userId');
    const username = localStorage.getItem('username');
    if (!sessionId) {
        window.location.href = '/login.html';
        return;
    }
    
    // Initialize user info
    initializeUser();
    
    // Load chat history
    loadChatHistory();
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
        sendBtn.disabled = messageInput.value.trim() === '';
    });
    
    // Handle Enter key (Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (messageInput.value.trim() !== '') {
                sendMessage();
            }
        }
    });
    
    // Send button click
    sendBtn.addEventListener('click', () => {
        sendMessage();
    });
    
    // New chat button
    newChatBtn.addEventListener('click', () => {
        createNewChat();
    });
    
    // Suggestion cards click
    document.querySelectorAll('.suggestion-card').forEach(card => {
        card.addEventListener('click', () => {
            const text = card.querySelector('.suggestion-text').textContent;
            messageInput.value = text;
            messageInput.dispatchEvent(new Event('input'));
            messageInput.focus();
        });
    });
    
    // User info click (logout)
    document.querySelector('.user-info').addEventListener('click', () => {
        if (confirm('Are you sure you want to log out?')) {
            localStorage.removeItem('sessionId');
            localStorage.removeItem('userId');
            localStorage.removeItem('username');
            window.location.href = '/login.html';
        }
    });
    
    // Initialize user info
    async function initializeUser() {
        try {
            // Use username from localStorage if available
            if (username) {
                userNameDisplay.textContent = username;
                userAvatarDisplay.textContent = username.charAt(0).toUpperCase();
            } else {
                userNameDisplay.textContent = 'User';
                userAvatarDisplay.textContent = 'U';
            }
        } catch (error) {
            console.error('Error loading user profile:', error);
            userNameDisplay.textContent = 'User';
            userAvatarDisplay.textContent = 'U';
        }
    }
    
    // Create new chat
    function createNewChat() {
        currentChatId = Date.now().toString();
        messagesContainer.innerHTML = '';
        welcomeScreen.style.display = 'flex';
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;
    }
    
    // Send message
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;
        
        // Hide welcome screen
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }
        
        // Add user message to UI
        addMessageToUI(message, 'user');
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;
        
        // Create new chat if needed
        if (!currentChatId) {
            currentChatId = Date.now().toString();
        }
        
        try {
            // Send message to API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    message: message,
                    context: {
                        chatId: currentChatId
                    }
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                // Add assistant response to UI
                addMessageToUI(data.response || 'I received your message!', 'assistant');
                
                // Update chat history
                updateChatHistory(currentChatId, message);
            } else {
                addMessageToUI('Sorry, I encountered an error. Please try again.', 'assistant');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessageToUI('Sorry, I encountered an error. Please try again.', 'assistant');
        }
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Add message to UI
    function addMessageToUI(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = type === 'user' ? userAvatarDisplay.textContent : 'AI';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Load chat history
    async function loadChatHistory() {
        try {
            // For now, just use local storage
            // const response = await fetch('/api/chat/history', {
            //     credentials: 'include'
            // });
            // 
            // if (response.ok) {
            //     const data = await response.json();
            //     chats = data.chats || [];
            //     renderChatHistory();
            // }
            renderChatHistory();
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    // Render chat history
    function renderChatHistory() {
        chatHistory.innerHTML = '';
        chats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.textContent = chat.title || 'New Chat';
            chatItem.addEventListener('click', () => {
                loadChat(chat.id);
            });
            chatHistory.appendChild(chatItem);
        });
    }
    
    // Update chat history
    function updateChatHistory(chatId, firstMessage) {
        const existingChat = chats.find(c => c.id === chatId);
        if (!existingChat) {
            const newChat = {
                id: chatId,
                title: firstMessage.substring(0, 30) + (firstMessage.length > 30 ? '...' : ''),
                timestamp: Date.now()
            };
            chats.unshift(newChat);
            renderChatHistory();
        }
    }
    
    // Load specific chat
    async function loadChat(chatId) {
        try {
            // For now, just use local storage
            // const response = await fetch(`/api/chat/${chatId}`, {
            //     credentials: 'include'
            // });
            // 
            // if (response.ok) {
            //     const data = await response.json();
            currentChatId = chatId;
            messagesContainer.innerHTML = '';
            welcomeScreen.style.display = 'none';
                
            //     // Load messages
            //     data.messages.forEach(msg => {
            //         addMessageToUI(msg.content, msg.role);
            //     });
            // }
        } catch (error) {
            console.error('Error loading chat:', error);
        }
    }
});
