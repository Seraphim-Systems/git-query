// Home page (Chat) functionality
document.addEventListener('DOMContentLoaded', () => {
    // Use empty string for same-origin (webserver proxies API calls to gateway)
    const API_BASE = '';
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const messagesContainer = document.getElementById('messages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const repoRecommendations = document.getElementById('repoRecommendations');
    const repoGrid = document.getElementById('repoGrid');
    const newChatBtn = document.getElementById('newChatBtn');
    const newFolderBtn = document.getElementById('newFolderBtn');
    const placeRepoBtn = document.getElementById('placeRepoBtn');
    const chatHistory = document.getElementById('chatHistory');
    const favoriteRepos = document.getElementById('favoriteRepos');
    const foldersList = document.getElementById('foldersList');
    const userNameDisplay = document.getElementById('userName');
    const userAvatarDisplay = document.getElementById('userAvatar');
    const themeToggle = document.getElementById('themeToggle');
    const repoSearchInput = document.getElementById('repoSearchInput');
    const closeRepoBtn = document.getElementById('closeRepoView');
    
    // Sidebar elements
    const sidebar = document.querySelector('.sidebar');
    const resizeHandle = document.querySelector('.sidebar-resize-handle');
    
    // Modal elements
    const modalOverlay = document.getElementById('modalOverlay');
    const modalTitle = document.getElementById('modalTitle');
    const modalInput = document.getElementById('modalInput');
    const modalList = document.getElementById('modalList');
    const modalMessage = document.getElementById('modalMessage');
    const modalClose = document.getElementById('modalClose');
    const modalCancel = document.getElementById('modalCancel');
    const modalConfirm = document.getElementById('modalConfirm');
    
    let currentChatId = null;
    let chats = [];
    let favorites = [];
    let folders = [];
    let currentView = 'welcome'; // 'welcome', 'repos', 'chat'
    let searchTimeout = null;
    
    // Check authentication
    const token = localStorage.getItem('token') || localStorage.getItem('sessionId');
    const userId = localStorage.getItem('userId');
    const username = localStorage.getItem('username');
    if (!token) {
        window.location.href = '/login.html';
        return;
    }

    // Build auth headers for every API request (JWT Bearer token)
    function authHeaders(extra) {
        const headers = { 'Content-Type': 'application/json', ...extra };
        if (token) headers['Authorization'] = `Bearer ${token}`;
        return headers;
    }
    
    // Initialize
    initializeUser();
    loadChatHistory();
    loadFavoriteRepos();
    loadFolders();
    initializeTheme();
    initializeSidebarResize();
    
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
    
    // New folder button
    newFolderBtn.addEventListener('click', () => {
        createNewFolder();
    });
    
    // Place repo button
    placeRepoBtn.addEventListener('click', () => {
        placeRepoDirectly();
    });
    
    // Repo search - auto search as you type
    repoSearchInput.addEventListener('input', () => {
        clearTimeout(searchTimeout);
        const query = repoSearchInput.value.trim();
        if (query.length > 0) {
            searchTimeout = setTimeout(() => {
                searchRepos();
            }, 300); // Debounce 300ms
        }
    });
    
    repoSearchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            clearTimeout(searchTimeout);
            searchRepos();
        }
    });
    
    // Theme toggle
    themeToggle.addEventListener('click', () => {
        toggleTheme();
    });
    
    // Close repo view button
    closeRepoBtn.addEventListener('click', () => {
        backToWelcome();
    });
    
    // Modal events
    modalClose.addEventListener('click', () => {
        closeModal();
    });
    
    modalCancel.addEventListener('click', () => {
        closeModal();
    });
    
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
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
    document.querySelector('.user-info').addEventListener('click', async () => {
        const confirmed = await showConfirm('Logout', 'Are you sure you want to log out?');
        if (confirmed) {
            // Invalidate server session (fire-and-forget)
            fetch('/auth/logout', {
                method: 'POST',
                headers: authHeaders(),
                credentials: 'include',
            }).catch(() => {});
            localStorage.removeItem('token');
            localStorage.removeItem('sessionId');
            localStorage.removeItem('userId');
            localStorage.removeItem('username');
            localStorage.removeItem('isAdmin');
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
        // Auto-save current chat if it has messages and hasn't been saved yet
        if (currentChatId && messagesContainer.children.length > 0) {
            const existingChat = chats.find(c => c.id === currentChatId);
            if (!existingChat) {
                // Chat not saved yet, save it automatically
                const chat = {
                    id: currentChatId,
                    title: 'New Chat',
                    timestamp: Date.now(),
                    messages: []
                };
                
                // Get all messages from the DOM
                const messageElements = messagesContainer.querySelectorAll('.message');
                messageElements.forEach(msgEl => {
                    const role = msgEl.classList.contains('user') ? 'user' : 'assistant';
                    const content = msgEl.querySelector('.message-content').textContent;
                    chat.messages.push({ content, role, timestamp: Date.now() });
                });
                
                // Use first AI response as title
                const firstAIMessage = chat.messages.find(m => m.role === 'assistant');
                if (firstAIMessage) {
                    chat.title = firstAIMessage.content.substring(0, 30) + (firstAIMessage.content.length > 30 ? '...' : '');
                }
                
                chats.unshift(chat);
                localStorage.setItem('chats', JSON.stringify(chats));
                renderChatHistory();
            }
        }
        
        // Create new chat
        currentChatId = Date.now().toString();
        messagesContainer.innerHTML = '';
        messagesContainer.style.display = 'none';
        welcomeScreen.style.display = 'flex';
        repoRecommendations.style.display = 'none';
        closeRepoBtn.style.display = 'none';
        currentView = 'welcome';
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;
    }
    
    // Detect if a message is asking for repo recommendations
    function isRecommendationRequest(message) {
        const lower = message.toLowerCase();
        const hasRecommendWord = /\b(recommend|suggest|find|show|give me|list|search for|looking for|can you|could you|what are)\b/.test(lower);
        const hasRepoWord = /\b(repo(s|sitories?)?|project(s)?|librar(y|ies)|package(s)?|tool(s)?|framework(s)?|resource(s)?)\b/.test(lower);
        return hasRecommendWord && hasRepoWord;
    }

    // Add AI message that includes inline repo cards
    function addMessageWithReposToUI(text, topRepos, moreRepos) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'AI';

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content';

        // Text part
        const textEl = document.createElement('div');
        textEl.textContent = text;
        contentWrapper.appendChild(textEl);

        // Top-3 repo cards
        if (topRepos.length > 0) {
            const reposSection = document.createElement('div');
            reposSection.className = 'message-repos';

            const topGrid = document.createElement('div');
            topGrid.className = 'message-repo-grid';
            topRepos.forEach(repo => topGrid.appendChild(createRepoCard(repo)));
            reposSection.appendChild(topGrid);

            // Remaining repos — hidden until user expands
            if (moreRepos.length > 0) {
                const moreGrid = document.createElement('div');
                moreGrid.className = 'message-repos-more';
                moreRepos.forEach(repo => moreGrid.appendChild(createRepoCard(repo)));
                reposSection.appendChild(moreGrid);

                const expandBtn = document.createElement('button');
                expandBtn.className = 'message-repos-expand-btn';
                expandBtn.textContent = `▼  Show ${moreRepos.length} more results`;
                let expanded = false;
                expandBtn.addEventListener('click', () => {
                    expanded = !expanded;
                    moreGrid.classList.toggle('visible', expanded);
                    expandBtn.textContent = expanded
                        ? `▲  Hide additional results`
                        : `▼  Show ${moreRepos.length} more results`;
                    if (expanded) {
                        moreGrid.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                });
                reposSection.appendChild(expandBtn);
            }

            contentWrapper.appendChild(reposSection);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentWrapper);
        messagesContainer.appendChild(messageDiv);

        // Persist text only to chat history (cards are loaded fresh each time)
        if (currentChatId) {
            saveChatMessage(currentChatId, text, 'assistant');
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Send message
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;
        
        // Switch to chat view
        welcomeScreen.style.display = 'none';
        repoRecommendations.style.display = 'none';
        messagesContainer.style.display = 'block';
        currentView = 'chat';
        
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

        const isRecoRequest = isRecommendationRequest(message);
        
        try {
            if (isRecoRequest) {
                // Fire chat + recommendation APIs in parallel
                const [chatRes, recoRes] = await Promise.all([
                    fetch(`${API_BASE}/chat/`, {
                        method: 'POST',
                        headers: authHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ message, context: { chatId: currentChatId } })
                    }),
                    fetch(`${API_BASE}/recommend/`, {
                        method: 'POST',
                        headers: authHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({
                            query: message,
                            user_id: userId || null,
                            top_k: 50,
                            enable_personalization: true,
                        })
                    })
                ]);

                const aiResponse = chatRes.ok
                    ? ((await chatRes.json()).response || 'Here are some repositories I found:')
                    : 'Here are some repositories I found:';

                let allRepos = [];
                if (recoRes.ok) {
                    const recoData = await recoRes.json();
                    allRepos = await enrichRepoIds(recoData.recommendations || recoData.results || []);
                }

                addMessageWithReposToUI(aiResponse, allRepos.slice(0, 3), allRepos.slice(3));
                updateChatHistory(currentChatId, aiResponse);

                // Fire-and-forget view signal
                if (userId) {
                    fetch(`${API_BASE}/recommend/feedback`, {
                        method: 'POST',
                        headers: authHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ repo_id: `search:${message}`, action: 'view' })
                    }).catch(() => {});
                }
            } else {
                // Regular chat — no repo results
                const response = await fetch(`${API_BASE}/chat/`, {
                    method: 'POST',
                    headers: authHeaders(),
                    credentials: 'include',
                    body: JSON.stringify({ message, context: { chatId: currentChatId } })
                });

                if (response.ok) {
                    const data = await response.json();
                    const aiResponse = data.response || 'I received your message!';
                    addMessageToUI(aiResponse, 'assistant');
                    updateChatHistory(currentChatId, aiResponse);
                } else {
                    addMessageToUI('Sorry, I encountered an error. Please try again.', 'assistant');
                }
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
        
        // Save message to chat history
        if (currentChatId) {
            saveChatMessage(currentChatId, text, type);
        }
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Save chat message
    function saveChatMessage(chatId, text, type) {
        const chat = chats.find(c => c.id === chatId);
        if (chat) {
            if (!chat.messages) {
                chat.messages = [];
            }
            chat.messages.push({ content: text, role: type, timestamp: Date.now() });
            localStorage.setItem('chats', JSON.stringify(chats));
        }
    }
    
    // Load chat history
    async function loadChatHistory() {
        try {
            const saved = localStorage.getItem('chats');
            chats = saved ? JSON.parse(saved) : [];
            renderChatHistory();
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    // Render chat history
    function renderChatHistory() {
        chatHistory.innerHTML = '';
        if (chats.length === 0) {
            chatHistory.innerHTML = '<div style="padding: 8px; font-size: 12px; color: var(--text-secondary);">No chats yet</div>';
            return;
        }
        chats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.innerHTML = `
                <span class="item-text">${chat.title || 'New Chat'}</span>
                <div class="item-actions">
                    <button class="item-action-btn" title="Add to folder">➜</button>
                    <button class="item-action-btn" title="Delete chat">✕</button>
                </div>
            `;
            
            const folderBtn = chatItem.querySelector('.item-action-btn:first-child');
            const deleteBtn = chatItem.querySelector('.item-action-btn:last-child');
            
            folderBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                showFolderSelection(chat);
            });
            
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChat(chat.id);
            });
            
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
                timestamp: Date.now(),
                messages: []
            };
            chats.unshift(newChat);
            localStorage.setItem('chats', JSON.stringify(chats));
            renderChatHistory();
        }
    }
    
    // Delete chat
    async function deleteChat(chatId) {
        const confirmed = await showConfirm('Delete Chat', 'Are you sure you want to delete this chat? This cannot be undone.');
        if (!confirmed) return;
        
        // Remove from chats array
        chats = chats.filter(c => c.id !== chatId);
        localStorage.setItem('chats', JSON.stringify(chats));
        renderChatHistory();
        
        // If the deleted chat is currently open, clear it
        if (currentChatId === chatId) {
            currentChatId = null;
            messagesContainer.innerHTML = '';
            messagesContainer.style.display = 'none';
            welcomeScreen.style.display = 'flex';
            repoRecommendations.style.display = 'none';
            closeRepoBtn.style.display = 'none';
            currentView = 'welcome';
        }
        
        // Also remove from any folders
        folders.forEach(folder => {
            folder.items = folder.items.filter(item => item.id !== chatId);
        });
        localStorage.setItem('folders', JSON.stringify(folders));
        renderFolders();
    }
    
    // Load specific chat
    async function loadChat(chatId) {
        try {
            const chat = chats.find(c => c.id === chatId);
            if (!chat) return;
            
            currentChatId = chatId;
            messagesContainer.innerHTML = '';
            welcomeScreen.style.display = 'none';
            repoRecommendations.style.display = 'none';
            messagesContainer.style.display = 'block';
            currentView = 'chat';
            
            // Load messages from saved chat
            if (chat.messages && chat.messages.length > 0) {
                chat.messages.forEach(msg => {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${msg.role}`;
                    
                    const avatar = document.createElement('div');
                    avatar.className = 'message-avatar';
                    avatar.textContent = msg.role === 'user' ? userAvatarDisplay.textContent : 'AI';
                    
                    const content = document.createElement('div');
                    content.className = 'message-content';
                    content.textContent = msg.content;
                    
                    messageDiv.appendChild(avatar);
                    messageDiv.appendChild(content);
                    messagesContainer.appendChild(messageDiv);
                });
            }
        } catch (error) {
            console.error('Error loading chat:', error);
        }
    }
    
    // Theme Management
    function initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            themeToggle.querySelector('.theme-icon').textContent = '☀️';
        }
    }
    
    function toggleTheme() {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');
        themeToggle.querySelector('.theme-icon').textContent = isLight ? '☀️' : '🌙';
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
    }
    
    // Sidebar Resize
    function initializeSidebarResize() {
        let isResizing = false;
        let startX = 0;
        let startWidth = 0;
        
        // Load saved width
        const savedWidth = localStorage.getItem('sidebarWidth');
        if (savedWidth) {
            sidebar.style.width = savedWidth + 'px';
        }
        
        resizeHandle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startWidth = sidebar.offsetWidth;
            resizeHandle.classList.add('resizing');
            document.body.style.cursor = 'ew-resize';
            document.body.style.userSelect = 'none';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const diff = e.clientX - startX;
            const newWidth = startWidth + diff;
            
            // Constrain width between min and max
            const minWidth = 200;
            const maxWidth = 500;
            const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
            
            sidebar.style.width = constrainedWidth + 'px';
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                resizeHandle.classList.remove('resizing');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                
                // Save width
                localStorage.setItem('sidebarWidth', sidebar.offsetWidth);
            }
        });
    }
    
    // Modal Utilities
    function showModal(title, placeholder = '') {
        return new Promise((resolve) => {
            // Reset all modal content
            modalInput.style.display = 'block';
            modalList.style.display = 'none';
            modalMessage.style.display = 'none';
            modalConfirm.style.display = 'inline-block';
            
            modalTitle.textContent = title;
            modalInput.placeholder = placeholder;
            modalInput.value = '';
            modalOverlay.style.display = 'flex';
            modalInput.focus();
            
            const handleConfirm = () => {
                const value = modalInput.value.trim();
                cleanup();
                resolve(value);
            };
            
            const handleCancel = () => {
                cleanup();
                resolve(null);
            };
            
            const handleKeydown = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleConfirm();
                } else if (e.key === 'Escape') {
                    handleCancel();
                }
            };
            
            const cleanup = () => {
                modalOverlay.style.display = 'none';
                modalConfirm.removeEventListener('click', handleConfirm);
                modalInput.removeEventListener('keydown', handleKeydown);
            };
            
            modalConfirm.addEventListener('click', handleConfirm);
            modalInput.addEventListener('keydown', handleKeydown);
        });
    }
    
    function showAlert(title, message) {
        return new Promise((resolve) => {
            // Reset all modal content
            modalInput.style.display = 'none';
            modalList.style.display = 'none';
            modalMessage.style.display = 'block';
            modalConfirm.style.display = 'none';
            
            modalTitle.textContent = title;
            modalMessage.textContent = message;
            modalOverlay.style.display = 'flex';
            modalCancel.textContent = 'OK';
            modalCancel.focus();
            
            const handleOk = () => {
                cleanup();
                resolve();
            };
            
            const handleKeydown = (e) => {
                if (e.key === 'Enter' || e.key === 'Escape') {
                    e.preventDefault();
                    handleOk();
                }
            };
            
            const cleanup = () => {
                modalOverlay.style.display = 'none';
                modalCancel.textContent = 'Cancel';
                document.removeEventListener('keydown', handleKeydown);
            };
            
            modalCancel.addEventListener('click', handleOk, { once: true });
            document.addEventListener('keydown', handleKeydown);
        });
    }
    
    function showConfirm(title, message) {
        return new Promise((resolve) => {
            // Reset all modal content
            modalInput.style.display = 'none';
            modalList.style.display = 'none';
            modalMessage.style.display = 'block';
            modalConfirm.style.display = 'inline-block';
            
            modalTitle.textContent = title;
            modalMessage.textContent = message;
            modalOverlay.style.display = 'flex';
            modalConfirm.textContent = 'Confirm';
            modalConfirm.focus();
            
            const handleConfirm = () => {
                cleanup();
                resolve(true);
            };
            
            const handleCancel = () => {
                cleanup();
                resolve(false);
            };
            
            const handleKeydown = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    handleConfirm();
                } else if (e.key === 'Escape') {
                    handleCancel();
                }
            };
            
            const cleanup = () => {
                modalOverlay.style.display = 'none';
                modalConfirm.textContent = 'Confirm';
                modalConfirm.removeEventListener('click', handleConfirm);
                document.removeEventListener('keydown', handleKeydown);
            };
            
            modalConfirm.addEventListener('click', handleConfirm);
            document.addEventListener('keydown', handleKeydown);
        });
    }
    
    function showListSelection(title, items) {
        return new Promise((resolve) => {
            // Reset all modal content
            modalInput.style.display = 'none';
            modalList.style.display = 'block';
            modalMessage.style.display = 'none';
            modalConfirm.style.display = 'none';
            
            modalTitle.textContent = title;
            modalList.innerHTML = '';
            modalOverlay.style.display = 'flex';
            
            items.forEach(item => {
                const listItem = document.createElement('div');
                listItem.className = 'modal-list-item';
                listItem.innerHTML = `<span>${item.icon || ''}</span><span>${item.name}</span>`;
                
                listItem.addEventListener('click', () => {
                    cleanup();
                    resolve(item);
                });
                
                modalList.appendChild(listItem);
            });
            
            const handleKeydown = (e) => {
                if (e.key === 'Escape') {
                    handleCancel();
                }
            };
            
            const handleCancel = () => {
                cleanup();
                resolve(null);
            };
            
            const cleanup = () => {
                modalOverlay.style.display = 'none';
                document.removeEventListener('keydown', handleKeydown);
            };
            
            modalCancel.addEventListener('click', handleCancel, { once: true });
            document.addEventListener('keydown', handleKeydown);
        });
    }
    
    function closeModal() {
        modalOverlay.style.display = 'none';
    }
    
    function backToWelcome() {
        welcomeScreen.style.display = 'flex';
        repoRecommendations.style.display = 'none';
        messagesContainer.style.display = 'none';
        currentView = 'welcome';
        closeRepoBtn.style.display = 'none';
        repoSearchInput.value = '';
    }
    
    // Repo Search
    // Fetch repo_ids from raw_{recommend} result, look them up in MongoDB,
    // and return fully-normalised repo objects ready for createRepoCard().
    // Falls back to local field mapping if the lookup endpoint is unavailable.
    async function enrichRepoIds(rawItems) {
        const ids = rawItems.map(r => r.repo_id || r.id).filter(Boolean);
        if (ids.length === 0) return [];
        try {
            const res = await fetch(`${API_BASE}/api/repos/lookup`, {
                method: 'POST',
                headers: authHeaders(),
                credentials: 'include',
                body: JSON.stringify({ repo_ids: ids })
            });
            if (res.ok) {
                const data = await res.json();
                if (data.repos && data.repos.length > 0) return data.repos;
            }
        } catch (e) {
            console.error('Repo enrichment lookup failed:', e);
        }
        // Fallback — use whatever the recommend API returned directly
        return rawItems.map(r => ({
            id: r.repo_id || r.id || `repo-${Math.random()}`,
            name: r.name || r.full_name?.split('/')[1] || 'Unknown',
            owner: r.full_name?.split('/')[0] || r.owner || 'Unknown',
            description: r.description || 'No description available',
            stars: r.stars || 0,
            forks: r.forks || 0,
            language: r.language || 'Unknown',
            url: r.url || `https://github.com/${r.full_name || ''}`
        }));
    }

    async function searchRepos() {
        const query = repoSearchInput.value.trim();
        if (!query) return;
        
        // Show loading state
        welcomeScreen.style.display = 'none';
        messagesContainer.style.display = 'none';
        repoRecommendations.style.display = 'block';
        closeRepoBtn.style.display = 'flex';
        currentView = 'repos';
        
        // Show loading indicator
        repoGrid.innerHTML = '<div style="padding: 16px; color: var(--text-secondary); text-align: center;">Searching repositories...</div>';
        
        try {
            const response = await fetch(`${API_BASE}/recommend/`, {
                method: 'POST',
                headers: authHeaders(),
                credentials: 'include',
                body: JSON.stringify({
                    query: query,
                    user_id: userId || null,
                    top_k: 20,
                    enable_personalization: true,
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                console.debug('[search] /api/recommend raw response:', data);
                const repos = await enrichRepoIds(data.recommendations || data.results || []);
                console.debug('[search] enriched repos:', repos);
                
                if (repos.length > 0) {
                    displayRepos(repos);
                } else {
                    repoGrid.innerHTML = '<div style="padding: 16px; color: var(--text-secondary); text-align: center;">No repositories found for your query.</div>';
                }
                
                // Record implicit view signal
                if (userId) {
                    fetch(`${API_BASE}/recommend/feedback`, {
                        method: 'POST',
                        headers: authHeaders(),
                        credentials: 'include',
                        body: JSON.stringify({ repo_id: `search:${query}`, action: 'view' })
                    }).catch(() => {}); // fire-and-forget
                }
            } else {
                console.error('Recommendation API error:', response.status);
                repoGrid.innerHTML = '<div style="padding: 16px; color: var(--text-secondary); text-align: center;">Failed to load recommendations. Please try again.</div>';
            }
        } catch (error) {
            console.error('Search error:', error);
            repoGrid.innerHTML = '<div style="padding: 16px; color: var(--text-secondary); text-align: center;">An error occurred. Please try again.</div>';
        }
    }
    
    function displayRepos(repos) {
        repoGrid.innerHTML = '';
        repos.forEach(repo => {
            const repoCard = createRepoCard(repo);
            repoGrid.appendChild(repoCard);
        });
    }
    
    function createRepoCard(repo) {
        const card = document.createElement('div');
        card.className = 'repo-card';
        
        const isFavorite = favorites.some(f => f.id === repo.id);
        
        card.innerHTML = `
            <div class="repo-card-header">
                <div>
                    <div class="repo-card-title">${repo.name}</div>
                    <div class="repo-card-owner">${repo.owner}</div>
                </div>
                <div class="repo-card-actions">
                    <button class="repo-action-btn star-btn ${isFavorite ? 'active' : ''}" data-repo-id="${repo.id}" title="Add to favorites">
                        ⭐
                    </button>
                    <button class="repo-action-btn folder-btn" data-repo-id="${repo.id}" title="Add to folder">
                        ➜
                    </button>
                </div>
            </div>
            <div class="repo-card-description">${repo.description}</div>
            <div class="repo-card-stats">
                <div class="repo-card-stat">⭐ ${formatNumber(repo.stars)}</div>
                <div class="repo-card-stat">🔄 ${formatNumber(repo.forks)}</div>
            </div>
            <div class="repo-card-language">${repo.language}</div>
        `;
        
        // Add event listeners
        const starBtn = card.querySelector('.star-btn');
        const folderBtn = card.querySelector('.folder-btn');
        
        starBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleFavorite(repo);
        });
        
        folderBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showFolderSelection(repo);
        });
        
        card.addEventListener('click', () => {
            window.open(repo.url, '_blank');
        });
        
        return card;
    }
    
    function formatNumber(num) {
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'k';
        }
        return num.toString();
    }
    
    // Favorites Management
    function loadFavoriteRepos() {
        const saved = localStorage.getItem('favoriteRepos');
        favorites = saved ? JSON.parse(saved) : [];
        renderFavoriteRepos();
    }
    
    function renderFavoriteRepos() {
        favoriteRepos.innerHTML = '';
        if (favorites.length === 0) {
            favoriteRepos.innerHTML = '<div style="padding: 8px; font-size: 12px; color: var(--text-secondary);">No favorites yet</div>';
            return;
        }
        favorites.forEach(repo => {
            const repoItem = document.createElement('div');
            repoItem.className = 'repo-item';
            repoItem.innerHTML = `
                <span class="item-text">${repo.name}</span>
                <div class="item-actions">
                    <button class="item-action-btn" title="Add to folder">➜</button>
                    <button class="item-action-btn" title="Remove from favorites">✕</button>
                </div>
            `;
            
            const folderBtn = repoItem.querySelector('.item-action-btn:first-child');
            const removeBtn = repoItem.querySelector('.item-action-btn:last-child');
            
            folderBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                showFolderSelection(repo);
            });
            
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeFavorite(repo.id);
            });
            
            repoItem.addEventListener('click', () => {
                window.open(repo.url, '_blank');
            });
            
            favoriteRepos.appendChild(repoItem);
        });
    }
    
    function toggleFavorite(repo) {
        const index = favorites.findIndex(f => f.id === repo.id);
        if (index >= 0) {
            favorites.splice(index, 1);
        } else {
            favorites.push(repo);
            // Record save interaction with recommender
            if (userId && repo.id) {
                fetch(`${API_BASE}/recommend/feedback`, {
                    method: 'POST',
                    headers: authHeaders(),
                    credentials: 'include',
                    body: JSON.stringify({ repo_id: repo.id, action: 'star' })
                }).catch(() => {});
            }
        }
        localStorage.setItem('favoriteRepos', JSON.stringify(favorites));
        renderFavoriteRepos();
        
        // Update the star button in the grid
        const starBtn = document.querySelector(`.star-btn[data-repo-id="${repo.id}"]`);
        if (starBtn) {
            starBtn.classList.toggle('active', index < 0);
        }
    }
    
    function removeFavorite(repoId) {
        favorites = favorites.filter(f => f.id !== repoId);
        localStorage.setItem('favoriteRepos', JSON.stringify(favorites));
        renderFavoriteRepos();
        
        // Update the star button in the grid
        const starBtn = document.querySelector(`.star-btn[data-repo-id="${repoId}"]`);
        if (starBtn) {
            starBtn.classList.remove('active');
        }
    }
    
    // Folders Management
    function loadFolders() {
        const saved = localStorage.getItem('folders');
        folders = saved ? JSON.parse(saved) : [];
        renderFolders();
    }
    
    function renderFolders() {
        foldersList.innerHTML = '';
        if (folders.length === 0) {
            foldersList.innerHTML = '<div style="padding: 8px; font-size: 12px; color: var(--text-secondary);">No folders yet</div>';
            return;
        }
        folders.forEach(folder => {
            const folderItem = document.createElement('div');
            folderItem.className = 'folder-item';
            
            // Check if folder is expanded
            const isExpanded = folder.expanded || false;
            
            folderItem.innerHTML = `
                <div class="folder-header">
                    <div class="folder-info">
                        <span class="folder-icon ${isExpanded ? 'expanded' : ''}">▶</span>
                        <span class="item-text">📁 ${folder.name} (${folder.items.length})</span>
                    </div>
                    <div class="item-actions">
                        <button class="item-action-btn" title="Delete folder">✕</button>
                    </div>
                </div>
                <div class="folder-content ${isExpanded ? 'expanded' : ''}"></div>
            `;
            
            const folderHeader = folderItem.querySelector('.folder-header');
            const folderContent = folderItem.querySelector('.folder-content');
            const folderIcon = folderItem.querySelector('.folder-icon');
            const deleteBtn = folderItem.querySelector('.item-action-btn');
            
            // Toggle folder expansion
            folderHeader.addEventListener('click', (e) => {
                if (e.target === deleteBtn || e.target.closest('.item-action-btn')) return;
                folder.expanded = !folder.expanded;
                folderIcon.classList.toggle('expanded');
                folderContent.classList.toggle('expanded');
                localStorage.setItem('folders', JSON.stringify(folders));
                
                // Render folder contents if expanded
                if (folder.expanded) {
                    renderFolderContents(folderContent, folder);
                }
            });
            
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteFolder(folder.id);
            });
            
            // Render contents if already expanded
            if (isExpanded) {
                renderFolderContents(folderContent, folder);
            }
            
            foldersList.appendChild(folderItem);
        });
    }
    
    function renderFolderContents(container, folder) {
        container.innerHTML = '';
        if (folder.items.length === 0) {
            container.innerHTML = '<div class="folder-child-item" style="color: var(--text-secondary); cursor: default;">Empty folder</div>';
            return;
        }
        
        folder.items.forEach(item => {
            const childItem = document.createElement('div');
            childItem.className = 'folder-child-item';
            childItem.style.display = 'flex';
            childItem.style.justifyContent = 'space-between';
            childItem.style.alignItems = 'center';
            
            // Determine item type and icon
            const isChat = item.title || item.messages;
            const icon = isChat ? '💬' : '📦';
            const name = item.title || item.name || 'Untitled';
            
            const itemContent = document.createElement('span');
            itemContent.textContent = `${icon} ${name}`;
            itemContent.style.flex = '1';
            itemContent.style.cursor = 'pointer';
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'item-action-btn';
            removeBtn.innerHTML = '✕';
            removeBtn.title = 'Remove from folder';
            removeBtn.style.opacity = '0.7';
            
            itemContent.addEventListener('click', (e) => {
                e.stopPropagation();
                if (isChat) {
                    loadChat(item.id);
                } else {
                    window.open(item.url, '_blank');
                }
            });
            
            removeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeFromFolder(folder.id, item.id);
            });
            
            childItem.appendChild(itemContent);
            childItem.appendChild(removeBtn);
            container.appendChild(childItem);
        });
    }
    
    async function createNewFolder() {
        const name = await showModal('Create New Folder', 'Enter folder name');
        if (!name || !name.trim()) return;
        
        const newFolder = {
            id: `folder-${Date.now()}`,
            name: name.trim(),
            items: []
        };
        folders.push(newFolder);
        localStorage.setItem('folders', JSON.stringify(folders));
        renderFolders();
    }
    
    async function deleteFolder(folderId) {
        const confirmed = await showConfirm('Delete Folder', 'Are you sure you want to delete this folder?');
        if (!confirmed) return;
        folders = folders.filter(f => f.id !== folderId);
        localStorage.setItem('folders', JSON.stringify(folders));
        renderFolders();
    }
    
    function removeFromFolder(folderId, itemId) {
        const folder = folders.find(f => f.id === folderId);
        if (!folder) return;
        
        folder.items = folder.items.filter(i => i.id !== itemId);
        localStorage.setItem('folders', JSON.stringify(folders));
        renderFolders();
    }
    
    async function showFolderSelection(item) {
        if (folders.length === 0) {
            await showAlert('No Folders', 'Please create a folder first!');
            return;
        }
        
        const folderItems = folders.map(f => ({
            id: f.id,
            name: f.name,
            icon: '📁'
        }));
        
        const selectedFolder = await showListSelection('Add to Folder', folderItems);
        
        if (!selectedFolder) return;
        
        addToFolder(selectedFolder.id, item);
    }
    
    function addToFolder(folderId, item) {
        const folder = folders.find(f => f.id === folderId);
        if (!folder) return;
        
        // Check if item already exists
        if (!folder.items.some(i => i.id === item.id)) {
            folder.items.push(item);
            localStorage.setItem('folders', JSON.stringify(folders));
            renderFolders();
            showAlert('Success', `Added to folder: ${folder.name}`);
        } else {
            showAlert('Duplicate', 'Item already in this folder');
        }
    }
    
    // Place Repo Directly
    async function placeRepoDirectly() {
        const url = await showModal('Add Repository', 'Enter GitHub repository URL');
        if (!url || !url.trim()) return;
        
        // Extract repo info from URL
        const match = url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
        if (!match) {
            await showAlert('Invalid URL', 'Invalid GitHub URL. Please enter a valid GitHub repository URL.');
            return;
        }
        
        const [, owner, repoName] = match;
        const repo = {
            id: `repo-${Date.now()}`,
            name: repoName,
            owner: owner,
            description: 'Manually added repository',
            stars: 0,
            forks: 0,
            language: 'Unknown',
            url: url.trim()
        };
        
        favorites.push(repo);
        localStorage.setItem('favoriteRepos', JSON.stringify(favorites));
        renderFavoriteRepos();
        await showAlert('Success', 'Repository added to favorites!');
    }
});
