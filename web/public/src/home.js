// Home page (Chat) functionality
document.addEventListener('DOMContentLoaded', () => {
    // Use empty string for same-origin (webserver proxies API calls to gateway)
    const API_BASE = '/api';

    // ── Example repos for dev preview ────────────────────────────────────
    const EXAMPLE_REPOS = [
        {
            id: 'DVelascoHerruzo/Portfolio-Website',
            name: 'Portfolio-Website',
            owner: 'DVelascoHerruzo',
            description: 'Online portfolio and website geared towards CD Projekt RED — features custom HLSL shaders, an animated skills component, and GitHub Actions CI/CD.',
            stars: 0,
            forks: 0,
            language: 'TypeScript',
            url: 'https://github.com/DVelascoHerruzo/Portfolio-Website'
        },
        {
            id: 'DVelascoHerruzo/KarsusInitiative',
            name: 'KarsusInitiative',
            owner: 'DVelascoHerruzo',
            description: 'Full-stack D&D initiative tracker deployed on Azure — React 18 + TypeScript frontend, Azure Functions v4 API, Cosmos DB, Blob Storage, and Bicep IaC.',
            stars: 0,
            forks: 0,
            language: 'TypeScript',
            url: 'https://github.com/DVelascoHerruzo/KarsusInitiative'
        }
    ];
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
    const loadExamplesBtn = document.getElementById('loadExamplesBtn');
    
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
    let lastQuery = '';       // last search/chat query — attached to all feedback events
    let lastVariant = 'hybrid'; // last recommendation variant returned by the API
    let typingIndicatorEl = null; // reference to the AI typing bubble
    
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

    // Per-(repo,action) cooldown to prevent feedback spam skewing the model.
    // Stores the last timestamp a given action was sent for a given repo.
    const _feedbackCooldowns = {};
    const FEEDBACK_COOLDOWN_MS = 5000; // 5 s between identical (repo, action) pairs

    // Fire-and-forget interaction event — sends all five fields MLOps needs
    function sendFeedback(repoId, action, position = null, query = '', variant = 'hybrid') {
        if (!userId || !repoId) return;
        const key = `${repoId}:${action}`;
        const now = Date.now();
        if (_feedbackCooldowns[key] && now - _feedbackCooldowns[key] < FEEDBACK_COOLDOWN_MS) return;
        _feedbackCooldowns[key] = now;
        fetch(`${API_BASE}/recommend/feedback`, {
            method: 'POST',
            headers: authHeaders(),
            credentials: 'include',
            body: JSON.stringify({
                repo_id: repoId,
                action,
                query,
                variant,
                position_in_results: position,
            }),
        }).catch(() => {});
    }

    // Show animated typing dots while waiting for AI response
    function showTypingIndicator() {
        if (typingIndicatorEl) return;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'AI';
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        typingIndicatorEl = messageDiv;
    }

    function removeTypingIndicator() {
        if (typingIndicatorEl) {
            typingIndicatorEl.remove();
            typingIndicatorEl = null;
        }
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

    // Load example repos (dev preview button)
    if (loadExamplesBtn) {
        loadExamplesBtn.addEventListener('click', () => {
            welcomeScreen.style.display = 'none';
            messagesContainer.style.display = 'none';
            repoRecommendations.style.display = 'block';
            closeRepoBtn.style.display = 'flex';
            currentView = 'repos';
            displayRepos(EXAMPLE_REPOS);
        });
    }
    
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
    function addMessageWithReposToUI(text, topRepos, moreRepos, query = '', variant = 'hybrid') {
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
            topRepos.forEach((repo, i) => topGrid.appendChild(createRepoCard(repo, i, query, variant)));
            reposSection.appendChild(topGrid);

            // Remaining repos — hidden until user expands
            if (moreRepos.length > 0) {
                const moreGrid = document.createElement('div');
                moreGrid.className = 'message-repos-more';
                moreRepos.forEach((repo, i) => moreGrid.appendChild(createRepoCard(repo, topRepos.length + i, query, variant)));
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
        showTypingIndicator();

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
                let recoVariant = 'hybrid';
                if (recoRes.ok) {
                    const recoData = await recoRes.json();
                    recoVariant = recoData.variant || 'hybrid';
                    allRepos = await enrichRepoIds(recoData.recommendations || recoData.results || []);
                }

                lastQuery = message;
                lastVariant = recoVariant;
                removeTypingIndicator();
                addMessageWithReposToUI(aiResponse, allRepos.slice(0, 3), allRepos.slice(3), message, recoVariant);
                updateChatHistory(currentChatId, aiResponse);
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
                    removeTypingIndicator();
                    addMessageToUI(aiResponse, 'assistant');
                    updateChatHistory(currentChatId, aiResponse);
                } else {
                    removeTypingIndicator();
                    addMessageToUI('Sorry, I encountered an error. Please try again.', 'assistant');
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
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
                <span class="item-text">${chat.title || 'New Session'}</span>
                <div class="item-actions">
                    <button class="item-action-btn" title="Add to collection">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                    </button>
                    <button class="item-action-btn" title="Delete session">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
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
    const ICON_MOON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>`;
    const ICON_SUN  = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;

    function initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.add('light-theme');
            document.getElementById('themeIconSvg').outerHTML = ICON_SUN;
            themeToggle.querySelector('.theme-icon').innerHTML = ICON_SUN;
        }
    }

    function toggleTheme() {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');
        themeToggle.querySelector('.theme-icon').innerHTML = isLight ? ICON_SUN : ICON_MOON;
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
            const res = await fetch(`${API_BASE}/repos/lookup`, {
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

                lastQuery = query;
                lastVariant = data.variant || 'hybrid';
                
                if (repos.length > 0) {
                    displayRepos(repos, query, lastVariant);
                } else {
                    repoGrid.innerHTML = '<div style="padding: 16px; color: var(--text-secondary); text-align: center;">No repositories found for your query.</div>';
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
    
    function displayRepos(repos, query = '', variant = 'hybrid') {
        repoGrid.innerHTML = '';
        repos.forEach((repo, index) => {
            const repoCard = createRepoCard(repo, index, query, variant);
            repoGrid.appendChild(repoCard);
        });
    }
    
    function createRepoCard(repo, position = null, query = '', variant = 'hybrid') {
        const card = document.createElement('div');
        card.className = 'repo-card';
        
        const isFavorite = favorites.some(f => f.id === repo.id);

        const starSvg = `<svg viewBox="0 0 24 24" fill="${isFavorite ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`;
        const folderSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>`;
        const starStatSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>`;
        const forkSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M18 9a9 9 0 01-9 9"/></svg>`;
        const thumbsUpSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>`;
        const thumbsDownSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>`;
        const dismissSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
        
        card.innerHTML = `
            <div class="repo-card-header">
                <div>
                    <div class="repo-card-title">${repo.name}</div>
                    <div class="repo-card-owner">${repo.owner}</div>
                </div>
                <div class="repo-card-actions">
                    <button class="repo-action-btn star-btn ${isFavorite ? 'active' : ''}" data-repo-id="${repo.id}" title="Save to favorites">
                        ${starSvg}
                    </button>
                    <button class="repo-action-btn folder-btn" data-repo-id="${repo.id}" title="Add to collection">
                        ${folderSvg}
                    </button>
                </div>
            </div>
            <div class="repo-card-description">${repo.description}</div>
            <div class="repo-card-footer">
                <div class="repo-card-stats">
                    <div class="repo-card-stat">${starStatSvg} ${formatNumber(repo.stars)}</div>
                    <div class="repo-card-stat">${forkSvg} ${formatNumber(repo.forks)}</div>
                </div>
                <div class="repo-card-footer-right">
                    <div class="repo-card-language">${repo.language}</div>
                    <button class="repo-action-btn thumbs-up-btn" data-repo-id="${repo.id}" title="Relevant">
                        ${thumbsUpSvg}
                    </button>
                    <button class="repo-action-btn thumbs-down-btn" data-repo-id="${repo.id}" title="Not relevant">
                        ${thumbsDownSvg}
                    </button>
                </div>
            </div>
        `;
        
        // Add event listeners
        const starBtn = card.querySelector('.star-btn');
        const folderBtn = card.querySelector('.folder-btn');
        const thumbsUpBtn = card.querySelector('.thumbs-up-btn');
        const thumbsDownBtn = card.querySelector('.thumbs-down-btn');
        
        starBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const wasFavorite = favorites.some(f => f.id === repo.id);
            toggleFavorite(repo);
            // Unstarring an already-saved repo counts as a dismiss signal
            if (wasFavorite) sendFeedback(repo.id, 'dismiss', position, query, variant);
        });
        
        folderBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showFolderSelection(repo);
        });

        thumbsUpBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const nowActive = !thumbsUpBtn.classList.contains('active');
            thumbsUpBtn.classList.toggle('active', nowActive);
            thumbsDownBtn.classList.remove('active');
            if (nowActive) sendFeedback(repo.id, 'thumbs_up', position, query, variant);
        });

        thumbsDownBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const nowActive = !thumbsDownBtn.classList.contains('active');
            thumbsDownBtn.classList.toggle('active', nowActive);
            thumbsUpBtn.classList.remove('active');
            if (nowActive) sendFeedback(repo.id, 'thumbs_down', position, query, variant);
        });
        
        card.addEventListener('click', () => {
            sendFeedback(repo.id, 'click', position, query, variant);
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
                    <button class="item-action-btn" title="Add to collection">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                    </button>
                    <button class="item-action-btn" title="Remove from saved">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
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
            sendFeedback(repo.id, 'save', null, lastQuery, lastVariant);
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
                        <span class="folder-icon ${isExpanded ? 'expanded' : ''}">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="11" height="11"><polyline points="9 18 15 12 9 6"/></svg>
                        </span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="14" height="14" style="flex-shrink:0;opacity:0.7"><path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/></svg>
                        <span class="item-text">${folder.name} <span style="opacity:0.5;font-size:11px">(${folder.items.length})</span></span>
                    </div>
                    <div class="item-actions">
                        <button class="item-action-btn" title="Delete collection">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        </button>
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
            const iconSvg = isChat
                ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13" style="flex-shrink:0;opacity:0.7"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>`
                : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13" style="flex-shrink:0;opacity:0.7"><path d="M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z"/></svg>`;
            const name = item.title || item.name || 'Untitled';
            
            const itemContent = document.createElement('span');
            itemContent.style.display = 'flex';
            itemContent.style.alignItems = 'center';
            itemContent.style.gap = '6px';
            itemContent.innerHTML = `${iconSvg}<span>${name}</span>`;
            itemContent.style.flex = '1';
            itemContent.style.cursor = 'pointer';
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'item-action-btn';
            removeBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="13" height="13"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
            removeBtn.title = 'Remove from collection';
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
            icon: ''
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
