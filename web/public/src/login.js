// Login functionality
document.addEventListener('DOMContentLoaded', () => {
    // /api prefix — webserver proxies these routes to the gateway
    const API_BASE = '/api';
    const loginForm = document.getElementById('loginForm');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const submitBtn = document.getElementById('submitBtn');
    const messageDiv = document.getElementById('message');
    const passwordToggle = document.getElementById('passwordToggle');
    
    const EYE_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
    const EYE_OFF  = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

    // Password toggle
    if (passwordToggle) {
        passwordToggle.addEventListener('click', () => {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            passwordToggle.innerHTML = type === 'password' ? EYE_OPEN : EYE_OFF;
        });
    }
    
    // Form validation
    const validateForm = () => {
        const email = emailInput.value.trim();
        const password = passwordInput.value.trim();
        
        if (email === '' || password === '') {
            return false;
        }
        
        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            return false;
        }
        
        return true;
    };
    
    // Enable/disable submit button based on validation
    [emailInput, passwordInput].forEach(input => {
        input.addEventListener('input', () => {
            submitBtn.disabled = !validateForm();
        });
    });
    
    // Handle form submission
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!validateForm()) {
            showMessage('Please fill in all fields correctly', 'error');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Signing in...';
        
        try {
            const response = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    email: emailInput.value.trim(),
                    password: passwordInput.value,
                }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Login successful! Redirecting...', 'success');
                // Store JWT token and session info
                localStorage.setItem('token', data.token || '');
                localStorage.setItem('sessionId', data.session_id || ('session_' + Date.now()));
                localStorage.setItem('userId', data.user_id || emailInput.value.trim());
                localStorage.setItem('username', data.username || emailInput.value.trim().split('@')[0]);
                localStorage.setItem('isAdmin', data.is_admin ? 'true' : 'false');
                // Redirect to home page
                setTimeout(() => {
                    window.location.href = '/home.html';
                }, 1000);
            } else {
                showMessage(data.detail || data.message || 'Login failed. Please try again.', 'error');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Sign In';
            }
        } catch (error) {
            console.error('Login error:', error);
            showMessage('An error occurred. Please try again.', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Sign In';
        }
    });
    
    // Show message helper
    function showMessage(text, type) {
        messageDiv.textContent = text;
        messageDiv.className = `message ${type} visible`;
        setTimeout(() => {
            messageDiv.classList.remove('visible');
        }, 5000);
    }
});
