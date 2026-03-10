// Login functionality
document.addEventListener('DOMContentLoaded', () => {
    // Use empty string for same-origin (webserver proxies API calls to gateway)
    const API_BASE = '';
    const loginForm = document.getElementById('loginForm');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const submitBtn = document.getElementById('submitBtn');
    const messageDiv = document.getElementById('message');
    const passwordToggle = document.getElementById('passwordToggle');
    
    // Password toggle
    if (passwordToggle) {
        passwordToggle.addEventListener('click', () => {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            passwordToggle.textContent = type === 'password' ? '👁️' : '🙈';
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
            const response = await fetch(`${API_BASE}/api/auth/login`, {
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
                // Store real session info from gateway response
                localStorage.setItem('sessionId', data.session_id || ('session_' + Date.now()));
                localStorage.setItem('userId', data.user_id || emailInput.value.trim());
                localStorage.setItem('username', data.username || emailInput.value.trim().split('@')[0]);
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
