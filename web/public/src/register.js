// Register functionality
document.addEventListener('DOMContentLoaded', () => {
    // Use empty string for same-origin (webserver proxies API calls to gateway)
    const API_BASE = '/api';
    const registerForm = document.getElementById('registerForm');
    const nameInput = document.getElementById('name');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const submitBtn = document.getElementById('submitBtn');
    const messageDiv = document.getElementById('message');
    const passwordToggle = document.getElementById('passwordToggle');
    const confirmPasswordToggle = document.getElementById('confirmPasswordToggle');
    
    const EYE_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
    const EYE_OFF  = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';

    // Password toggle for password field
    if (passwordToggle) {
        passwordToggle.addEventListener('click', () => {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            passwordToggle.innerHTML = type === 'password' ? EYE_OPEN : EYE_OFF;
        });
    }
    
    // Password toggle for confirm password field
    if (confirmPasswordToggle) {
        confirmPasswordToggle.addEventListener('click', () => {
            const type = confirmPasswordInput.type === 'password' ? 'text' : 'password';
            confirmPasswordInput.type = type;
            confirmPasswordToggle.innerHTML = type === 'password' ? EYE_OPEN : EYE_OFF;
        });
    }
    
    // Form validation
    const validateForm = () => {
        const name = nameInput.value.trim();
        const email = emailInput.value.trim();
        const password = passwordInput.value;
        const confirmPassword = confirmPasswordInput.value;
        
        if (name === '' || email === '' || password === '' || confirmPassword === '') {
            return { valid: false, message: 'All fields are required' };
        }
        
        // Basic email validation
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            return { valid: false, message: 'Please enter a valid email' };
        }
        
        // Password length check
        if (password.length < 8) {
            return { valid: false, message: 'Password must be at least 8 characters' };
        }
        
        // Password match check
        if (password !== confirmPassword) {
            return { valid: false, message: 'Passwords do not match' };
        }
        
        return { valid: true };
    };
    
    // Enable/disable submit button based on validation
    [nameInput, emailInput, passwordInput, confirmPasswordInput].forEach(input => {
        input.addEventListener('input', () => {
            const validation = validateForm();
            submitBtn.disabled = !validation.valid;
        });
    });
    
    // Handle form submission
    const toBoolean = (value) => {
        if (typeof value === 'boolean') return value;
        if (typeof value === 'string') {
            const normalized = value.trim().toLowerCase();
            return normalized === 'true' || normalized === '1' || normalized === 'yes';
        }
        if (typeof value === 'number') return value === 1;
        return false;
    };

    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const validation = validateForm();
        if (!validation.valid) {
            showMessage(validation.message, 'error');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Creating account...';
        
        try {
            const response = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    username: nameInput.value.trim(),
                    email: emailInput.value.trim(),
                    password: passwordInput.value,
                }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Account created successfully! Signing you in...', 'success');
                // Store JWT token and session info
                localStorage.setItem('token', data.token || '');
                localStorage.setItem('sessionId', data.session_id || ('session_' + Date.now()));
                localStorage.setItem('userId', data.user_id || emailInput.value.trim());
                localStorage.setItem('username', data.username || nameInput.value.trim());
                localStorage.setItem('isAdmin', String(toBoolean(data.is_admin)));
                
                // Redirect to home page
                setTimeout(() => {
                    window.location.href = '/home.html';
                }, 2000);
            } else {
                showMessage(data.detail || data.message || 'Registration failed. Please try again.', 'error');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Create Account';
            }
        } catch (error) {
            console.error('Registration error:', error);
            showMessage('An error occurred. Please try again.', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Create Account';
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
