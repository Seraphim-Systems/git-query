// Register functionality
document.addEventListener('DOMContentLoaded', () => {
    // Use empty string for same-origin (webserver proxies API calls to gateway)
    const API_BASE = '';
    const registerForm = document.getElementById('registerForm');
    const nameInput = document.getElementById('name');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const submitBtn = document.getElementById('submitBtn');
    const messageDiv = document.getElementById('message');
    const passwordToggle = document.getElementById('passwordToggle');
    const confirmPasswordToggle = document.getElementById('confirmPasswordToggle');
    
    // Password toggle for password field
    if (passwordToggle) {
        passwordToggle.addEventListener('click', () => {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            passwordToggle.textContent = type === 'password' ? '👁️' : '🙈';
        });
    }
    
    // Password toggle for confirm password field
    if (confirmPasswordToggle) {
        confirmPasswordToggle.addEventListener('click', () => {
            const type = confirmPasswordInput.type === 'password' ? 'text' : 'password';
            confirmPasswordInput.type = type;
            confirmPasswordToggle.textContent = type === 'password' ? '👁️' : '🙈';
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
                // Store real session info from gateway response
                localStorage.setItem('sessionId', data.session_id || ('session_' + Date.now()));
                localStorage.setItem('userId', data.user_id || emailInput.value.trim());
                localStorage.setItem('username', data.username || nameInput.value.trim());
                
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
