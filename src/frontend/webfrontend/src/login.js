// Login functionality
document.addEventListener('DOMContentLoaded', () => {
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
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    email: emailInput.value.trim(),
                    password: passwordInput.value,
                }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Login successful! Redirecting...', 'success');
                // Create temporary session for demo (will be replaced with real session from gateway)
                const tempUsername = emailInput.value.trim().split('@')[0];
                localStorage.setItem('sessionId', 'session_' + Date.now());
                localStorage.setItem('userId', emailInput.value.trim());
                localStorage.setItem('username', tempUsername);
                // Redirect to home page
                setTimeout(() => {
                    window.location.href = '/home.html';
                }, 1000);
            } else {
                showMessage(data.message || 'Login failed. Please try again.', 'error');
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
