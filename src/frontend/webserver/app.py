from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration from environment variables
WEBSERVER_HOST = os.environ.get('WEBSERVER_HOST', '0.0.0.0')
WEBSERVER_PORT = int(os.environ.get('WEBSERVER_PORT', '8080'))
GATEWAY_URL = os.environ.get('GATEWAY_URL', 'http://gateway:80')

# Path to the webfrontend directory
WEBFRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'webfrontend')
PAGES_DIR = os.path.join(WEBFRONTEND_DIR, 'pages')
STYLES_DIR = os.path.join(WEBFRONTEND_DIR, 'styles')
SRC_DIR = os.path.join(WEBFRONTEND_DIR, 'src')

# Serve HTML pages
@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory(PAGES_DIR, 'login.html')

@app.route('/login.html')
def login():
    return send_from_directory(PAGES_DIR, 'login.html')

@app.route('/register.html')
def register():
    return send_from_directory(PAGES_DIR, 'register.html')

@app.route('/home.html')
def home():
    return send_from_directory(PAGES_DIR, 'home.html')

# Serve static files
@app.route('/styles/<path:filename>')
def serve_styles(filename):
    return send_from_directory(STYLES_DIR, filename)

@app.route('/src/<path:filename>')
def serve_src(filename):
    return send_from_directory(SRC_DIR, filename)

# API Routes
@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle login requests"""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # TODO: Implement actual authentication
    # For now, accept any credentials
    if email and password:
        return jsonify({
            'success': True,
            'token': 'dummy_token_12345',
            'message': 'Login successful'
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid credentials'
        }), 401

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Handle registration requests"""
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    # TODO: Implement actual registration
    # For now, accept any registration
    if name and email and password:
        return jsonify({
            'success': True,
            'message': 'Registration successful'
        }), 200
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid registration data'
        }), 400

@app.route('/api/user/profile', methods=['GET'])
def api_user_profile():
    """Get user profile"""
    # TODO: Implement actual user profile retrieval
    return jsonify({
        'name': 'GitQuery User',
        'email': 'user@example.com'
    }), 200

@app.route('/api/chat/message', methods=['POST'])
def api_chat_message():
    """Handle chat messages"""
    data = request.get_json()
    chat_id = data.get('chatId')
    message = data.get('message')
    
    # TODO: Implement actual chat functionality
    # For now, return a simple echo response
    return jsonify({
        'success': True,
        'response': f'I received your message: "{message}". This is a placeholder response. The actual chat functionality will be connected to your backend.'
    }), 200

@app.route('/api/chat/history', methods=['GET'])
def api_chat_history():
    """Get chat history"""
    # TODO: Implement actual chat history retrieval
    return jsonify({
        'chats': []
    }), 200

@app.route('/api/chat/<chat_id>', methods=['GET'])
def api_get_chat(chat_id):
    """Get specific chat"""
    # TODO: Implement actual chat retrieval
    return jsonify({
        'id': chat_id,
        'messages': []
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'webserver'
    }), 200

# Proxy health endpoints to gateway
@app.route('/api/health', methods=['GET'])
@app.route('/api/health/<service>', methods=['GET'])
def proxy_health(service=None):
    """Proxy health check requests to the gateway"""
    try:
        # Build the gateway URL
        if service:
            url = f"{GATEWAY_URL}/api/health/{service}"
        else:
            url = f"{GATEWAY_URL}/api/health"
        
        # Forward the request to the gateway
        response = requests.get(url, timeout=5)
        
        return jsonify(response.json()), response.status_code
        
    except requests.exceptions.Timeout:
        logger.error(f"Gateway health check timeout for: {service or 'all'}")
        return jsonify({
            'status': 'error',
            'service': service or 'all',
            'error': 'Gateway timeout'
        }), 504
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to gateway for health check: {service or 'all'}")
        return jsonify({
            'status': 'error',
            'service': service or 'all',
            'error': 'Gateway unavailable'
        }), 503
        
    except Exception as e:
        logger.error(f"Health check proxy error: {e}")
        return jsonify({
            'status': 'error',
            'service': service or 'all',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = WEBSERVER_PORT
    logger.info(f"Starting webserver on {WEBSERVER_HOST}:{port}")
    logger.info(f"Gateway URL: {GATEWAY_URL}")
    app.run(host=WEBSERVER_HOST, port=port, debug=os.environ.get('FLASK_ENV') == 'development')
