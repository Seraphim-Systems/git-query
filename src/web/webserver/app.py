from flask import Flask, send_from_directory, jsonify, request, make_response
from flask_cors import CORS
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS - allow requests from the same host on different ports (dev setup)
# In production, the webserver and gateway are on the same host, so this is same-origin
CORS(app, 
     supports_credentials=True,
     origins=["http://localhost:8080", "http://localhost:80", "http://localhost", "http://127.0.0.1:8080"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     expose_headers=["Set-Cookie"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)

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


def _proxy_to_gateway(gateway_path, method=None, forward_cookies=True):
    """Generic proxy helper: forwards the incoming Flask request to the gateway.

    Args:
        gateway_path: Path on the gateway (e.g. '/auth/login').
        method: HTTP method to use (defaults to the incoming request method).
        forward_cookies: Whether to pass the request cookies to the gateway.

    Returns:
        A Flask Response object with the gateway's reply.
    """
    url = f"{GATEWAY_URL}{gateway_path}"
    method = method or request.method

    # Build headers - forward content-type but not host
    headers = {k: v for k, v in request.headers if k.lower() not in ('host', 'content-length')}

    # Forward cookies as Cookie header so the gateway session middleware can read them
    cookies = request.cookies if forward_cookies else {}

    try:
        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=request.get_json(silent=True) if method in ('POST', 'PUT', 'PATCH') else None,
            params=request.args,
            cookies=cookies,
            timeout=60,
            allow_redirects=False,
        )
    except requests.exceptions.Timeout:
        logger.error("Gateway timeout for %s %s", method, url)
        return jsonify({'detail': 'Gateway timeout'}), 504
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to gateway for %s %s", method, url)
        return jsonify({'detail': 'Gateway unavailable'}), 503
    except Exception as e:
        logger.error("Proxy error for %s %s: %s", method, url, e)
        return jsonify({'detail': str(e)}), 500

    # Build Flask response, forwarding status and JSON body
    try:
        body = resp.json()
    except Exception:
        body = resp.text

    flask_resp = make_response(jsonify(body) if isinstance(body, (dict, list)) else body, resp.status_code)

    # Forward important headers from gateway (Set-Cookie, CORS, etc.)
    cors_headers = [
        'access-control-allow-origin',
        'access-control-allow-credentials',
        'access-control-allow-methods',
        'access-control-allow-headers',
        'access-control-expose-headers',
        'access-control-max-age',
    ]
    for header_name, header_value in resp.headers.items():
        header_lower = header_name.lower()
        if header_lower == 'set-cookie' or header_lower in cors_headers:
            flask_resp.headers.add(header_name, header_value)

    return flask_resp


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def api_login():
    """Proxy login to the gateway."""
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 204
    return _proxy_to_gateway('/auth/login')


@app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def api_register():
    """Proxy register to the gateway."""
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 204
    return _proxy_to_gateway('/auth/register')


@app.route('/api/auth/logout', methods=['POST', 'OPTIONS'])
def api_logout():
    """Proxy logout to the gateway."""
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 204
    return _proxy_to_gateway('/auth/logout')


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def api_chat():
    """Proxy chat to the gateway."""
    if request.method == 'OPTIONS':
        return '', 204
    return _proxy_to_gateway('/chat/')


# ── Recommendation / search endpoints ────────────────────────────────────────

@app.route('/api/recommend', methods=['GET', 'POST', 'OPTIONS'])
def api_recommend():
    """Proxy recommendation requests to the gateway."""
    if request.method == 'OPTIONS':
        return '', 204
    return _proxy_to_gateway('/recommend/')


@app.route('/api/recommend/feedback', methods=['POST', 'OPTIONS'])
def api_recommend_feedback():
    """Proxy recommendation feedback to the gateway."""
    if request.method == 'OPTIONS':
        return '', 204
    return _proxy_to_gateway('/recommend/feedback')


# ── User endpoints ────────────────────────────────────────────────────────────

@app.route('/api/user/profile', methods=['GET', 'OPTIONS'])
def api_user_profile():
    """Proxy user profile to the gateway."""
    if request.method == 'OPTIONS':
        return '', 204
    return _proxy_to_gateway('/user/profile')


@app.route('/api/user/preferences', methods=['GET', 'PUT', 'OPTIONS'])
def api_user_preferences():
    """Proxy user preferences to the gateway."""
    if request.method == 'OPTIONS':
        return '', 204
    return _proxy_to_gateway('/user/preferences')


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for this webserver."""
    return jsonify({
        'status': 'healthy',
        'service': 'webserver'
    }), 200


@app.route('/api/health', methods=['GET'])
@app.route('/api/health/<service>', methods=['GET'])
def proxy_health(service=None):
    """Proxy health check requests to the gateway"""
    try:
        if service:
            url = f"{GATEWAY_URL}/api/health/{service}"
        else:
            url = f"{GATEWAY_URL}/api/health"

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
