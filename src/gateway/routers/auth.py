"""Authentication router."""

from fastapi import APIRouter, HTTPException, status, Request, Response
from pydantic import BaseModel, EmailStr
import hashlib

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request model."""

    email: EmailStr
    username: str
    password: str


class AuthResponse(BaseModel):
    """Authentication response model."""

    session_id: str
    user_id: str
    username: str
    message: str


@router.post("/login", response_model=AuthResponse)
async def login(request: Request, response: Response, credentials: LoginRequest):
    """
    Login endpoint - creates a session.

    TODO: Implement proper password hashing and user verification
    """
    user_service = request.app.state.user_service
    session_manager = request.app.state.session_manager

    # TODO: Verify password against stored hash
    # For now, basic implementation
    user = await user_service.get_user(credentials.email)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )

    # Create session
    session_id = await session_manager.create_session(
        user_id=user["user_id"],
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return AuthResponse(
        session_id=session_id,
        user_id=user["user_id"],
        username=user["username"],
        message="Login successful",
    )


@router.post("/register", response_model=AuthResponse)
async def register(request: Request, response: Response, data: RegisterRequest):
    """
    Registration endpoint - creates user and session.

    TODO: Implement proper password hashing
    """
    user_service = request.app.state.user_service
    session_manager = request.app.state.session_manager

    # Check if user exists
    existing_user = await user_service.get_user(data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists"
        )

    # Create user
    # TODO: Hash password properly (bcrypt, argon2, etc.)
    password_hash = hashlib.sha256(data.password.encode()).hexdigest()

    user = await user_service.create_user(
        user_id=data.email,  # Using email as user_id for simplicity
        email=data.email,
        username=data.username,
        password_hash=password_hash,
    )

    # Create session
    session_id = await session_manager.create_session(
        user_id=user["user_id"],
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=86400,
    )

    return AuthResponse(
        session_id=session_id,
        user_id=user["user_id"],
        username=user["username"],
        message="Registration successful",
    )


@router.post("/logout")
async def logout(request: Request, response: Response):
    """Logout endpoint - deletes session."""
    session_id = request.cookies.get("session_id")

    if session_id:
        session_manager = request.app.state.session_manager
        await session_manager.delete_session(session_id)

    # Clear cookie
    response.delete_cookie("session_id")

    return {"message": "Logout successful"}
