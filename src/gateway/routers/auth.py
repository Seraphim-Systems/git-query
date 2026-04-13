"""Authentication router."""

from fastapi import APIRouter, HTTPException, status, Request, Response
from pydantic import BaseModel, EmailStr
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError

from src.gateway.services.jwt_service import create_access_token

router = APIRouter()
password_hasher = PasswordHasher()


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
    token: str
    is_admin: bool = False


@router.post("/login", response_model=AuthResponse)
async def login(request: Request, response: Response, credentials: LoginRequest):
    """Login endpoint - creates a session."""
    user_service = request.app.state.user_service
    session_manager = request.app.state.session_manager

    user = await user_service.get_user_by_email(credentials.email)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Verify password against stored hash
    stored_hash = user.get("password_hash", "")
    try:
        password_hasher.verify(stored_hash, credentials.password)
    except (VerifyMismatchError, InvalidHashError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Create session
    session_id = await session_manager.create_session(
        user_id=user["user_id"],
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown"),
    )

    is_admin = bool(user.get("is_admin", False))
    token = create_access_token(user["user_id"], user["username"], is_admin)

    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,  # 24 hours
    )

    return AuthResponse(
        session_id=session_id,
        user_id=user["user_id"],
        username=user["username"],
        message="Login successful",
        token=token,
        is_admin=is_admin,
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
    existing_user = await user_service.get_user_by_email(data.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")

    # Create user with strong password hashing.
    password_hash = password_hasher.hash(data.password)

    user = await user_service.create_user(
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

    token = create_access_token(user["user_id"], user["username"], is_admin=False)

    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=86400,
    )

    return AuthResponse(
        session_id=session_id,
        user_id=user["user_id"],
        username=user["username"],
        message="Registration successful",
        token=token,
        is_admin=False,
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
