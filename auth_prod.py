# auth.py
from datetime import datetime, timedelta
from typing import Optional, Union
import os
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from models import User, get_db
from db_utils import get_user, verify_password

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        raise AuthenticationError(f"Failed to create access token: {e}")

def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            return None
            
        return {"username": username, "payload": payload}
    except JWTError:
        return None

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password
    
    Args:
        db: Database session
        username: Username
        password: Plain text password
        
    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        user = get_user(db, username)
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        token: JWT token from Authorization header
        db: Database session
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify token
        token_data = verify_token(token)
        if token_data is None:
            raise credentials_exception
        
        username = token_data["username"]
        if username is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user (for future extensibility)
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Active user object
        
    Raises:
        HTTPException: If user is inactive
    """
    # For now, all users are considered active
    # You can add user.is_active field later if needed
    return current_user

def create_user_token(user: User) -> dict:
    """
    Create access token for user
    
    Args:
        user: User object
        
    Returns:
        Token response dict
    """
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
    }

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)

def check_password_strength(password: str) -> tuple[bool, str]:
    """
    Check password strength
    
    Args:
        password: Password to check
        
    Returns:
        Tuple of (is_strong, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and digit characters"
    
    return True, "Password is strong"

# Utility functions for optional features
def invalidate_token(token: str) -> bool:
    """
    Invalidate a token (for logout functionality)
    Note: This is a placeholder - in production you'd want to use a token blacklist
    
    Args:
        token: Token to invalidate
        
    Returns:
        True if successful
    """
    # In a real implementation, you'd store invalidated tokens in a cache/database
    # For now, we just return True as tokens will expire naturally
    return True

def refresh_token(token: str) -> Optional[str]:
    """
    Refresh an access token
    
    Args:
        token: Current token
        
    Returns:
        New token or None if refresh failed
    """
    token_data = verify_token(token)
    if not token_data:
        return None
    
    # Create new token with same user data
    new_token = create_access_token({"sub": token_data["username"]})
    return new_token