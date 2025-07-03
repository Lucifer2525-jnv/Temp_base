#auth.py
from datetime import datetime, timedelta
from typing import Optional, Union
import os
import json
import requests
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import msal
from dotenv import load_dotenv

load_dotenv()

from models import User, get_db
from db_utils import get_user, verify_password

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Azure AD Configuration
AZURE_AD_CLIENT_ID = os.getenv("AZURE_AD_CLIENT_ID")
AZURE_AD_CLIENT_SECRET = os.getenv("AZURE_AD_CLIENT_SECRET")
AZURE_AD_TENANT_ID = os.getenv("AZURE_AD_TENANT_ID")
AZURE_AD_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}"
AZURE_AD_REDIRECT_URI = os.getenv("AZURE_AD_REDIRECT_URI", "http://localhost:8501")
AZURE_AD_SCOPE = os.getenv("AZURE_AD_SCOPE", '["User.Read"]').replace("'", '"')


# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

def get_sso_auth_url():
    """Generate Azure AD login URL"""
    if not all([AZURE_AD_CLIENT_ID, AZURE_AD_TENANT_ID, AZURE_AD_REDIRECT_URI, AZURE_AD_SCOPE]):
        raise ValueError("Azure AD settings are not configured")

    cca = msal.ConfidentialClientApplication(
        client_id=AZURE_AD_CLIENT_ID,
        authority=AZURE_AD_AUTHORITY,
        client_credential=AZURE_AD_CLIENT_SECRET,
    )

    auth_url = cca.get_authorization_request_url(
        scopes=json.loads(AZURE_AD_SCOPE),
        redirect_uri=AZURE_AD_REDIRECT_URI
    )
    return auth_url

def get_sso_token(auth_code: str) -> Optional[dict]:
    """Get Azure AD token from authorization code"""
    if not all([AZURE_AD_CLIENT_ID, AZURE_AD_CLIENT_SECRET, AZURE_AD_TENANT_ID, AZURE_AD_REDIRECT_URI, AZURE_AD_SCOPE]):
        raise ValueError("Azure AD settings are not configured")

    cca = msal.ConfidentialClientApplication(
        client_id=AZURE_AD_CLIENT_ID,
        authority=AZURE_AD_AUTHORITY,
        client_credential=AZURE_AD_CLIENT_SECRET,
    )

    result = cca.acquire_token_by_authorization_code(
        code=auth_code,
        scopes=json.loads(AZURE_AD_SCOPE),
        redirect_uri=AZURE_AD_REDIRECT_URI
    )
    return result

def get_sso_user_info(access_token: str) -> Optional[dict]:
    """Get user info from Azure AD"""
    if not access_token:
        return None
    
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers)
    
    if response.status_code == 200:
        return response.json()
    return None

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
    # We can add user.is_active field later if needed
    return current_user


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
    # In a real implementation, we need to store invalidated tokens in a cache/database
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