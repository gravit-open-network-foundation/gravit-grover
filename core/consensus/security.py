"""
Consensus Security Module
=========================
Implements security mechanisms for consensus protocol.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives import serialization


@dataclass
class SecurityConfig:
    """Security configuration."""

    signature_algorithm: str = "ed25519"  # ed25519, rsa
    key_size: int = 2048
    hash_algorithm: str = "sha256"
    require_signatures: bool = True
    verify_identities: bool = True


class SignatureManager:
    """
    Manages cryptographic signatures for consensus messages.
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.private_key = None
        self.public_key = None
        self.public_keys = {}  # node_id -> public_key

    def generate_keypair(self) -> Tuple[Any, Any]:
        """
        Generate cryptographic keypair.
        """
        if self.config.signature_algorithm == "ed25519":
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
        elif self.config.signature_algorithm == "rsa":
            private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=self.config.key_size
            )
            public_key = private_key.public_key()
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.signature_algorithm}")

        self.private_key = private_key
        self.public_key = public_key

        return private_key, public_key

    def sign_message(self, message: Any) -> bytes:
        """
        Sign a message with private key.
        """
        if self.private_key is None:
            raise ValueError("Private key not set")

        # Convert message to bytes if needed
        if not isinstance(message, bytes):
            message = json.dumps(message).encode()

        # Compute hash
        digest = hashlib.sha256(message).digest()

        # Sign
        if self.config.signature_algorithm == "ed25519":
            signature = self.private_key.sign(digest)
        elif self.config.signature_algorithm == "rsa":
            from cryptography.hazmat.primitives.asymmetric import padding

            signature = self.private_key.sign(
                digest,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.signature_algorithm}")

        return signature

    def verify_message(
        self, message: Any, signature: bytes, public_key: Optional[Any] = None
    ) -> bool:
        """
        Verify message signature.
        """
        if public_key is None:
            public_key = self.public_key

        if public_key is None:
            raise ValueError("Public key not set")

        # Convert message to bytes if needed
        if not isinstance(message, bytes):
            message = json.dumps(message).encode()

        # Compute hash
        digest = hashlib.sha256(message).digest()

        # Verify
        try:
            if self.config.signature_algorithm == "ed25519":
                public_key.verify(signature, digest)
            elif self.config.signature_algorithm == "rsa":
                from cryptography.hazmat.primitives.asymmetric import padding

                public_key.verify(
                    signature,
                    digest,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256(),
                )
            else:
                raise ValueError(f"Unsupported algorithm: {self.config.signature_algorithm}")
            return True
        except Exception:
            return False

    def register_public_key(self, node_id: str, public_key: Any):
        """
        Register public key for a node.
        """
        self.public_keys[node_id] = public_key

    def get_public_key(self, node_id: str) -> Optional[Any]:
        """
        Get public key for a node.
        """
        return self.public_keys.get(node_id)


class IdentityVerifier:
    """
    Verifies node identities for consensus participation.
    """

    def __init__(self):
        self.identities = {}
        self.revoked = set()

    def register_identity(self, node_id: str, identity_data: Dict[str, Any]):
        """
        Register node identity.
        """
        self.identities[node_id] = {
            "data": identity_data,
            "registered_at": time.time(),
            "verified": False,
        }

    def verify_identity(self, node_id: str, signature: bytes, verification_key: Any) -> bool:
        """
        Verify node identity.
        """
        if node_id not in self.identities:
            return False

        if node_id in self.revoked:
            return False

        identity_data = self.identities[node_id]["data"]

        # Verify signature
        message = json.dumps(identity_data).encode()

        try:
            verification_key.verify(signature, message)
            self.identities[node_id]["verified"] = True
            return True
        except Exception:
            return False

    def revoke_identity(self, node_id: str, reason: str):
        """
        Revoke node identity.
        """
        self.revoked.add(node_id)
        self.identities[node_id]["revoked"] = True
        self.identities[node_id]["revoked_reason"] = reason
        self.identities[node_id]["revoked_at"] = time.time()

    def is_valid(self, node_id: str) -> bool:
        """
        Check if node identity is valid.
        """
        if node_id not in self.identities:
            return False

        if node_id in self.revoked:
            return False

        return self.identities[node_id]["verified"]

    def get_identity_status(self, node_id: str) -> Dict[str, Any]:
        """
        Get identity status.
        """
        if node_id not in self.identities:
            return {"status": "unknown"}

        status = {
            "status": "active" if self.is_valid(node_id) else "inactive",
            "registered_at": self.identities[node_id]["registered_at"],
            "verified": self.identities[node_id]["verified"],
        }

        if node_id in self.revoked:
            status["status"] = "revoked"
            status["revoked_at"] = self.identities[node_id]["revoked_at"]
            status["revoked_reason"] = self.identities[node_id]["revoked_reason"]

        return status


class SecureMessageHandler:
    """
    Handles secure message processing with authentication.
    """

    def __init__(
        self, node_id: str, signature_manager: SignatureManager, identity_verifier: IdentityVerifier
    ):
        self.node_id = node_id
        self.signature_manager = signature_manager
        self.identity_verifier = identity_verifier
        self.seen_messages = set()
        self.max_seen_messages = 10000

    def create_secure_message(
        self, payload: Dict[str, Any], recipient: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a signed and authenticated message.
        """
        message = {
            "sender": self.node_id,
            "timestamp": time.time(),
            "payload": payload,
            "message_id": self._generate_message_id(payload),
        }

        # Add recipient if specified
        if recipient:
            message["recipient"] = recipient

        # Sign message
        signature = self.signature_manager.sign_message(message)
        message["signature"] = signature.hex()

        return message

    def verify_secure_message(self, message: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify message authenticity and integrity.
        """
        # Check if message already seen
        message_id = message.get("message_id")
        if message_id in self.seen_messages:
            return False, "Duplicate message"

        # Add to seen messages
        self.seen_messages.add(message_id)
        if len(self.seen_messages) > self.max_seen_messages:
            # Clean old messages
            self.seen_messages = set(list(self.seen_messages)[-self.max_seen_messages // 2 :])

        # Verify sender identity
        sender = message.get("sender")
        if not sender:
            return False, "No sender"

        if not self.identity_verifier.is_valid(sender):
            return False, "Invalid sender identity"

        # Verify signature
        signature = bytes.fromhex(message.get("signature", ""))
        if not signature:
            return False, "No signature"

        # Get sender's public key
        public_key = self.signature_manager.get_public_key(sender)
        if public_key is None:
            return False, "Unknown sender public key"

        # Remove signature before verification
        message_copy = message.copy()
        message_copy.pop("signature", None)

        if not self.signature_manager.verify_message(message_copy, signature, public_key):
            return False, "Invalid signature"

        # Check timestamp (prevent replay attacks)
        timestamp = message.get("timestamp", 0)
        current_time = time.time()
        if abs(current_time - timestamp) > 300:  # 5 minutes
            return False, "Message timestamp out of sync"

        return True, "Valid message"

    def _generate_message_id(self, payload: Dict[str, Any]) -> str:
        """
        Generate unique message ID.
        """
        content = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(f"{self.node_id}_{content}_{time.time()}".encode()).hexdigest()[:16]
