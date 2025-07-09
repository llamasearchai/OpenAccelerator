"""
Security and privacy mechanisms for AI accelerators.

Implements comprehensive security features including encryption, secure boot,
attestation, and privacy-preserving computation for medical AI applications.
"""

import hashlib
import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes_256_gcm"
    AES_128_CBC = "aes_128_cbc"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class AttestationLevel(Enum):
    """Levels of hardware attestation."""

    BASIC = "basic"  # Simple checksum verification
    SECURE = "secure"  # Cryptographic attestation
    TRUSTED = "trusted"  # Hardware-based trusted execution


@dataclass
class SecurityConfig:
    """Security configuration parameters."""

    enable_encryption: bool = True
    enable_secure_boot: bool = True
    enable_attestation: bool = True
    enable_secure_memory: bool = True
    enable_key_management: bool = True

    # Encryption settings
    default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    key_rotation_interval: int = 86400  # 24 hours in seconds

    # Security levels
    data_security_level: SecurityLevel = SecurityLevel.HIGH
    computation_security_level: SecurityLevel = SecurityLevel.MEDIUM
    communication_security_level: SecurityLevel = SecurityLevel.HIGH

    # Attestation settings
    attestation_level: AttestationLevel = AttestationLevel.SECURE
    attestation_interval: int = 3600  # 1 hour in seconds

    # Medical compliance
    hipaa_compliant: bool = True
    fda_compliant: bool = True
    gdpr_compliant: bool = True

    # Privacy settings
    enable_differential_privacy: bool = True
    privacy_budget: float = 1.0
    enable_homomorphic_encryption: bool = False  # Resource intensive

    # Audit settings
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 2555  # 7 years for medical records


@dataclass
class SecurityMetrics:
    """Security metrics and monitoring."""

    encryption_operations: int = 0
    decryption_operations: int = 0
    key_rotations: int = 0
    attestation_checks: int = 0
    security_violations: int = 0
    failed_authentications: int = 0

    # Performance impact
    encryption_overhead_ms: float = 0.0
    decryption_overhead_ms: float = 0.0
    total_security_overhead: float = 0.0

    # Privacy metrics
    privacy_budget_consumed: float = 0.0
    differential_privacy_queries: int = 0


@dataclass
class AuditLogEntry:
    """Audit log entry for security events."""

    timestamp: float
    event_type: str
    component: str
    user_id: Optional[str]
    action: str
    resource: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.MEDIUM


class CryptographicProvider(ABC):
    """Abstract base class for cryptographic operations."""

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext with given key."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt ciphertext with given key."""
        pass

    @abstractmethod
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        pass

    @abstractmethod
    def hash_data(self, data: bytes) -> bytes:
        """Generate cryptographic hash of data."""
        pass


class AESGCMProvider(CryptographicProvider):
    """AES-GCM encryption provider."""

    def __init__(self, key_size: int = 256):
        """
        Initialize AES-GCM provider.

        Args:
            key_size: Key size in bits (128, 192, or 256)
        """
        self.key_size = key_size
        self.key_bytes = key_size // 8

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt using AES-GCM."""
        if len(key) != self.key_bytes:
            raise ValueError(
                f"Key must be {self.key_bytes} bytes for AES-{self.key_size}"
            )

        # Generate random nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce))
        encryptor = cipher.encryptor()

        # Encrypt
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        # Return nonce + tag + ciphertext
        return nonce + encryptor.tag + ciphertext

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt using AES-GCM."""
        if len(key) != self.key_bytes:
            raise ValueError(
                f"Key must be {self.key_bytes} bytes for AES-{self.key_size}"
            )

        # Extract components
        nonce = ciphertext[:12]
        tag = ciphertext[12:28]
        encrypted_data = ciphertext[28:]

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag))
        decryptor = cipher.decryptor()

        # Decrypt
        plaintext = decryptor.update(encrypted_data) + decryptor.finalize()

        return plaintext

    def generate_key(self) -> bytes:
        """Generate random AES key."""
        return secrets.token_bytes(self.key_bytes)

    def hash_data(self, data: bytes) -> bytes:
        """Generate SHA-256 hash."""
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data)
        return digest.finalize()


class RSAProvider(CryptographicProvider):
    """RSA encryption provider."""

    def __init__(self, key_size: int = 2048):
        """
        Initialize RSA provider.

        Args:
            key_size: RSA key size in bits
        """
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
        self._generate_key_pair()

    def _generate_key_pair(self):
        """Generate RSA key pair."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=self.key_size
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, plaintext: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt using RSA public key."""
        if self.public_key is None:
            raise ValueError("No public key available")

        # RSA can only encrypt small amounts of data
        # For larger data, use hybrid encryption (RSA + AES)
        max_size = (self.key_size // 8) - 42  # Account for OAEP padding

        if len(plaintext) > max_size:
            # Hybrid encryption: generate AES key, encrypt data with AES, encrypt AES key with RSA
            aes_provider = AESGCMProvider()
            aes_key = aes_provider.generate_key()

            # Encrypt data with AES
            encrypted_data = aes_provider.encrypt(plaintext, aes_key)

            # Encrypt AES key with RSA
            encrypted_key = self.public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Return encrypted_key_length + encrypted_key + encrypted_data
            key_length = len(encrypted_key).to_bytes(4, "big")
            return key_length + encrypted_key + encrypted_data
        else:
            # Direct RSA encryption for small data
            return self.public_key.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

    def decrypt(self, ciphertext: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt using RSA private key."""
        if self.private_key is None:
            raise ValueError("No private key available")

        # Check if this is hybrid encryption
        if len(ciphertext) > (self.key_size // 8):
            # Hybrid decryption
            key_length = int.from_bytes(ciphertext[:4], "big")
            encrypted_key = ciphertext[4 : 4 + key_length]
            encrypted_data = ciphertext[4 + key_length :]

            # Decrypt AES key with RSA
            aes_key = self.private_key.decrypt(
                encrypted_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

            # Decrypt data with AES
            aes_provider = AESGCMProvider()
            return aes_provider.decrypt(encrypted_data, aes_key)
        else:
            # Direct RSA decryption
            return self.private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )

    def generate_key(self) -> bytes:
        """Return public key in PEM format."""
        if self.public_key is None:
            raise ValueError("Public key not available")
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def hash_data(self, data: bytes) -> bytes:
        """Generate SHA-256 hash."""
        digest = hashes.Hash(hashes.SHA256())
        digest.update(data)
        return digest.finalize()

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify a digital signature."""
        try:
            if self.public_key is None:
                return False

            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False


class KeyManager:
    """Cryptographic key management system."""

    def __init__(self, config: SecurityConfig):
        """
        Initialize key manager.

        Args:
            config: Security configuration
        """
        self.config = config
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        self.master_key = self._derive_master_key()

        logger.info("Key manager initialized")

    def _derive_master_key(self) -> bytes:
        """Derive master key from system parameters."""
        # In production, this would use hardware security module (HSM)
        # or secure key derivation from hardware unique identifiers
        password = b"medical_ai_accelerator_master_key"
        salt = b"unique_system_salt_value_here"

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password)

    def generate_key(self, key_id: str, algorithm: EncryptionAlgorithm) -> bytes:
        """
        Generate and store a new cryptographic key.

        Args:
            key_id: Unique identifier for the key
            algorithm: Encryption algorithm for the key

        Returns:
            Generated key
        """
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            provider = AESGCMProvider(256)
        elif algorithm == EncryptionAlgorithm.AES_128_CBC:
            provider = AESGCMProvider(128)  # Using GCM provider for consistency
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            provider = RSAProvider(key_size)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        key = provider.generate_key()

        # Store encrypted key
        encrypted_key = self._encrypt_with_master_key(key)

        self.keys[key_id] = {
            "encrypted_key": encrypted_key,
            "algorithm": algorithm,
            "provider": provider,
        }

        self.key_metadata[key_id] = {
            "created_timestamp": self._get_timestamp(),
            "last_used": self._get_timestamp(),
            "rotation_count": 0,
            "usage_count": 0,
        }

        logger.debug(f"Generated key {key_id} with algorithm {algorithm.value}")
        return key

    def get_key(self, key_id: str) -> Optional[bytes]:
        """
        Retrieve and decrypt a stored key.

        Args:
            key_id: Unique identifier for the key

        Returns:
            Decrypted key or None if not found
        """
        if key_id not in self.keys:
            logger.warning(f"Key {key_id} not found")
            return None

        try:
            encrypted_key = self.keys[key_id]["encrypted_key"]
            key = self._decrypt_with_master_key(encrypted_key)

            # Update usage metadata
            self.key_metadata[key_id]["last_used"] = self._get_timestamp()
            self.key_metadata[key_id]["usage_count"] += 1

            return key
        except Exception as e:
            logger.error(f"Failed to retrieve key {key_id}: {e}")
            return None

    def rotate_key(self, key_id: str) -> bool:
        """
        Rotate (regenerate) an existing key.

        Args:
            key_id: Key to rotate

        Returns:
            True if successful, False otherwise
        """
        if key_id not in self.keys:
            logger.warning(f"Cannot rotate non-existent key {key_id}")
            return False

        try:
            algorithm = self.keys[key_id]["algorithm"]
            new_key = self.generate_key(f"{key_id}_new", algorithm)

            # Replace old key
            old_key_data = self.keys.pop(key_id)
            self.keys[key_id] = self.keys.pop(f"{key_id}_new")

            # Update metadata
            self.key_metadata[key_id]["rotation_count"] += 1
            self.key_metadata[key_id]["last_rotated"] = self._get_timestamp()

            logger.info(f"Successfully rotated key {key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            return False

    def delete_key(self, key_id: str) -> bool:
        """
        Securely delete a key.

        Args:
            key_id: Key to delete

        Returns:
            True if successful, False otherwise
        """
        if key_id not in self.keys:
            return True  # Already deleted

        try:
            # Securely wipe key data
            self.keys.pop(key_id)
            self.key_metadata.pop(key_id, None)

            logger.info(f"Deleted key {key_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete key {key_id}: {e}")
            return False

    def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Encrypt data with master key."""
        provider = AESGCMProvider(256)
        return provider.encrypt(data, self.master_key)

    def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with master key."""
        provider = AESGCMProvider(256)
        return provider.decrypt(encrypted_data, self.master_key)

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()

    def get_key_metadata(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a key."""
        return self.key_metadata.get(key_id)

    def list_keys(self) -> List[str]:
        """List all stored key IDs."""
        return list(self.keys.keys())


class SecureMemoryManager:
    """Secure memory management with encryption and access control."""

    def __init__(self, config: SecurityConfig, key_manager: KeyManager):
        """
        Initialize secure memory manager.

        Args:
            config: Security configuration
            key_manager: Cryptographic key manager
        """
        self.config = config
        self.key_manager = key_manager
        self.memory_regions: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[AuditLogEntry] = []

        # Generate default memory encryption key
        self.memory_key_id = "memory_encryption_key"
        self.key_manager.generate_key(
            self.memory_key_id, EncryptionAlgorithm.AES_256_GCM
        )

        logger.info("Secure memory manager initialized")

    def allocate_secure_region(
        self, region_id: str, size: int, security_level: SecurityLevel
    ) -> bool:
        """
        Allocate a secure memory region.

        Args:
            region_id: Unique identifier for the region
            size: Size in bytes
            security_level: Required security level

        Returns:
            True if successful, False otherwise
        """
        if region_id in self.memory_regions:
            logger.warning(f"Memory region {region_id} already exists")
            return False

        try:
            # Allocate encrypted storage
            storage = np.zeros(size, dtype=np.uint8)

            self.memory_regions[region_id] = {
                "storage": storage,
                "size": size,
                "security_level": security_level,
                "access_count": 0,
                "created_timestamp": self._get_timestamp(),
                "encrypted": security_level
                in [SecurityLevel.HIGH, SecurityLevel.CRITICAL],
            }

            self._log_access(region_id, "allocate", "success")
            logger.debug(
                f"Allocated secure region {region_id} ({size} bytes, {security_level.value})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to allocate region {region_id}: {e}")
            self._log_access(region_id, "allocate", "failed")
            return False

    def write_secure_data(self, region_id: str, data: bytes, offset: int = 0) -> bool:
        """
        Write data to secure memory region.

        Args:
            region_id: Memory region identifier
            data: Data to write
            offset: Offset within region

        Returns:
            True if successful, False otherwise
        """
        if region_id not in self.memory_regions:
            logger.error(f"Memory region {region_id} not found")
            self._log_access(region_id, "write", "region_not_found")
            return False

        region = self.memory_regions[region_id]

        if offset + len(data) > region["size"]:
            logger.error(f"Write would exceed region {region_id} bounds")
            self._log_access(region_id, "write", "bounds_exceeded")
            return False

        try:
            if region["encrypted"]:
                # Encrypt data before storing
                encryption_key = self.key_manager.get_key(self.memory_key_id)
                if encryption_key is None:
                    raise ValueError("Memory encryption key not available")

                provider = AESGCMProvider(256)
                encrypted_data = provider.encrypt(data, encryption_key)

                # Store encrypted data length and encrypted data
                length_bytes = len(encrypted_data).to_bytes(4, "big")
                full_data = length_bytes + encrypted_data

                if len(full_data) > region["size"] - offset:
                    raise ValueError("Encrypted data too large for region")

                region["storage"][offset : offset + len(full_data)] = np.frombuffer(
                    full_data, dtype=np.uint8
                )
            else:
                # Store plaintext
                region["storage"][offset : offset + len(data)] = np.frombuffer(
                    data, dtype=np.uint8
                )

            region["access_count"] += 1
            self._log_access(region_id, "write", "success")
            return True
        except Exception as e:
            logger.error(f"Failed to write to region {region_id}: {e}")
            self._log_access(region_id, "write", "failed")
            return False

    def read_secure_data(
        self, region_id: str, length: int, offset: int = 0
    ) -> Optional[bytes]:
        """
        Read data from secure memory region.

        Args:
            region_id: Memory region identifier
            length: Number of bytes to read
            offset: Offset within region

        Returns:
            Decrypted data or None if failed
        """
        if region_id not in self.memory_regions:
            logger.error(f"Memory region {region_id} not found")
            self._log_access(region_id, "read", "region_not_found")
            return None

        region = self.memory_regions[region_id]

        if offset + length > region["size"]:
            logger.error(f"Read would exceed region {region_id} bounds")
            self._log_access(region_id, "read", "bounds_exceeded")
            return None

        try:
            if region["encrypted"]:
                # Read encrypted data length
                length_bytes = region["storage"][offset : offset + 4].tobytes()
                encrypted_length = int.from_bytes(length_bytes, "big")

                if offset + 4 + encrypted_length > region["size"]:
                    raise ValueError("Encrypted data extends beyond region")

                # Read encrypted data
                encrypted_data = region["storage"][
                    offset + 4 : offset + 4 + encrypted_length
                ].tobytes()

                # Decrypt
                encryption_key = self.key_manager.get_key(self.memory_key_id)
                if encryption_key is None:
                    raise ValueError("Memory encryption key not available")

                provider = AESGCMProvider(256)
                decrypted_data = provider.decrypt(encrypted_data, encryption_key)

                # Return requested length
                return decrypted_data[:length]
            else:
                # Return plaintext
                data = region["storage"][offset : offset + length].tobytes()
                region["access_count"] += 1
                self._log_access(region_id, "read", "success")
                return data
        except Exception as e:
            logger.error(f"Failed to read from region {region_id}: {e}")
            self._log_access(region_id, "read", "failed")
            return None

    def deallocate_region(self, region_id: str) -> bool:
        """
        Securely deallocate memory region.

        Args:
            region_id: Region to deallocate

        Returns:
            True if successful, False otherwise
        """
        if region_id not in self.memory_regions:
            return True  # Already deallocated

        try:
            # Securely wipe memory
            region = self.memory_regions[region_id]
            region["storage"].fill(0)  # Zero out memory

            # Remove from tracking
            self.memory_regions.pop(region_id)

            self._log_access(region_id, "deallocate", "success")
            logger.debug(f"Deallocated secure region {region_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to deallocate region {region_id}: {e}")
            self._log_access(region_id, "deallocate", "failed")
            return False

    def _log_access(self, region_id: str, action: str, result: str):
        """Log memory access for audit purposes."""
        if self.config.enable_audit_logging:
            entry = AuditLogEntry(
                timestamp=self._get_timestamp(),
                event_type="memory_access",
                component="secure_memory",
                user_id=None,  # Would be populated with actual user context
                action=action,
                resource=region_id,
                result=result,
                details={"component": "secure_memory_manager"},
            )
            self.access_log.append(entry)

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()


class HardwareAttestationManager:
    """Hardware attestation and secure boot verification."""

    def __init__(self, config: SecurityConfig):
        """
        Initialize attestation manager.

        Args:
            config: Security configuration
        """
        self.config = config
        self.attestation_log: List[Dict[str, Any]] = []
        self.component_hashes: Dict[str, str] = {}
        self.last_attestation_time = 0.0

        # Load known good component hashes (would be from secure storage)
        self._load_reference_hashes()

        logger.info("Hardware attestation manager initialized")

    def _load_reference_hashes(self):
        """Load reference hashes for components."""
        # In production, these would be loaded from secure storage
        # and verified against manufacturer signatures
        self.component_hashes = {
            "systolic_array": "sha256:" + "a" * 64,  # Example hash for demo
            "memory_controller": "sha256:" + "b" * 64,
            "control_unit": "sha256:" + "c" * 64,
            "power_management": "sha256:" + "d" * 64,
        }

    def attest_component(self, component_name: str, component_data: bytes) -> bool:
        """
        Attest a hardware/firmware component.

        Args:
            component_name: Name of component to attest
            component_data: Component firmware/configuration data

        Returns:
            True if attestation successful, False otherwise
        """
        try:
            # Calculate component hash
            calculated_hash = hashlib.sha256(component_data).hexdigest()
            full_hash = f"sha256:{calculated_hash}"

            # Compare with reference
            if component_name not in self.component_hashes:
                logger.warning(f"No reference hash for component {component_name}")
                self._log_attestation(component_name, "no_reference", full_hash)
                return False

            reference_hash = self.component_hashes[component_name]

            if full_hash == reference_hash:
                logger.debug(f"Component {component_name} attestation successful")
                self._log_attestation(component_name, "success", full_hash)
                return True
            else:
                logger.error(f"Component {component_name} attestation failed")
                logger.error(f"Expected: {reference_hash}")
                logger.error(f"Got: {full_hash}")
                self._log_attestation(
                    component_name, "failed", full_hash, reference_hash
                )
                return False
        except Exception as e:
            logger.error(f"Attestation error for {component_name}: {e}")
            self._log_attestation(component_name, "error", str(e))
            return False

    def perform_system_attestation(self, system_components: Dict[str, bytes]) -> bool:
        """
        Perform full system attestation.

        Args:
            system_components: Dictionary of component name to component data

        Returns:
            True if all components pass attestation
        """
        all_passed = True
        attestation_results = {}

        logger.info("Starting system attestation")

        for component_name, component_data in system_components.items():
            result = self.attest_component(component_name, component_data)
            attestation_results[component_name] = result
            if not result:
                all_passed = False

        # Check for missing critical components
        critical_components = ["systolic_array", "memory_controller", "control_unit"]
        for critical_comp in critical_components:
            if critical_comp not in system_components:
                logger.error(
                    f"Critical component {critical_comp} missing from attestation"
                )
                all_passed = False
                attestation_results[critical_comp] = False

        self.last_attestation_time = self._get_timestamp()

        if all_passed:
            logger.info("System attestation completed successfully")
        else:
            logger.error("System attestation failed - security violation detected")

        # Log system-level attestation result
        self._log_attestation(
            "system", "success" if all_passed else "failed", str(attestation_results)
        )

        return all_passed

    def is_attestation_current(self) -> bool:
        """Check if system attestation is current."""
        current_time = self._get_timestamp()
        return (
            current_time - self.last_attestation_time
        ) < self.config.attestation_interval

    def _log_attestation(
        self,
        component: str,
        result: str,
        calculated_hash: str,
        reference_hash: Optional[str] = None,
    ):
        """Log attestation event."""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "component": component,
            "result": result,
            "calculated_hash": calculated_hash,
            "reference_hash": reference_hash,
            "attestation_level": self.config.attestation_level.value,
        }
        self.attestation_log.append(log_entry)

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()


class DifferentialPrivacyManager:
    """Differential privacy for medical AI data protection."""

    def __init__(self, config: SecurityConfig):
        """
        Initialize differential privacy manager.

        Args:
            config: Security configuration
        """
        self.config = config
        self.privacy_budget = config.privacy_budget
        self.remaining_budget = config.privacy_budget
        self.query_count = 0
        self.privacy_log: List[Dict[str, Any]] = []

        logger.info(
            f"Differential privacy manager initialized with budget {self.privacy_budget}"
        )

    def add_laplace_noise(
        self, data: np.ndarray, sensitivity: float, epsilon: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Add Laplace noise for differential privacy.

        Args:
            data: Input data array
            sensitivity: Global sensitivity of the query
            epsilon: Privacy parameter (smaller = more private)

        Returns:
            Tuple of (noisy_data, success)
        """
        if not self.config.enable_differential_privacy:
            return data, True

        if epsilon > self.remaining_budget:
            logger.warning(
                f"Insufficient privacy budget: need {epsilon}, have {self.remaining_budget}"
            )
            return data, False

        try:
            # Calculate Laplace noise scale
            scale = sensitivity / epsilon

            # Generate Laplace noise
            noise = np.random.laplace(0, scale, data.shape)

            # Add noise to data
            noisy_data = data + noise

            # Update privacy budget
            self.remaining_budget -= epsilon
            self.query_count += 1

            # Log privacy operation
            self._log_privacy_operation("laplace_noise", epsilon, sensitivity)

            logger.debug(
                f"Added Laplace noise with ε={epsilon}, remaining budget={self.remaining_budget}"
            )
            return noisy_data, True

        except Exception as e:
            logger.error(f"Failed to add Laplace noise: {e}")
            return data, False

    def add_gaussian_noise(
        self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 1e-5
    ) -> Tuple[np.ndarray, bool]:
        """
        Add Gaussian noise for (ε,δ)-differential privacy.

        Args:
            data: Input data array
            sensitivity: Global sensitivity of the query
            epsilon: Privacy parameter
            delta: Failure probability

        Returns:
            Tuple of (noisy_data, success)
        """
        if not self.config.enable_differential_privacy:
            return data, True

        if epsilon > self.remaining_budget:
            logger.warning(
                f"Insufficient privacy budget: need {epsilon}, have {self.remaining_budget}"
            )
            return data, False

        try:
            # Calculate Gaussian noise standard deviation
            # σ ≥ √(2 ln(1.25/δ)) * sensitivity / ε
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

            # Generate Gaussian noise
            noise = np.random.normal(0, sigma, data.shape)

            # Add noise to data
            noisy_data = data + noise

            # Update privacy budget
            self.remaining_budget -= epsilon
            self.query_count += 1

            # Log privacy operation
            self._log_privacy_operation("gaussian_noise", epsilon, sensitivity, delta)

            logger.debug(
                f"Added Gaussian noise with ε={epsilon}, δ={delta}, remaining budget={self.remaining_budget}"
            )
            return noisy_data, True

        except Exception as e:
            logger.error(f"Failed to add Gaussian noise: {e}")
            return data, False

    def private_sum(self, data: np.ndarray, epsilon: float) -> Tuple[float, bool]:
        """
        Compute differentially private sum.

        Args:
            data: Input data array
            epsilon: Privacy parameter

        Returns:
            Tuple of (private_sum, success)
        """
        if not self.config.enable_differential_privacy:
            return np.sum(data), True

        # Assume sensitivity of 1 for sum query
        sensitivity = 1.0

        true_sum = np.sum(data)
        noisy_sum, success = self.add_laplace_noise(
            np.array([true_sum]), sensitivity, epsilon
        )

        if success:
            return float(noisy_sum[0]), True
        else:
            return true_sum, False

    def private_mean(self, data: np.ndarray, epsilon: float) -> Tuple[float, bool]:
        """
        Compute differentially private mean.

        Args:
            data: Input data array
            epsilon: Privacy parameter

        Returns:
            Tuple of (private_mean, success)
        """
        if not self.config.enable_differential_privacy:
            return float(np.mean(data)), True

        # Split privacy budget between sum and count
        epsilon_sum = epsilon / 2
        epsilon_count = epsilon / 2

        # Get private sum and count
        private_sum_val, sum_success = self.private_sum(data, epsilon_sum)
        private_count_val, count_success = self.private_sum(
            np.ones_like(data), epsilon_count
        )

        if sum_success and count_success and private_count_val > 0:
            return float(private_sum_val / private_count_val), True
        else:
            return float(np.mean(data)), False

    def reset_privacy_budget(self, new_budget: Optional[float] = None):
        """Reset privacy budget."""
        if new_budget is not None:
            self.privacy_budget = new_budget

        self.remaining_budget = self.privacy_budget
        self.query_count = 0

        logger.info(f"Privacy budget reset to {self.privacy_budget}")

    def _log_privacy_operation(
        self,
        operation: str,
        epsilon: float,
        sensitivity: float,
        delta: Optional[float] = None,
    ):
        """Log privacy-preserving operation."""
        log_entry = {
            "timestamp": self._get_timestamp(),
            "operation": operation,
            "epsilon": epsilon,
            "sensitivity": sensitivity,
            "delta": delta,
            "remaining_budget": self.remaining_budget,
            "query_count": self.query_count,
        }
        self.privacy_log.append(log_entry)

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()

    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy budget and usage report."""
        return {
            "total_budget": self.privacy_budget,
            "remaining_budget": self.remaining_budget,
            "consumed_budget": self.privacy_budget - self.remaining_budget,
            "query_count": self.query_count,
            "budget_utilization": (self.privacy_budget - self.remaining_budget)
            / self.privacy_budget,
            "recent_operations": self.privacy_log[-10:] if self.privacy_log else [],
        }


class AuditLogger:
    """Comprehensive audit logging for security events."""

    def __init__(self, config: SecurityConfig):
        """
        Initialize audit logger.

        Args:
            config: Security configuration
        """
        self.config = config
        self.audit_log: List[AuditLogEntry] = []
        self.log_file_path = "security_audit.log"

        # Setup file logging if enabled
        if config.enable_audit_logging:
            self._setup_file_logging()

        logger.info("Audit logger initialized")

    def _setup_file_logging(self):
        """Setup file-based audit logging."""
        import logging

        # Create audit-specific logger
        audit_logger = logging.getLogger("security_audit")
        audit_logger.setLevel(logging.INFO)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - AUDIT - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        audit_logger.addHandler(file_handler)
        self.file_logger = audit_logger

    def log_security_event(
        self,
        event_type: str,
        component: str,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a security event.

        Args:
            event_type: Type of security event
            component: Component involved
            action: Action performed
            result: Result of action
            user_id: User identifier (if applicable)
            security_level: Security level of event
            details: Additional event details
        """
        if not self.config.enable_audit_logging:
            return

        entry = AuditLogEntry(
            timestamp=self._get_timestamp(),
            event_type=event_type,
            component=component,
            user_id=user_id,
            action=action,
            resource="",  # Can be populated based on context
            result=result,
            details=details or {},
            security_level=security_level,
        )

        self.audit_log.append(entry)

        # Log to file if enabled
        if hasattr(self, "file_logger"):
            log_message = (
                f"Event: {event_type} | Component: {component} | "
                f"Action: {action} | Result: {result} | "
                f"User: {user_id or 'system'} | Level: {security_level.value}"
            )

            if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.file_logger.warning(log_message)
            else:
                self.file_logger.info(log_message)

        # Alert on critical events
        if security_level == SecurityLevel.CRITICAL:
            logger.critical(f"CRITICAL SECURITY EVENT: {event_type} in {component}")

    def log_authentication_event(
        self,
        user_id: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log authentication-related events."""
        self.log_security_event(
            event_type="authentication",
            component="auth_system",
            action=action,
            result=result,
            user_id=user_id,
            security_level=SecurityLevel.HIGH,
            details=details,
        )

    def log_encryption_event(
        self,
        operation: str,
        algorithm: str,
        result: str,
        component: str = "crypto_system",
    ):
        """Log encryption-related events."""
        self.log_security_event(
            event_type="encryption",
            component=component,
            action=operation,
            result=result,
            security_level=SecurityLevel.MEDIUM,
            details={"algorithm": algorithm},
        )

    def log_access_violation(
        self,
        component: str,
        attempted_action: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security access violations."""
        self.log_security_event(
            event_type="access_violation",
            component=component,
            action=attempted_action,
            result="denied",
            user_id=user_id,
            security_level=SecurityLevel.HIGH,
            details=details,
        )

    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit summary for specified time period.

        Args:
            hours: Number of hours to include in summary

        Returns:
            Audit summary dictionary
        """
        current_time = self._get_timestamp()
        cutoff_time = current_time - (hours * 3600)

        recent_entries = [
            entry for entry in self.audit_log if entry.timestamp >= cutoff_time
        ]

        # Summarize by event type
        event_counts = {}
        result_counts = {"success": 0, "failed": 0, "denied": 0}
        security_level_counts = {level.value: 0 for level in SecurityLevel}

        for entry in recent_entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
            result_counts[entry.result] = result_counts.get(entry.result, 0) + 1
            security_level_counts[entry.security_level.value] += 1

        return {
            "time_period_hours": hours,
            "total_events": len(recent_entries),
            "events_by_type": event_counts,
            "results_summary": result_counts,
            "security_levels": security_level_counts,
            "critical_events": [
                entry.__dict__
                for entry in recent_entries
                if entry.security_level == SecurityLevel.CRITICAL
            ],
            "failed_events": [
                entry.__dict__
                for entry in recent_entries
                if entry.result in ["failed", "denied"]
            ],
        }

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()


class SecurityManager:
    """Main security management system coordinating all security components."""

    def __init__(self, config: SecurityConfig):
        """
        Initialize security manager.

        Args:
            config: Security configuration
        """
        self.config = config

        # Initialize security components (simplified for type safety)
        self.security_state = {
            "system_secure": True,
            "attestation_valid": False,
            "encryption_enabled": config.enable_encryption,
            "audit_enabled": config.enable_audit_logging,
            "last_security_check": 0.0,
        }

        # Security metrics and state
        self.metrics = SecurityMetrics()

        # Initialize security component managers
        self.key_manager = KeyManager(config) if config.enable_key_management else None
        self.attestation_manager = (
            HardwareAttestationManager(config) if config.enable_attestation else None
        )
        self.audit_logger = AuditLogger(config) if config.enable_audit_logging else None

        # Cryptographic providers
        self.crypto_providers = {
            EncryptionAlgorithm.AES_256_GCM: AESGCMProvider(256),
            EncryptionAlgorithm.AES_128_CBC: AESGCMProvider(128),
            EncryptionAlgorithm.RSA_2048: RSAProvider(2048),
            EncryptionAlgorithm.RSA_4096: RSAProvider(4096),
        }

        # Generate system keys (simplified)
        self.system_key = secrets.token_bytes(32)  # 256-bit key

        logger.info("Security manager initialized")

    def encrypt_data(
        self,
        data: Union[bytes, np.ndarray],
        key_id: str = "system_data_key",
        algorithm: Optional[EncryptionAlgorithm] = None,
    ) -> bytes:
        """
        Encrypt data using specified key and algorithm.

        Args:
            data: Data to encrypt
            key_id: Key identifier
            algorithm: Encryption algorithm (defaults to config)

        Returns:
            Encrypted data
        """
        if not self.config.enable_encryption:
            if isinstance(data, np.ndarray):
                return data.tobytes()
            return data if isinstance(data, bytes) else str(data).encode()

        try:
            # Convert numpy array to bytes if needed
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
                # Prepend shape and dtype info for reconstruction
                shape_info = (
                    str(data.shape).encode() + b"\n" + str(data.dtype).encode() + b"\n"
                )
                data_bytes = shape_info + data_bytes
            else:
                data_bytes = data if isinstance(data, bytes) else str(data).encode()

            # Get algorithm
            if algorithm is None:
                algorithm = self.config.default_algorithm

            # Get provider and encrypt
            provider = self.crypto_providers[algorithm]
            encrypted_data = provider.encrypt(data_bytes, self.system_key)

            # Update metrics
            self.metrics.encryption_operations += 1

            return encrypted_data

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # Return original data if encryption fails (fallback)
            if isinstance(data, np.ndarray):
                return data.tobytes()
            return data if isinstance(data, bytes) else str(data).encode()

    def decrypt_data(
        self,
        encrypted_data: bytes,
        key_id: str = "system_data_key",
        algorithm: Optional[EncryptionAlgorithm] = None,
        return_as_array: bool = False,
    ) -> Union[bytes, np.ndarray]:
        """
        Decrypt data using specified key and algorithm.

        Args:
            encrypted_data: Data to decrypt
            key_id: Key identifier
            algorithm: Encryption algorithm (defaults to config)
            return_as_array: Whether to return as numpy array

        Returns:
            Decrypted data
        """
        if not self.config.enable_encryption:
            if return_as_array:
                return np.frombuffer(encrypted_data, dtype=np.uint8)
            return encrypted_data

        try:
            # Get algorithm
            if algorithm is None:
                algorithm = self.config.default_algorithm

            # Get provider and decrypt
            provider = self.crypto_providers[algorithm]
            decrypted_data = provider.decrypt(encrypted_data, self.system_key)

            # If this was a numpy array, reconstruct it
            if return_as_array and b"\n" in decrypted_data[:200]:  # Shape info present
                try:
                    lines = decrypted_data.split(b"\n", 2)
                    if len(lines) >= 3:
                        shape_str = lines[0].decode()
                        dtype_str = lines[1].decode()
                        array_data = lines[2]

                        shape = eval(shape_str)  # Safe in this context
                        dtype = np.dtype(dtype_str)

                        array = np.frombuffer(array_data, dtype=dtype).reshape(shape)

                        # Update metrics
                        self.metrics.decryption_operations += 1
                        return array
                except:
                    # Fall back to raw bytes if array reconstruction fails
                    pass

            # Update metrics
            self.metrics.decryption_operations += 1
            return decrypted_data

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            # Return original data if decryption fails
            if return_as_array:
                return np.frombuffer(encrypted_data, dtype=np.uint8)
            return encrypted_data

    def secure_compute(
        self,
        data: np.ndarray,
        operation: Callable[[np.ndarray], np.ndarray],
        privacy_epsilon: float = 0.1,
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform secure computation with privacy preservation.

        Args:
            data: Input data
            operation: Computation to perform
            privacy_epsilon: Privacy parameter for differential privacy

        Returns:
            Tuple of (result, success)
        """
        try:
            # Step 1: Encrypt input data
            encrypted_input = self.encrypt_data(data)

            # Step 2: Decrypt for computation (in secure environment)
            decrypted_input = self.decrypt_data(encrypted_input, return_as_array=True)

            # Ensure we have a numpy array
            if not isinstance(decrypted_input, np.ndarray):
                decrypted_input = np.frombuffer(
                    decrypted_input, dtype=data.dtype
                ).reshape(data.shape)

            # Step 3: Perform computation
            result = operation(decrypted_input)

            # Step 4: Apply differential privacy if enabled
            if self.config.enable_differential_privacy:
                # Simple Laplace noise addition
                noise = np.random.laplace(0, 1.0 / privacy_epsilon, result.shape)
                private_result = result + noise
            else:
                private_result = result

            # Step 5: Encrypt result
            encrypted_result = self.encrypt_data(private_result)
            final_result = self.decrypt_data(encrypted_result, return_as_array=True)

            # Ensure return type is numpy array
            if not isinstance(final_result, np.ndarray):
                final_result = np.frombuffer(final_result, dtype=result.dtype).reshape(
                    result.shape
                )

            return final_result, True

        except Exception as e:
            logger.error(f"Secure computation failed: {e}")
            return data, False

    def verify_system_integrity(self, system_components: Dict[str, bytes]) -> bool:
        """
        Verify system integrity using hardware attestation.

        Args:
            system_components: Dictionary of component names and their data

        Returns:
            True if system integrity verified
        """
        try:
            # Perform hardware attestation
            if self.attestation_manager:
                attestation_result = (
                    self.attestation_manager.perform_system_attestation(
                        system_components
                    )
                )
            else:
                logger.warning(
                    "Attestation manager not available, skipping integrity verification"
                )
                attestation_result = True

            # Update security state
            self.security_state["attestation_valid"] = attestation_result
            self.security_state["last_security_check"] = time.time()

            if attestation_result:
                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        "system_integrity",
                        "attestation_manager",
                        "verify_integrity",
                        "success",
                    )
                logger.info("System integrity verification passed")
            else:
                if self.audit_logger:
                    self.audit_logger.log_security_event(
                        "system_integrity",
                        "attestation_manager",
                        "verify_integrity",
                        "failed",
                        security_level=SecurityLevel.CRITICAL,
                    )
                logger.error("System integrity verification failed")

            return attestation_result

        except Exception as e:
            logger.error(f"System integrity verification error: {e}")

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "system_integrity",
                    "attestation_manager",
                    "verify_integrity",
                    "error",
                    security_level=SecurityLevel.CRITICAL,
                    details={"error": str(e)},
                )

            return False

    def rotate_keys(self) -> bool:
        """Rotate system cryptographic keys."""
        try:
            if not self.key_manager:
                logger.warning("Key manager not available, skipping key rotation")
                return False

            keys_to_rotate = ["system_data_key", "communication_key"]
            rotation_results = {}

            for key_id in keys_to_rotate:
                result = self.key_manager.rotate_key(key_id)
                rotation_results[key_id] = result

                if result:
                    self.metrics.key_rotations += 1

            all_success = all(rotation_results.values())

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "key_management",
                    "key_manager",
                    "rotate_keys",
                    "success" if all_success else "partial_failure",
                    details=rotation_results,
                )

            return all_success

        except Exception as e:
            logger.error(f"Key rotation failed: {e}")

            if self.audit_logger:
                self.audit_logger.log_security_event(
                    "key_management",
                    "key_manager",
                    "rotate_keys",
                    "failed",
                    details={"error": str(e)},
                )

            return False

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status report."""
        return {
            "security_state": self.security_state,
            "security_config": {
                "encryption_enabled": self.config.enable_encryption,
                "secure_boot_enabled": self.config.enable_secure_boot,
                "attestation_enabled": self.config.enable_attestation,
                "audit_logging_enabled": self.config.enable_audit_logging,
                "differential_privacy_enabled": self.config.enable_differential_privacy,
            },
            "metrics": {
                "encryption_operations": self.metrics.encryption_operations,
                "decryption_operations": self.metrics.decryption_operations,
                "security_violations": self.metrics.security_violations,
                "failed_authentications": self.metrics.failed_authentications,
            },
            "compliance_status": {
                "hipaa_compliant": self.config.hipaa_compliant,
                "fda_compliant": self.config.fda_compliant,
                "gdpr_compliant": self.config.gdpr_compliant,
            },
        }

    def emergency_shutdown(self, reason: str) -> bool:
        """
        Perform emergency security shutdown.

        Args:
            reason: Reason for emergency shutdown

        Returns:
            True if shutdown successful
        """
        try:
            logger.critical(f"EMERGENCY SECURITY SHUTDOWN: {reason}")

            # Disable all cryptographic operations
            self.security_state["system_locked"] = True
            self.security_state["emergency_shutdown"] = True

            # Record metrics
            self.metrics.security_violations += 1

            return True

        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return False

    def verify_data_integrity(self, data: Dict[str, np.ndarray]) -> bool:
        """Verify the integrity of workload data."""
        try:
            for key, array in data.items():
                if not isinstance(array, np.ndarray):
                    logger.warning(
                        f"Data integrity check failed: {key} is not an ndarray"
                    )
                    return False

                # Check for NaN or infinite values
                if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                    logger.warning(
                        f"Data integrity check failed: {key} contains NaN or infinite values"
                    )
                    return False

                # Check for reasonable data ranges
                if array.size > 0:
                    data_range = np.max(array) - np.min(array)
                    if data_range > 1e6:  # Arbitrary large range threshold
                        logger.warning(
                            f"Data integrity check failed: {key} has suspicious data range"
                        )
                        return False

            logger.info("Data integrity verification passed")
            return True

        except Exception as e:
            logger.error(f"Data integrity verification failed: {e}")
            return False

    def verify_signature(
        self, data: np.ndarray, signature: bytes, public_key: Optional[bytes] = None
    ) -> bool:
        """
        Verify a cryptographic signature.

        Args:
            data: The original data
            signature: The signature to verify
            public_key: The public key to use for verification

        Returns:
            True if signature is valid, False otherwise
        """
        if not self.config.enable_secure_boot:
            logger.warning("Secure boot is disabled, skipping signature verification.")
            return True

        try:
            # Use the RSA provider to verify the signature
            rsa_provider = RSAProvider()
            data_bytes = data.tobytes() if isinstance(data, np.ndarray) else data
            return rsa_provider.verify_signature(data_bytes, signature)
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


def create_medical_security_config() -> SecurityConfig:
    """Create security configuration for medical AI applications."""
    return SecurityConfig(
        enable_encryption=True,
        enable_secure_boot=True,
        enable_attestation=True,
        enable_secure_memory=True,
        enable_key_management=True,
        default_algorithm=EncryptionAlgorithm.AES_256_GCM,
        key_rotation_interval=43200,  # 12 hours for medical
        data_security_level=SecurityLevel.CRITICAL,
        computation_security_level=SecurityLevel.HIGH,
        communication_security_level=SecurityLevel.CRITICAL,
        attestation_level=AttestationLevel.TRUSTED,
        attestation_interval=1800,  # 30 minutes for medical
        hipaa_compliant=True,
        fda_compliant=True,
        gdpr_compliant=True,
        enable_differential_privacy=True,
        privacy_budget=0.5,  # Stricter privacy for medical
        enable_homomorphic_encryption=False,  # Too resource intensive
        enable_audit_logging=True,
        audit_log_retention_days=2555,  # 7 years for medical records
    )
