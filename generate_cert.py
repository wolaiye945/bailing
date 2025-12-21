import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

def generate_self_signed_cert(cert_dir="ssl"):
    if not os.path.exists(cert_dir):
        os.makedirs(cert_dir)
    
    key_path = os.path.join(cert_dir, "key.pem")
    cert_path = os.path.join(cert_dir, "cert.pem")

    # Generate key
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
    )

    # Write key
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # Generate cert
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Beijing"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Beijing"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Bailing"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        # Our certificate will be valid for 1 year
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost")]),
        critical=False,
    ).sign(key, hashes.SHA256())

    # Write cert
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificates generated: {key_path}, {cert_path}")

if __name__ == "__main__":
    generate_self_signed_cert()
