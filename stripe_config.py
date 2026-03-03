"""
PalmPay — Stripe Payment Configuration
=======================================
Loads Stripe API keys from environment / .env file.
Provides helper functions for creating PaymentIntents.
"""

import os
import stripe
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY      = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_CURRENCY        = os.getenv("STRIPE_CURRENCY", "inr")

# Configure stripe module
stripe.api_key = STRIPE_SECRET_KEY

# Check if keys are configured
STRIPE_ENABLED = bool(STRIPE_SECRET_KEY and STRIPE_SECRET_KEY != "sk_test_REPLACE_ME")

if STRIPE_ENABLED:
    print("  ✓ Stripe: Configured  (mode: "
          f"{'live' if 'live' in STRIPE_SECRET_KEY else 'test'})")
else:
    print("  ⚠ Stripe: Not configured — set STRIPE_SECRET_KEY in .env")
    print("    Deposits will use simulated balance (no real charges)")


def create_payment_intent(amount_rupees, account_number, description=""):
    """
    Create a Stripe PaymentIntent.
    amount_rupees: float amount in INR (e.g. 500.00)
    Returns: dict with client_secret, payment_intent_id, or error.
    """
    if not STRIPE_ENABLED:
        return {"error": "Stripe not configured. Set API keys in .env"}

    try:
        # Stripe expects amount in smallest currency unit (paise for INR)
        amount_paise = int(round(amount_rupees * 100))

        intent = stripe.PaymentIntent.create(
            amount=amount_paise,
            currency=STRIPE_CURRENCY,
            metadata={
                "account_number": account_number,
                "source": "palmpay_deposit",
            },
            description=description or f"PalmPay deposit — {account_number}",
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        )

        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "amount_paise": amount_paise,
        }
    except stripe.error.StripeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Payment error: {str(e)}"}


def verify_payment_intent(payment_intent_id):
    """
    Retrieve a PaymentIntent and check if it succeeded.
    Returns: (succeeded: bool, intent_obj_or_error: dict)
    """
    if not STRIPE_ENABLED:
        return False, {"error": "Stripe not configured"}

    try:
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        return intent.status == "succeeded", {
            "id": intent.id,
            "status": intent.status,
            "amount": intent.amount,
            "currency": intent.currency,
            "metadata": dict(intent.metadata),
        }
    except stripe.error.StripeError as e:
        return False, {"error": str(e)}
    except Exception as e:
        return False, {"error": str(e)}


def construct_webhook_event(payload, sig_header):
    """
    Verify and construct a Stripe webhook event.
    Returns: (event_or_none, error_msg)
    """
    if not STRIPE_WEBHOOK_SECRET:
        return None, "Webhook secret not configured"
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
        return event, None
    except ValueError:
        return None, "Invalid payload"
    except stripe.error.SignatureVerificationError:
        return None, "Invalid signature"
